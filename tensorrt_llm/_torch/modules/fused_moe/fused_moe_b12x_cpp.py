# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

from __future__ import annotations

from typing import Optional, Tuple

import torch

_NVFP4_BLOCK_SIZE = 16
_SCALE_STORAGE_CACHE: dict[Tuple[int, int, int, int], torch.Tensor] = {}


def _align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def convert_sf_to_mma_layout(
    sf: torch.Tensor,
    m: int,
    k: int,
    num_groups: int = 1,
    sf_vec_size: int = _NVFP4_BLOCK_SIZE,
) -> torch.Tensor:
    """Convert swizzled NVFP4 scales to the 6D MMA view used by b12x kernels."""
    sf_k = (k + sf_vec_size - 1) // sf_vec_size
    m_tiles = (m + 127) // 128
    k_tiles = (sf_k + 3) // 4
    expected_elements = num_groups * m_tiles * k_tiles * 32 * 4 * 4
    if sf.numel() != expected_elements:
        raise ValueError(
            f"Scale factor tensor has {sf.numel()} elements, expected {expected_elements} "
            f"for m={m}, k={k}, num_groups={num_groups}"
        )
    return sf.view(num_groups, m_tiles, k_tiles, 32, 4, 4).permute(3, 4, 1, 5, 2, 0)


def _scale_storage_from_mma_layout(
    scale_6d: torch.Tensor,
    *,
    rows: int,
    cols: int,
    num_groups: int,
) -> torch.Tensor:
    key = (scale_6d.data_ptr(), rows, cols, num_groups)
    cached = _SCALE_STORAGE_CACHE.get(key)
    if cached is not None:
        return cached

    sf_cols = (cols + _NVFP4_BLOCK_SIZE - 1) // _NVFP4_BLOCK_SIZE
    rows_padded = _align_up(rows, 128)
    cols_padded = _align_up(sf_cols, 4)
    storage = scale_6d.permute(5, 2, 4, 0, 1, 3).contiguous()
    storage = storage.reshape(num_groups * rows_padded, cols_padded)
    _SCALE_STORAGE_CACHE[key] = storage
    return storage


def _activation_id(activation: str) -> int:
    if activation == "silu":
        return 0
    if activation == "relu2":
        return 1
    raise ValueError(f"Unsupported B12x MoE activation: {activation}")


class B12xCppMoEWrapper:
    """Native TRT-LLM CUDA C reference backend for B12x MoE.

    This mirrors FlashInfer's ``B12xMoEWrapper.run`` signature so NemotronH can
    switch between the CuTe DSL backend and this compiled C++ backend without
    changing weight preparation or routing code.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        use_cuda_graph: bool = False,
        max_num_tokens: int = 4096,
        num_local_experts: Optional[int] = None,
        output_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
        activation: str = "silu",
    ):
        if output_dtype != torch.bfloat16:
            raise ValueError(
                f"B12x C++ MoE only supports torch.bfloat16 output, got {output_dtype}"
            )
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_local_experts = num_local_experts or num_experts
        self.output_dtype = output_dtype
        self.device = torch.device(device)
        self.activation = activation
        self.activation_id = _activation_id(activation)
        self.max_num_tokens = max_num_tokens
        self.use_cuda_graph = use_cuda_graph
        self._moe_output: Optional[torch.Tensor] = None
        self._weight_views: dict[Tuple[int, int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        if use_cuda_graph:
            self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        self._moe_output = torch.empty(
            (self.max_num_tokens, self.hidden_size),
            dtype=self.output_dtype,
            device=self.device,
        )

    def prepare_weights(
        self,
        *,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_sf: torch.Tensor,
        **_: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (
            w1_weight.data_ptr(),
            w1_weight_sf.data_ptr(),
            w2_weight.data_ptr(),
            w2_weight_sf.data_ptr(),
        )
        cached = self._weight_views.get(key)
        if cached is not None:
            return cached

        w1_rows = w1_weight.size(1)
        w2_rows = w2_weight.size(1)
        w2_cols = w2_weight.size(2) * 2
        w1_sf_storage = _scale_storage_from_mma_layout(
            w1_weight_sf,
            rows=w1_rows,
            cols=self.hidden_size,
            num_groups=self.num_local_experts,
        )
        w2_sf_storage = _scale_storage_from_mma_layout(
            w2_weight_sf,
            rows=w2_rows,
            cols=w2_cols,
            num_groups=self.num_local_experts,
        )
        self._weight_views[key] = (w1_sf_storage, w2_sf_storage)
        return w1_sf_storage, w2_sf_storage

    def _get_output(self, x: torch.Tensor) -> torch.Tensor:
        num_tokens = x.size(0)
        if self.use_cuda_graph:
            if self._moe_output is None:
                self._allocate_buffers()
            if num_tokens > self._moe_output.size(0):
                raise ValueError(
                    f"num_tokens={num_tokens} exceeds max_num_tokens={self._moe_output.size(0)} "
                    "for B12x C++ CUDA graph output buffer"
                )
            return self._moe_output[:num_tokens]
        return torch.empty((num_tokens, self.hidden_size), dtype=self.output_dtype, device=x.device)

    def run(
        self,
        x: torch.Tensor,
        w1_weight: torch.Tensor,
        w1_weight_sf: torch.Tensor,
        w2_weight: torch.Tensor,
        w2_weight_sf: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        *,
        w1_alpha: torch.Tensor,
        w2_alpha: torch.Tensor,
        fc2_input_scale: torch.Tensor,
    ) -> torch.Tensor:
        if x.dtype != torch.bfloat16:
            raise ValueError(f"B12x C++ MoE expects bf16 input, got {x.dtype}")
        if x.dim() != 2 or x.size(1) != self.hidden_size:
            raise ValueError(
                f"x must have shape [num_tokens, {self.hidden_size}], got {tuple(x.shape)}"
            )
        if token_selected_experts.shape != (x.size(0), self.top_k):
            raise ValueError("token_selected_experts must have shape [num_tokens, top_k]")
        if token_final_scales.shape != (x.size(0), self.top_k):
            raise ValueError("token_final_scales must have shape [num_tokens, top_k]")

        w1_sf_storage, w2_sf_storage = self.prepare_weights(
            w1_weight=w1_weight,
            w1_weight_sf=w1_weight_sf,
            w2_weight=w2_weight,
            w2_weight_sf=w2_weight_sf,
        )
        output = self._get_output(x)
        return torch.ops.trtllm.b12x_moe_cuda(
            x.contiguous(),
            w1_weight.contiguous(),
            w1_sf_storage,
            w1_alpha.contiguous().to(torch.float32),
            fc2_input_scale.contiguous().to(torch.float32),
            w2_weight.contiguous(),
            w2_sf_storage,
            w2_alpha.contiguous().to(torch.float32),
            token_selected_experts.contiguous(),
            token_final_scales.contiguous().to(torch.float32),
            output,
            self.num_experts,
            self.top_k,
            self.num_local_experts,
            self.activation_id,
        )
