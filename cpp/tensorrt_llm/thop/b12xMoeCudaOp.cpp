/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/kernels/b12xMoeCuda.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <optional>

#include <torch/extension.h>

using torch::Tensor;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

void checkB12xTensorInputs(Tensor const& x, Tensor const& w1Weight, Tensor const& w1SfStorage, Tensor const& w1Alpha,
    Tensor const& fc2InputScale, Tensor const& w2Weight, Tensor const& w2SfStorage, Tensor const& w2Alpha,
    Tensor const& topkIds, Tensor const& topkWeights, std::optional<Tensor> const& output, int64_t numExperts,
    int64_t topK, int64_t numLocalExperts, int64_t activation)
{
    CHECK_INPUT(x, torch::ScalarType::BFloat16);
    CHECK_INPUT(w1Weight, FLOAT4_E2M1X2);
    CHECK_INPUT(w1SfStorage, SF_DTYPE);
    CHECK_INPUT(w1Alpha, torch::ScalarType::Float);
    CHECK_INPUT(fc2InputScale, torch::ScalarType::Float);
    CHECK_INPUT(w2Weight, FLOAT4_E2M1X2);
    CHECK_INPUT(w2SfStorage, SF_DTYPE);
    CHECK_INPUT(w2Alpha, torch::ScalarType::Float);
    CHECK_INPUT(topkWeights, torch::ScalarType::Float);
    CHECK_TH_CUDA(topkIds);
    CHECK_CONTIGUOUS(topkIds);
    TORCH_CHECK(topkIds.scalar_type() == torch::ScalarType::Int || topkIds.scalar_type() == torch::ScalarType::Long,
        "topk_ids must be int32 or int64");
    CHECK_OPTIONAL_INPUT(output, torch::ScalarType::BFloat16);

    TORCH_CHECK(x.dim() == 2, "x must be [num_tokens, hidden_size]");
    TORCH_CHECK(w1Weight.dim() == 3, "w1_weight must be [num_local_experts, rows, hidden_size / 2]");
    TORCH_CHECK(w2Weight.dim() == 3, "w2_weight must be [num_local_experts, hidden_size, intermediate_size / 2]");
    TORCH_CHECK(topkIds.dim() == 2, "topk_ids must be [num_tokens, top_k]");
    TORCH_CHECK(topkWeights.dim() == 2, "topk_weights must be [num_tokens, top_k]");
    TORCH_CHECK(activation == static_cast<int64_t>(tensorrt_llm::kernels::B12xMoeActivation::kSilu)
            || activation == static_cast<int64_t>(tensorrt_llm::kernels::B12xMoeActivation::kRelu2),
        "activation must be 0 (silu) or 1 (relu2)");

    auto const numTokens = x.size(0);
    auto const hiddenSize = x.size(1);
    auto const w1Rows = w1Weight.size(1);
    auto const intermediateSize
        = activation == static_cast<int64_t>(tensorrt_llm::kernels::B12xMoeActivation::kSilu) ? w1Rows / 2 : w1Rows;

    TORCH_CHECK(numExperts > 0, "num_experts must be positive");
    TORCH_CHECK(topK > 0, "top_k must be positive");
    TORCH_CHECK(numLocalExperts > 0 && numLocalExperts <= numExperts, "num_local_experts must be in (0, num_experts]");
    TORCH_CHECK(w1Weight.size(0) == numLocalExperts, "w1_weight dim0 must match num_local_experts");
    TORCH_CHECK(w2Weight.size(0) == numLocalExperts, "w2_weight dim0 must match num_local_experts");
    TORCH_CHECK(hiddenSize % 2 == 0, "hidden_size must be even for packed FP4 weights");
    TORCH_CHECK(intermediateSize % 2 == 0, "intermediate_size must be even for packed FP4 weights");
    TORCH_CHECK(w1Weight.size(2) == hiddenSize / 2, "w1_weight dim2 must be hidden_size / 2");
    TORCH_CHECK(w2Weight.size(1) == hiddenSize, "w2_weight dim1 must be hidden_size");
    TORCH_CHECK(w2Weight.size(2) == intermediateSize / 2, "w2_weight dim2 must be intermediate_size / 2");
    TORCH_CHECK(topkIds.size(0) == numTokens && topkIds.size(1) == topK, "topk_ids has incorrect shape");
    TORCH_CHECK(topkWeights.size(0) == numTokens && topkWeights.size(1) == topK, "topk_weights has incorrect shape");
    TORCH_CHECK(
        w1Alpha.numel() == 1 || w1Alpha.numel() == numLocalExperts, "w1_alpha must be scalar or [num_local_experts]");
    TORCH_CHECK(
        w2Alpha.numel() == 1 || w2Alpha.numel() == numLocalExperts, "w2_alpha must be scalar or [num_local_experts]");
    TORCH_CHECK(fc2InputScale.numel() == 1 || fc2InputScale.numel() == numLocalExperts,
        "fc2_input_scale must be scalar or [num_local_experts]");
    if (output.has_value())
    {
        TORCH_CHECK(output->size(0) == numTokens && output->size(1) == hiddenSize, "output has incorrect shape");
    }
}

} // namespace

Tensor b12x_moe_cuda(Tensor const& x, Tensor const& w1Weight, Tensor const& w1SfStorage, Tensor const& w1Alpha,
    Tensor const& fc2InputScale, Tensor const& w2Weight, Tensor const& w2SfStorage, Tensor const& w2Alpha,
    Tensor const& topkIds, Tensor const& topkWeights, std::optional<Tensor> const& output, int64_t numExperts,
    int64_t topK, int64_t numLocalExperts, int64_t activation)
{
    checkB12xTensorInputs(x, w1Weight, w1SfStorage, w1Alpha, fc2InputScale, w2Weight, w2SfStorage, w2Alpha, topkIds,
        topkWeights, output, numExperts, topK, numLocalExperts, activation);

    Tensor out = output.has_value() ? output.value() : at::empty({x.size(0), x.size(1)}, x.options());
    auto const stream = at::cuda::getCurrentCUDAStream(x.get_device());

    auto const w1Rows = static_cast<int32_t>(w1Weight.size(1));
    auto const intermediateSize = activation == static_cast<int64_t>(tensorrt_llm::kernels::B12xMoeActivation::kSilu)
        ? static_cast<int32_t>(w1Rows / 2)
        : w1Rows;
    bool const topkIdsInt64 = topkIds.scalar_type() == torch::ScalarType::Long;

    tensorrt_llm::kernels::invokeB12xMoeCuda(reinterpret_cast<__nv_bfloat16 const*>(x.data_ptr()),
        static_cast<std::uint8_t const*>(w1Weight.data_ptr()), static_cast<std::uint8_t const*>(w1SfStorage.data_ptr()),
        static_cast<float const*>(w1Alpha.data_ptr()), static_cast<float const*>(fc2InputScale.data_ptr()),
        static_cast<std::uint8_t const*>(w2Weight.data_ptr()), static_cast<std::uint8_t const*>(w2SfStorage.data_ptr()),
        static_cast<float const*>(w2Alpha.data_ptr()), topkIds.data_ptr(), topkIdsInt64,
        static_cast<float const*>(topkWeights.data_ptr()), reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
        static_cast<int32_t>(x.size(0)), static_cast<int32_t>(x.size(1)), intermediateSize, w1Rows,
        static_cast<int32_t>(numExperts), static_cast<int32_t>(numLocalExperts), static_cast<int32_t>(topK),
        w1Alpha.numel(), w2Alpha.numel(), fc2InputScale.numel(),
        static_cast<tensorrt_llm::kernels::B12xMoeActivation>(activation), stream);

    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "b12x_moe_cuda(Tensor x, Tensor w1_weight, Tensor w1_sf_storage, Tensor w1_alpha, Tensor fc2_input_scale, "
        "Tensor w2_weight, Tensor w2_sf_storage, Tensor w2_alpha, Tensor topk_ids, Tensor topk_weights, "
        "Tensor? output, int num_experts, int top_k, int num_local_experts, int activation) -> Tensor");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("b12x_moe_cuda", &tensorrt_llm::torch_ext::b12x_moe_cuda);
}
