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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

enum class B12xMoeActivation : int32_t
{
    kSilu = 0,
    kRelu2 = 1,
};

void invokeB12xMoeCuda(__nv_bfloat16 const* x, std::uint8_t const* w1Weight, std::uint8_t const* w1SfStorage,
    float const* w1Alpha, float const* fc2InputScale, std::uint8_t const* w2Weight, std::uint8_t const* w2SfStorage,
    float const* w2Alpha, void const* topkIds, bool topkIdsInt64, float const* topkWeights, __nv_bfloat16* output,
    int32_t numTokens, int32_t hiddenSize, int32_t intermediateSize, int32_t w1Rows, int32_t numExperts,
    int32_t numLocalExperts, int32_t topK, int64_t w1AlphaCount, int64_t w2AlphaCount, int64_t fc2InputScaleCount,
    B12xMoeActivation activation, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
