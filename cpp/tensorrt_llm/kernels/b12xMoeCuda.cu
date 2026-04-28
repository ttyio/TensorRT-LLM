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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/b12xMoeCuda.h"

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

constexpr int32_t kSfVecSize = 16;
constexpr int32_t kThreadsPerBlock = 128;

__device__ __forceinline__ float e2m1ToFloat(std::uint8_t value)
{
    float result = 0.0f;
    switch (value & 7U)
    {
    case 0: result = 0.0f; break;
    case 1: result = 0.5f; break;
    case 2: result = 1.0f; break;
    case 3: result = 1.5f; break;
    case 4: result = 2.0f; break;
    case 5: result = 3.0f; break;
    case 6: result = 4.0f; break;
    default: result = 6.0f; break;
    }
    return (value & 8U) ? -result : result;
}

__device__ __forceinline__ std::uint8_t floatToE2m1(float value)
{
    float const absValue = fabsf(value);
    std::uint8_t result = value < 0.0f ? 8U : 0U;
    int32_t fp4AbsValue = 7;
    static constexpr float kThresholds[8] = {0.0f, 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    for (; fp4AbsValue > 0; --fp4AbsValue)
    {
        if (kThresholds[fp4AbsValue] < absValue)
        {
            break;
        }
        if (kThresholds[fp4AbsValue] == absValue && !(fp4AbsValue & 1))
        {
            break;
        }
    }
    return result | static_cast<std::uint8_t>(fp4AbsValue);
}

__device__ __forceinline__ float e4m3fnToFloat(std::uint8_t value)
{
    if ((value & 0x7fU) == 0U)
    {
        return (value & 0x80U) ? -0.0f : 0.0f;
    }

    int32_t const sign = value & 0x80U;
    int32_t const exp = (value >> 3) & 0x0fU;
    int32_t const mant = value & 0x07U;
    float result;
    if (exp == 0)
    {
        result = ldexpf(static_cast<float>(mant), -9);
    }
    else
    {
        result = ldexpf(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
    }
    return sign ? -result : result;
}

__device__ __forceinline__ float scaleValue(float const* scales, int64_t scaleCount, int32_t expert)
{
    if (scaleCount <= 0)
    {
        return 1.0f;
    }
    if (scaleCount == 1)
    {
        return scales[0];
    }
    return scales[expert];
}

__device__ __forceinline__ int32_t alignUp(int32_t value, int32_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

__device__ __forceinline__ int64_t swizzledSfOffset(
    int32_t expert, int32_t row, int32_t sfCol, int32_t rows, int32_t sfCols)
{
    int32_t const rowsPadded = alignUp(rows, 128);
    int32_t const colsPadded = alignUp(sfCols, 4);

    int32_t const columnIdxInGroup0 = sfCol & 3;
    int32_t const columnGroupIdx = sfCol >> 2;
    int32_t const rowIdxInGroup0 = row & 31;
    int32_t const rowIdxInGroup1 = (row & 127) >> 5;
    int32_t const rowGroupIdx = row >> 7;

    int32_t const perExpertOffset = columnIdxInGroup0 + columnGroupIdx * (4 * 128) + rowIdxInGroup0 * 16
        + rowIdxInGroup1 * 4 + rowGroupIdx * (128 * colsPadded);
    return static_cast<int64_t>(expert) * rowsPadded * colsPadded + perExpertOffset;
}

__device__ __forceinline__ float dequantFp4Weight(std::uint8_t const* packed, std::uint8_t const* scaleStorage,
    float const* alpha, int64_t alphaCount, int32_t expert, int32_t row, int32_t col, int32_t rows, int32_t cols)
{
    int32_t const packedCols = cols / 2;
    std::uint8_t const packedByte = packed[(static_cast<int64_t>(expert) * rows + row) * packedCols + (col >> 1)];
    std::uint8_t const nibble = (col & 1) ? (packedByte >> 4) : (packedByte & 0x0fU);

    int32_t const sfCols = (cols + kSfVecSize - 1) / kSfVecSize;
    int32_t const sfCol = col / kSfVecSize;
    std::uint8_t const encodedScale = scaleStorage[swizzledSfOffset(expert, row, sfCol, rows, sfCols)];
    float const blockScale = e4m3fnToFloat(encodedScale);
    return e2m1ToFloat(nibble & 0x0fU) * blockScale * scaleValue(alpha, alphaCount, expert);
}

__device__ __forceinline__ float silu(float x)
{
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ void quantDequantActivationBlock(float values[16], int32_t count, float globalScale)
{
    if (globalScale == 0.0f)
    {
        globalScale = 1.0f;
    }

    float maxAbs = 0.0f;
#pragma unroll
    for (int32_t i = 0; i < 16; ++i)
    {
        if (i < count)
        {
            maxAbs = fmaxf(maxAbs, fabsf(values[i] * globalScale));
        }
    }
    if (maxAbs == 0.0f)
    {
        return;
    }

    float const blockScale = maxAbs / 6.0f;
    float const invBlockScale = 1.0f / blockScale;
#pragma unroll
    for (int32_t i = 0; i < 16; ++i)
    {
        if (i < count)
        {
            std::uint8_t const q = floatToE2m1(values[i] * globalScale * invBlockScale);
            values[i] = e2m1ToFloat(q) * blockScale / globalScale;
        }
    }
}

template <typename IdType>
__global__ void b12xMoeReferenceKernel(__nv_bfloat16 const* x, std::uint8_t const* w1Weight,
    std::uint8_t const* w1SfStorage, float const* w1Alpha, float const* fc2InputScale, std::uint8_t const* w2Weight,
    std::uint8_t const* w2SfStorage, float const* w2Alpha, IdType const* topkIds, float const* topkWeights,
    __nv_bfloat16* output, int32_t numTokens, int32_t hiddenSize, int32_t intermediateSize, int32_t w1Rows,
    int32_t numExperts, int32_t numLocalExperts, int32_t topK, int64_t w1AlphaCount, int64_t w2AlphaCount,
    int64_t fc2InputScaleCount, int32_t activation)
{
    int32_t const idx = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    int32_t const total = numTokens * hiddenSize;
    if (idx >= total)
    {
        return;
    }

    int32_t const token = idx / hiddenSize;
    int32_t const outCol = idx - token * hiddenSize;
    float outputAcc = 0.0f;

    for (int32_t tk = 0; tk < topK; ++tk)
    {
        int32_t const expertId = static_cast<int32_t>(topkIds[token * topK + tk]);
        if (expertId < 0 || expertId >= numExperts || expertId >= numLocalExperts)
        {
            continue;
        }
        float const routeWeight = topkWeights[token * topK + tk];
        float expertAcc = 0.0f;

        for (int32_t blockStart = 0; blockStart < intermediateSize; blockStart += 16)
        {
            int32_t const count = intermediateSize - blockStart < 16 ? intermediateSize - blockStart : 16;
            float act[16];

#pragma unroll
            for (int32_t j = 0; j < 16; ++j)
            {
                float value = 0.0f;
                if (j < count)
                {
                    int32_t const inter = blockStart + j;
                    if (activation == static_cast<int32_t>(B12xMoeActivation::kSilu))
                    {
                        float upAcc = 0.0f;
                        float gateAcc = 0.0f;
                        int32_t const upRow = inter;
                        int32_t const gateRow = intermediateSize + inter;
                        if (gateRow < w1Rows)
                        {
                            for (int32_t h = 0; h < hiddenSize; ++h)
                            {
                                float const xVal = __bfloat162float(x[token * hiddenSize + h]);
                                upAcc += xVal
                                    * dequantFp4Weight(w1Weight, w1SfStorage, w1Alpha, w1AlphaCount, expertId, upRow, h,
                                        w1Rows, hiddenSize);
                                gateAcc += xVal
                                    * dequantFp4Weight(w1Weight, w1SfStorage, w1Alpha, w1AlphaCount, expertId, gateRow,
                                        h, w1Rows, hiddenSize);
                            }
                        }
                        value = silu(gateAcc) * upAcc;
                    }
                    else
                    {
                        float fc1Acc = 0.0f;
                        int32_t const fc1Row = inter;
                        if (fc1Row < w1Rows)
                        {
                            for (int32_t h = 0; h < hiddenSize; ++h)
                            {
                                float const xVal = __bfloat162float(x[token * hiddenSize + h]);
                                fc1Acc += xVal
                                    * dequantFp4Weight(w1Weight, w1SfStorage, w1Alpha, w1AlphaCount, expertId, fc1Row,
                                        h, w1Rows, hiddenSize);
                            }
                        }
                        float const relu = fmaxf(fc1Acc, 0.0f);
                        value = relu * relu;
                    }
                }
                act[j] = value;
            }

            quantDequantActivationBlock(act, count, scaleValue(fc2InputScale, fc2InputScaleCount, expertId));

#pragma unroll
            for (int32_t j = 0; j < 16; ++j)
            {
                if (j < count)
                {
                    int32_t const inter = blockStart + j;
                    float const w2Val = dequantFp4Weight(w2Weight, w2SfStorage, w2Alpha, w2AlphaCount, expertId, outCol,
                        inter, hiddenSize, intermediateSize);
                    expertAcc += act[j] * w2Val;
                }
            }
        }
        outputAcc += routeWeight * expertAcc;
    }

    output[idx] = __float2bfloat16_rn(outputAcc);
}

template <typename IdType>
void launchB12xMoeReference(__nv_bfloat16 const* x, std::uint8_t const* w1Weight, std::uint8_t const* w1SfStorage,
    float const* w1Alpha, float const* fc2InputScale, std::uint8_t const* w2Weight, std::uint8_t const* w2SfStorage,
    float const* w2Alpha, IdType const* topkIds, float const* topkWeights, __nv_bfloat16* output, int32_t numTokens,
    int32_t hiddenSize, int32_t intermediateSize, int32_t w1Rows, int32_t numExperts, int32_t numLocalExperts,
    int32_t topK, int64_t w1AlphaCount, int64_t w2AlphaCount, int64_t fc2InputScaleCount, B12xMoeActivation activation,
    cudaStream_t stream)
{
    int32_t const total = numTokens * hiddenSize;
    if (total == 0)
    {
        return;
    }
    int32_t const blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
    b12xMoeReferenceKernel<<<blocks, kThreadsPerBlock, 0, stream>>>(x, w1Weight, w1SfStorage, w1Alpha, fc2InputScale,
        w2Weight, w2SfStorage, w2Alpha, topkIds, topkWeights, output, numTokens, hiddenSize, intermediateSize, w1Rows,
        numExperts, numLocalExperts, topK, w1AlphaCount, w2AlphaCount, fc2InputScaleCount,
        static_cast<int32_t>(activation));
}

} // namespace

void invokeB12xMoeCuda(__nv_bfloat16 const* x, std::uint8_t const* w1Weight, std::uint8_t const* w1SfStorage,
    float const* w1Alpha, float const* fc2InputScale, std::uint8_t const* w2Weight, std::uint8_t const* w2SfStorage,
    float const* w2Alpha, void const* topkIds, bool topkIdsInt64, float const* topkWeights, __nv_bfloat16* output,
    int32_t numTokens, int32_t hiddenSize, int32_t intermediateSize, int32_t w1Rows, int32_t numExperts,
    int32_t numLocalExperts, int32_t topK, int64_t w1AlphaCount, int64_t w2AlphaCount, int64_t fc2InputScaleCount,
    B12xMoeActivation activation, cudaStream_t stream)
{
    if (topkIdsInt64)
    {
        launchB12xMoeReference(reinterpret_cast<__nv_bfloat16 const*>(x), w1Weight, w1SfStorage, w1Alpha, fc2InputScale,
            w2Weight, w2SfStorage, w2Alpha, static_cast<int64_t const*>(topkIds), topkWeights, output, numTokens,
            hiddenSize, intermediateSize, w1Rows, numExperts, numLocalExperts, topK, w1AlphaCount, w2AlphaCount,
            fc2InputScaleCount, activation, stream);
    }
    else
    {
        launchB12xMoeReference(reinterpret_cast<__nv_bfloat16 const*>(x), w1Weight, w1SfStorage, w1Alpha, fc2InputScale,
            w2Weight, w2SfStorage, w2Alpha, static_cast<int32_t const*>(topkIds), topkWeights, output, numTokens,
            hiddenSize, intermediateSize, w1Rows, numExperts, numLocalExperts, topK, w1AlphaCount, w2AlphaCount,
            fc2InputScaleCount, activation, stream);
    }
    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
