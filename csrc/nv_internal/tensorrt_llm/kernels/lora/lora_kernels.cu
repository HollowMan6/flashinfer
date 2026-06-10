/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <flashinfer/gemm/group_gemm_lora.cuh>
#include <utility>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/kernels/lora/lora.h"

namespace tensorrt_llm::kernels {

namespace {

size_t dtypeSize(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kHALF:
      return sizeof(half);
    case nvinfer1::DataType::kBF16:
      return sizeof(__nv_bfloat16);
    default:
      TLLM_THROW("LoRA only supports float, half, and bfloat16 data types.");
  }
  return 0;
}

}  // namespace

LoraImpl::LoraImpl(int in_hidden_size, std::vector<int> out_hidden_sizes, bool transA, bool transB,
                   int num_lora_modules, nvinfer1::DataType type, int max_low_rank,
                   std::shared_ptr<CublasGemmWrapper> /*cublasWrapper*/)
    : mType(type),
      mNumLoraModules(num_lora_modules),
      mInHiddenSize(in_hidden_size),
      mOutHiddenSizes(std::move(out_hidden_sizes)),
      mMaxLowRank(max_low_rank) {
  TLLM_CHECK_WITH_INFO(transA == false && transB == true,
                       "FlashInfer LoRA SGMV expects transA=false and transB=true");
  TLLM_CHECK_WITH_INFO(static_cast<int>(mOutHiddenSizes.size()) == mNumLoraModules,
                       "out_hidden_sizes must have num_lora_modules entries");
}

size_t LoraImpl::getWorkspaceSize(int64_t numTokens, int64_t numReqs,
                                  nvinfer1::DataType type) const noexcept {
  (void)numReqs;
  return flashinfer::group_gemm::lora::get_workspace_size(numTokens, mNumLoraModules, mMaxLowRank,
                                                          dtypeSize(type));
}

size_t LoraImpl::getDeviceWorkspaceSize(int64_t numTokens, int64_t numReqs,
                                        nvinfer1::DataType type) const noexcept {
  (void)numReqs;
  return flashinfer::group_gemm::lora::get_device_workspace_size(numTokens, mNumLoraModules,
                                                                 mMaxLowRank, dtypeSize(type));
}

int LoraImpl::run(int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
  void const* const* loraWeightsPtr, int weightIndex, void* const* outputs,
  void* workspace, cudaStream_t stream) {
  (void)numReqs;
  TLLM_CHECK_WITH_INFO(input != nullptr, "LoRA input must not be null");
  TLLM_CHECK_WITH_INFO(loraRanks != nullptr, "LoRA ranks must not be null");
  TLLM_CHECK_WITH_INFO(loraWeightsPtr != nullptr, "LoRA weight pointers must not be null");
  TLLM_CHECK_WITH_INFO(outputs != nullptr, "LoRA outputs must not be null");
  TLLM_CHECK_WITH_INFO(workspace != nullptr, "LoRA workspace must not be null");

  if (numTokens == 0) {
    return 0;
  }

  switch (mType) {
    case nvinfer1::DataType::kFLOAT:
      flashinfer::group_gemm::lora::run_lora_sgmv<float>(
          numTokens, mNumLoraModules, mInHiddenSize, mOutHiddenSizes.data(), mMaxLowRank, input,
          loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
      break;
    case nvinfer1::DataType::kHALF:
      flashinfer::group_gemm::lora::run_lora_sgmv<half>(
          numTokens, mNumLoraModules, mInHiddenSize, mOutHiddenSizes.data(), mMaxLowRank, input,
          loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
      break;
    case nvinfer1::DataType::kBF16:
      flashinfer::group_gemm::lora::run_lora_sgmv<__nv_bfloat16>(
          numTokens, mNumLoraModules, mInHiddenSize, mOutHiddenSizes.data(), mMaxLowRank, input,
          loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
      break;
    default:
      TLLM_THROW("Unsupported LoRA data type.");
  }

  return 0;
}

int LoraImpl::runDevice(int64_t numTokens, int64_t numReqs, void const* input,
                        int32_t const* loraRanks, void const* const* loraWeightsPtr,
  int weightIndex, void* const* outputs, void* workspace,
  cudaStream_t stream) {
  (void)numReqs;
  TLLM_CHECK_WITH_INFO(input != nullptr, "LoRA input must not be null");
  TLLM_CHECK_WITH_INFO(loraRanks != nullptr, "LoRA ranks must not be null");
  TLLM_CHECK_WITH_INFO(loraWeightsPtr != nullptr, "LoRA weight pointers must not be null");
  TLLM_CHECK_WITH_INFO(outputs != nullptr, "LoRA outputs must not be null");
  TLLM_CHECK_WITH_INFO(workspace != nullptr, "LoRA workspace must not be null");

  if (numTokens == 0) {
    return 0;
  }

  switch (mType) {
    case nvinfer1::DataType::kFLOAT:
      flashinfer::group_gemm::lora::run_lora_sgmv_device<float>(
          numTokens, mNumLoraModules, mInHiddenSize, mOutHiddenSizes.data(), mMaxLowRank, input,
          loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
      break;
    case nvinfer1::DataType::kHALF:
      flashinfer::group_gemm::lora::run_lora_sgmv_device<half>(
          numTokens, mNumLoraModules, mInHiddenSize, mOutHiddenSizes.data(), mMaxLowRank, input,
          loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
      break;
    case nvinfer1::DataType::kBF16:
      flashinfer::group_gemm::lora::run_lora_sgmv_device<__nv_bfloat16>(
          numTokens, mNumLoraModules, mInHiddenSize, mOutHiddenSizes.data(), mMaxLowRank, input,
          loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
      break;
    default:
      TLLM_THROW("Unsupported LoRA data type.");
  }

  return 0;
}

int LoraImpl::runDeviceLowRank(int64_t numTokens, int64_t numReqs, void const* input,
                               int32_t const* loraRanks, void const* const* loraWeightsPtr,
  int weightIndex, void* workspace, cudaStream_t stream) {
  (void)numReqs;
  TLLM_CHECK_WITH_INFO(input != nullptr, "LoRA input must not be null");
  TLLM_CHECK_WITH_INFO(loraRanks != nullptr, "LoRA ranks must not be null");
  TLLM_CHECK_WITH_INFO(loraWeightsPtr != nullptr, "LoRA weight pointers must not be null");
  TLLM_CHECK_WITH_INFO(workspace != nullptr, "LoRA workspace must not be null");

  if (numTokens == 0) {
    return 0;
  }

  switch (mType) {
    case nvinfer1::DataType::kFLOAT:
      flashinfer::group_gemm::lora::run_lora_sgmv_device_low_rank<float>(
          numTokens, mNumLoraModules, mInHiddenSize, mMaxLowRank, input, loraRanks,
          loraWeightsPtr, weightIndex, workspace, stream);
      break;
    case nvinfer1::DataType::kHALF:
      flashinfer::group_gemm::lora::run_lora_sgmv_device_low_rank<half>(
          numTokens, mNumLoraModules, mInHiddenSize, mMaxLowRank, input, loraRanks,
          loraWeightsPtr, weightIndex, workspace, stream);
      break;
    case nvinfer1::DataType::kBF16:
      flashinfer::group_gemm::lora::run_lora_sgmv_device_low_rank<__nv_bfloat16>(
          numTokens, mNumLoraModules, mInHiddenSize, mMaxLowRank, input, loraRanks,
          loraWeightsPtr, weightIndex, workspace, stream);
      break;
    default:
      TLLM_THROW("Unsupported LoRA data type.");
  }

  return 0;
}

int LoraImpl::runDeviceStridedOutputs(
    int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
    void const* const* loraWeightsPtr, int weightIndex, void* const* outputs,
    int64_t const* outputStrides, void* workspace, cudaStream_t stream) {
  (void)numReqs;
  TLLM_CHECK_WITH_INFO(input != nullptr, "LoRA input must not be null");
  TLLM_CHECK_WITH_INFO(loraRanks != nullptr, "LoRA ranks must not be null");
  TLLM_CHECK_WITH_INFO(loraWeightsPtr != nullptr, "LoRA weight pointers must not be null");
  TLLM_CHECK_WITH_INFO(outputs != nullptr, "LoRA outputs must not be null");
  TLLM_CHECK_WITH_INFO(outputStrides != nullptr, "LoRA output strides must not be null");
  TLLM_CHECK_WITH_INFO(workspace != nullptr, "LoRA workspace must not be null");

  if (numTokens == 0) {
    return 0;
  }

  switch (mType) {
    case nvinfer1::DataType::kFLOAT:
      flashinfer::group_gemm::lora::run_lora_sgmv_device_strided_outputs<float>(
          numTokens, mNumLoraModules, mInHiddenSize, mOutHiddenSizes.data(), outputStrides,
          mMaxLowRank, input, loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
      break;
    case nvinfer1::DataType::kHALF:
      flashinfer::group_gemm::lora::run_lora_sgmv_device_strided_outputs<half>(
          numTokens, mNumLoraModules, mInHiddenSize, mOutHiddenSizes.data(), outputStrides,
          mMaxLowRank, input, loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
      break;
    case nvinfer1::DataType::kBF16:
      flashinfer::group_gemm::lora::run_lora_sgmv_device_strided_outputs<__nv_bfloat16>(
          numTokens, mNumLoraModules, mInHiddenSize, mOutHiddenSizes.data(), outputStrides,
          mMaxLowRank, input, loraRanks, loraWeightsPtr, weightIndex, outputs, workspace, stream);
      break;
    default:
      TLLM_THROW("Unsupported LoRA data type.");
  }

  return 0;
}

int Lora_run(LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input,
             int32_t const* loraRanks, void const* const* loraWeightsPtr, int weightIndex,
             void* const* outputs, void* workspace, cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(impl != nullptr, "Attempt to run an empty LoraImpl");
  return impl->run(numTokens, numReqs, input, loraRanks, loraWeightsPtr, weightIndex, outputs,
                   workspace, stream);
}

int Lora_run_device(LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input,
                    int32_t const* loraRanks, void const* const* loraWeightsPtr, int weightIndex,
                    void* const* outputs, void* workspace, cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(impl != nullptr, "Attempt to run an empty LoraImpl");
  return impl->runDevice(numTokens, numReqs, input, loraRanks, loraWeightsPtr, weightIndex, outputs,
                         workspace, stream);
}

int Lora_run_device_low_rank(LoraImpl* impl, int64_t numTokens, int64_t numReqs,
                             void const* input, int32_t const* loraRanks,
                             void const* const* loraWeightsPtr, int weightIndex,
                             void* workspace, cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(impl != nullptr, "Attempt to run an empty LoraImpl");
  return impl->runDeviceLowRank(numTokens, numReqs, input, loraRanks, loraWeightsPtr, weightIndex,
                                workspace, stream);
}

int Lora_run_device_strided_outputs(
    LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input,
    int32_t const* loraRanks, void const* const* loraWeightsPtr, int weightIndex,
    void* const* outputs, int64_t const* outputStrides, void* workspace, cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(impl != nullptr, "Attempt to run an empty LoraImpl");
  return impl->runDeviceStridedOutputs(numTokens, numReqs, input, loraRanks, loraWeightsPtr,
                                       weightIndex, outputs, outputStrides, workspace, stream);
}

}  // namespace tensorrt_llm::kernels
