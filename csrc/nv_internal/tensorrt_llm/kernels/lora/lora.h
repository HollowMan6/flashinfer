/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include <memory>
#include <vector>

#include "tensorrt_llm/common/NvInferRuntime.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"

namespace tensorrt_llm::kernels {

using CublasGemmWrapper = tensorrt_llm::common::CublasMMWrapper;

class LoraImpl {
 public:
  // max_low_rank is the padded storage rank used for A/B strides and workspace size.
  LoraImpl(int in_hidden_size, std::vector<int> out_hidden_sizes, bool transA, bool transB,
           int num_lora_modules, nvinfer1::DataType type, int max_low_rank,
           std::shared_ptr<CublasGemmWrapper> cublasWrapper);

  [[nodiscard]] size_t getWorkspaceSize(int64_t numTokens, int64_t numReqs,
                                        nvinfer1::DataType type) const noexcept;
  [[nodiscard]] size_t getDeviceWorkspaceSize(int64_t numTokens, int64_t numReqs,
                                              nvinfer1::DataType type) const noexcept;
  // loraRanks are math ranks. Weight storage is padded to the constructor max_low_rank.
  int run(int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
          void const* const* loraWeightsPtr, int weightIndex, void* const* outputs, void* workspace,
          cudaStream_t stream);
  int runDevice(int64_t numTokens, int64_t numReqs, void const* input, int32_t const* loraRanks,
                void const* const* loraWeightsPtr, int weightIndex, void* const* outputs,
                void* workspace, cudaStream_t stream);
  int runDeviceLowRank(int64_t numTokens, int64_t numReqs, void const* input,
                       int32_t const* loraRanks, void const* const* loraWeightsPtr,
                       int weightIndex, void* workspace, cudaStream_t stream);
  int runDeviceStridedOutputs(int64_t numTokens, int64_t numReqs, void const* input,
                              int32_t const* loraRanks, void const* const* loraWeightsPtr,
                              int weightIndex, void* const* outputs,
                              int64_t const* outputStrides, void* workspace,
                              cudaStream_t stream);

 private:
  nvinfer1::DataType mType;
  int mNumLoraModules;

  int mInHiddenSize;
  std::vector<int> mOutHiddenSizes;
  int mMaxLowRank;
};

// Change to following declarations must sync with moe_kernels.h in internal kernel repo
int Lora_run(LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input,
             int32_t const* loraRanks, void const* const* loraWeightsPtr, int weightIndex,
             void* const* outputs, void* workspace, cudaStream_t stream);

int Lora_run_device(LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input,
                    int32_t const* loraRanks, void const* const* loraWeightsPtr, int weightIndex,
                    void* const* outputs, void* workspace, cudaStream_t stream);

int Lora_run_device_low_rank(LoraImpl* impl, int64_t numTokens, int64_t numReqs,
                             void const* input, int32_t const* loraRanks,
                             void const* const* loraWeightsPtr, int weightIndex,
                             void* workspace, cudaStream_t stream);

int Lora_run_device_strided_outputs(
    LoraImpl* impl, int64_t numTokens, int64_t numReqs, void const* input,
    int32_t const* loraRanks, void const* const* loraWeightsPtr, int weightIndex,
    void* const* outputs, int64_t const* outputStrides, void* workspace, cudaStream_t stream);

}  // namespace tensorrt_llm::kernels
