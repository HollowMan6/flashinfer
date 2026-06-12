/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <mma.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <type_traits>

#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/workspace.h"

// Ignore CUTLASS warnings about type punning
#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_extensions/epilogue/thread/fused_activations.h"

#ifdef __GNUC__  // Check if the compiler is GCC or Clang
#pragma GCC diagnostic pop
#endif

#include "moe_kernels.h"
#include "moe_util_kernels.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/cutlass_type_conversion.h"
#include "tensorrt_llm/kernels/preQuantScaleKernel.h"
#include "tensorrt_llm/kernels/quantization.cuh"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels::cutlass_kernels {

constexpr int CVT_ELTS_PER_THREAD = 8;

template <typename Fn>
auto dispatchNVFP44Over6Config(Fn&& fn) {
  bool const use4Over6 = tensorrt_llm::common::getEnvNVFP4Use4Over6();
  bool const disableFP4QuantFastMath = tensorrt_llm::common::getEnvDisableFP4QuantFastMath();

  auto dispatchDisableFastMath = [&](auto e4m3MaxTag, auto errModeTag, auto errUseFastMathTag) {
    if (disableFP4QuantFastMath) {
      return fn(std::true_type{},
                NVFP44Over6Config<decltype(e4m3MaxTag)::value, decltype(errModeTag)::value,
                                  decltype(errUseFastMathTag)::value>{});
    }
    return fn(std::false_type{},
              NVFP44Over6Config<decltype(e4m3MaxTag)::value, decltype(errModeTag)::value,
                                decltype(errUseFastMathTag)::value>{});
  };

  if (use4Over6) {
    NVFP44Over6ErrMode const errMode = tensorrt_llm::common::getEnvNVFP44Over6ErrMode();
    bool const errUseFastMath = tensorrt_llm::common::getEnvNVFP44Over6ErrUseFastMath();
    int e4m3Max = 448;
    if (tensorrt_llm::common::getEnvNVFP44Over6E4M3Use256()) {
      e4m3Max = 256;
    }

    auto dispatchErrUseFastMath = [&](auto e4m3MaxTag, auto errModeTag) {
      if (errUseFastMath) {
        return dispatchDisableFastMath(e4m3MaxTag, errModeTag, std::true_type{});
      }
      return dispatchDisableFastMath(e4m3MaxTag, errModeTag, std::false_type{});
    };
    auto dispatchErrMode = [&](auto e4m3MaxTag) {
      switch (errMode) {
        case NVFP44Over6ErrMode::MAE:
          return dispatchErrUseFastMath(
              e4m3MaxTag, std::integral_constant<NVFP44Over6ErrMode, NVFP44Over6ErrMode::MAE>{});
        case NVFP44Over6ErrMode::MSE:
          return dispatchErrUseFastMath(
              e4m3MaxTag, std::integral_constant<NVFP44Over6ErrMode, NVFP44Over6ErrMode::MSE>{});
        default:
          TLLM_CHECK_WITH_INFO(false, "Unsupported NVFP4 4over6 error mode.");
          return dispatchErrUseFastMath(
              e4m3MaxTag, std::integral_constant<NVFP44Over6ErrMode, NVFP44Over6ErrMode::MAE>{});
      }
    };
    if (e4m3Max == 256) {
      return dispatchErrMode(std::integral_constant<int, 256>{});
    }
    TLLM_CHECK_WITH_INFO(e4m3Max == 448, "Unsupported NVFP4 4over6 E4M3 max.");
    return dispatchErrMode(std::integral_constant<int, 448>{});
  }

  if (disableFP4QuantFastMath) {
    return fn(std::true_type{}, std::false_type{});
  }
  return fn(std::false_type{}, std::false_type{});
}

/**
 * Takes the input maps and prepares the expanded maps for min latency
 * @param num_active_experts_per_node: Number of active experts on current node
 * @param experts_to_token_scores: The score of each token for each activated expert. 0 if the
 * expert is not chosen by the token. Only the first num_active_experts_per_ rows are valid
 * @param active_expert_global_ids: The global expert id for each activated expert
 * Only the first num_active_experts_per_ values are valid
 * @param expert_first_token_offset: Store the first token offset for each expert
 */
template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void initTensor(T* value, int const tid, int const total_num,
                                           T const init_value) {
  for (int i = tid; i < total_num; i += BLOCK_SIZE) {
    value[i] = init_value;
  }
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void setLocalExperts(int* s_local_experts,
                                                T const* token_selected_experts,
                                                int const total_num_experts, int const tid,
                                                int const start_expert, int const end_expert) {
  for (int i = tid; i < total_num_experts; i += BLOCK_SIZE) {
    int const expert = token_selected_experts[i];

    // If expert is in the current node, subtract start_expert to shift the range to [0,
    // num_experts_per_node)
    bool is_valid_expert = expert >= start_expert && expert < end_expert;
    if (is_valid_expert) {
      int local_expert_id = expert - start_expert;
      if (s_local_experts[local_expert_id] == 0) {
        s_local_experts[local_expert_id] =
            1;  // @TODO: Make sure that we allow duplicated write here
      }
    }
  }
  __syncthreads();
}

template <typename T, int BLOCK_SIZE>
__device__ __forceinline__ void prefixSum(T* out, T* in, int const num, int const tid) {
  typedef cub::BlockScan<T, BLOCK_SIZE> BlockScan;
  __shared__ typename BlockScan::TempStorage tempStorage;

  T threadData = 0;
  if (tid < num) {
    threadData = in[tid];
  }

  BlockScan(tempStorage).InclusiveSum(threadData, threadData);
  __syncthreads();

  if (tid < num) {
    out[tid] = threadData;
  }
  __syncthreads();
}

__device__ __forceinline__ void setActiveNum(int& num_active, int& num_active_offset_start,
                                             int& num_active_offset_end, int const cluster_size,
                                             int const cluster_rank) {
  int num_remainder = num_active % cluster_size;
  int num_active_per_node =
      max(0, num_active - 1) / cluster_size;  // num_active_per_node shouldn't be neg
  if (cluster_rank < num_remainder) {
    num_active = num_active_per_node + 1;
    num_active_offset_start = cluster_rank * num_active;
  } else {
    num_active = num_active_per_node;
    num_active_offset_start = cluster_rank * num_active_per_node + num_remainder;
  }
  num_active_offset_end = num_active_offset_start + num_active;
}

template <int BLOCK_SIZE>
__global__ void buildMinLatencyActiveExpertMapsKernel(
    int* num_active_experts_per_node, float* experts_to_token_scores, int* active_expert_global_ids,
    int64_t* expert_first_token_offset, int const* token_selected_experts,
    float const* token_final_scales, int64_t const num_tokens, int const num_experts_per_token,
    int const start_expert, int const end_expert, int const num_experts_per_node,
    bool const smart_routing, int const cluster_rank, int const cluster_size,
    int const num_experts_smem) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  // Use one block to process the min latency case
  int tid = threadIdx.x;
  // 0. init the global memory experts_to_token_scores [num_experts_per_node, num_token]
  int const total_local_scales = num_experts_per_node * num_tokens;
  initTensor<float, BLOCK_SIZE>(experts_to_token_scores, tid, total_local_scales, 0.0f);
  initTensor<int, BLOCK_SIZE>(active_expert_global_ids, tid, num_experts_per_node, -1);

  __threadfence();  //@Todo: check do I need this fence for previous zero setting

  // 1. mask for the active expert: 1 stands for active
  extern __shared__ int s_local_experts[];
  int* s_store_experts = s_local_experts + num_experts_smem;
  initTensor<int, BLOCK_SIZE>(s_local_experts, tid, num_experts_smem, 0);
  __syncthreads();

  // 2. set the shared array s_local_experts[]
  int const total_num_experts = num_tokens * num_experts_per_token;
  setLocalExperts<int, BLOCK_SIZE>(s_local_experts, token_selected_experts, total_num_experts, tid,
                                   start_expert, end_expert);

  // 3. perform prefix sum to acquire the store position and total active experts
  //@TODO: Use cub first, might need to change it to self-defined api
  prefixSum<int, BLOCK_SIZE>(s_store_experts, s_local_experts, num_experts_smem, tid);

  // 4. store the num of active experts
  int num_active = s_store_experts[num_experts_smem - 1];
  int num_active_offset_start = 0;
  int num_active_offset_end = 0;

  if (smart_routing) {
    setActiveNum(num_active, num_active_offset_start, num_active_offset_end, cluster_size,
                 cluster_rank);
  }

  if (tid == 0) {
    *num_active_experts_per_node = num_active;
  }

  // 5. store the global expert id for each expert
  if (smart_routing) {
    for (int i = tid; i < num_experts_smem; i += BLOCK_SIZE) {
      if (s_local_experts[i]) {
        int offset = s_store_experts[i] - 1;
        if (offset >= num_active_offset_start && offset < num_active_offset_end) {
          active_expert_global_ids[offset - num_active_offset_start] = i;
        } else {
          s_local_experts[i] = 0;
        }
      }
    }
    __syncthreads();  // Need sync to update the s_local_experts
  } else {
    for (int i = tid; i < num_experts_smem; i += BLOCK_SIZE) {
      if (s_local_experts[i]) {
        int offset = s_store_experts[i] - 1;
        active_expert_global_ids[offset] = i + start_expert;
      }
    }
  }

  // 6. store the scale values
  __threadfence();  //@Todo: check do I need this fence for previous zero setting
  for (int i = tid; i < total_num_experts; i += BLOCK_SIZE) {
    int const expert = token_selected_experts[i];

    // If expert is not in the current node, set it to num_experts_per_node
    // If expert is in the current node, subtract start_expert to shift the range to [0,
    // num_experts_per_node)
    bool is_valid_expert =
        smart_routing ? s_local_experts[expert] : (expert >= start_expert && expert < end_expert);

    if (is_valid_expert) {
      int token = i / num_experts_per_token;
      float const scale = token_final_scales[i];
      int offset = s_store_experts[expert - start_expert] - 1 - num_active_offset_start;
      experts_to_token_scores[offset * num_tokens + token] = scale;
    }
  }
  // 7. set default value for redundant memory
  for (int i_exp = num_active + tid; i_exp < num_experts_per_node; i_exp += BLOCK_SIZE) {
    active_expert_global_ids[i_exp] = -1;
  }
  // 8. set expert_first_token_offset
  for (int i_exp = tid; i_exp < num_experts_per_node + 1; i_exp += BLOCK_SIZE) {
    if (i_exp < num_active) {
      expert_first_token_offset[i_exp] = i_exp * num_tokens;
    } else {
      expert_first_token_offset[i_exp] = num_active * num_tokens;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void buildMinLatencyActiveExpertMaps(
    int* num_active_experts_per_node, float* experts_to_token_scores, int* active_expert_global_ids,
    int64_t* expert_first_token_offset, int const* token_selected_experts,
    float const* token_final_scales, int64_t const num_tokens, int const experts_per_token,
    int const start_expert, int const end_expert, int const num_experts_per_node,
    int const cluster_rank, int const cluster_size, int const num_experts_smem, bool enable_pdl,
    cudaStream_t const stream) {
  TLLM_CHECK_WITH_INFO(num_experts_per_node == (end_expert - start_expert),
                       "num_experts_per_node must be equal to end_expert - start_expert");

  TLLM_CHECK_WITH_INFO(num_experts_per_node <= 256,
                       "don't support num_experts_per_node > 256 cases");

  int const threads = 256;
  int const blocks = 1;
  bool const smart_routing = cluster_size > 1;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = num_experts_smem * sizeof(int) * 2;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, buildMinLatencyActiveExpertMapsKernel<threads>,
                     num_active_experts_per_node, experts_to_token_scores, active_expert_global_ids,
                     expert_first_token_offset, token_selected_experts, token_final_scales,
                     num_tokens, experts_per_token, start_expert, end_expert, num_experts_per_node,
                     smart_routing, cluster_rank, cluster_size, num_experts_smem);
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
__global__ void fusedBuildExpertMapsSortFirstTokenKernel(
    int const* const token_selected_experts, int* const permuted_row_to_unpermuted_row,
    int* const unpermuted_row_to_permuted_row, int64_t* const expert_first_token_offset,
    int64_t const num_tokens, int const experts_per_token, int const start_expert,
    int const end_expert, int const num_experts_per_node) {
  // Only using block wise collective so we can only have one block
  assert(gridDim.x == 1);

  assert(start_expert <= end_expert);
  assert(num_experts_per_node == (end_expert - start_expert));
  assert(num_experts_per_node <= (1 << LOG2_NUM_EXPERTS));

  int const token = blockIdx.x * BLOCK_SIZE + threadIdx.x;

  bool is_valid_token = token < num_tokens;

  // This is the masked expert id for this token
  int local_token_selected_experts[EXPERTS_PER_TOKEN];
  // This is the final permuted rank of this token (ranked by selected expert)
  int local_token_permuted_indices[EXPERTS_PER_TOKEN];

  // Wait PDL before reading token_selected_experts
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

// build expert map
// we need to populate expert ids for all threads, even if there are
// fewer tokens
#pragma unroll
  for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
    int const expert = is_valid_token ? token_selected_experts[token * EXPERTS_PER_TOKEN + i]
                                      : num_experts_per_node;

    // If the token is not valid, set the expert id to num_experts_per_node + 1
    // If expert is not in the current node, set it to num_experts_per_node
    // If expert is in the current node, subtract start_expert to shift the range to [0,
    // num_experts_per_node)
    bool is_valid_expert = expert >= start_expert && expert < end_expert;
    local_token_selected_experts[i] = !is_valid_token   ? num_experts_per_node + 1
                                      : is_valid_expert ? (expert - start_expert)
                                                        : num_experts_per_node;
  }

  // TODO: decompose cub's sort to expose the bucket starts, and just return
  // that to elide the binary search

  // sort the expert map
  using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
  extern __shared__ unsigned char temp_storage[];
  auto& sort_temp = *reinterpret_cast<typename BlockRadixRank::TempStorage*>(temp_storage);

  // Sanity check that the number of bins do correspond to the number of experts
  static_assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= (1 << LOG2_NUM_EXPERTS));
  assert(BlockRadixRank::BINS_TRACKED_PER_THREAD * BLOCK_SIZE >= num_experts_per_node);

  int local_expert_first_token_offset[BlockRadixRank::BINS_TRACKED_PER_THREAD];

  cub::BFEDigitExtractor<int> extractor(0, LOG2_NUM_EXPERTS);
  BlockRadixRank(sort_temp).RankKeys(local_token_selected_experts, local_token_permuted_indices,
                                     extractor, local_expert_first_token_offset);

// We are done with compute, launch the dependent kernels while the stores are in flight
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // write to shared memory and global memory
  if (is_valid_token) {
#pragma unroll
    for (int i = 0; i < EXPERTS_PER_TOKEN; i++) {
      int const unpermuted_row = i * num_tokens + token;
      int const permuted_row = local_token_permuted_indices[i];
      permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
      unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
    }
  }

#pragma unroll
  for (int expert_id = 0; expert_id < BlockRadixRank::BINS_TRACKED_PER_THREAD; expert_id++) {
    int out_expert_id = expert_id + token * BlockRadixRank::BINS_TRACKED_PER_THREAD;
    if (out_expert_id < num_experts_per_node + 1) {
      expert_first_token_offset[out_expert_id] = local_expert_first_token_offset[expert_id];
    }
  }
}

template <int BLOCK_SIZE, int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenDispatch(
    int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token,
    int const start_expert, int const end_expert, bool enable_pdl, cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(num_experts_per_node == (end_expert - start_expert),
                       "num_experts_per_node must be equal to end_expert - start_expert");
  int const threads = BLOCK_SIZE;
  int const blocks = (num_tokens + threads - 1) / threads;
  TLLM_CHECK_WITH_INFO(blocks == 1, "Current implementation requires single block");

  using BlockRadixRank = cub::BlockRadixRank<BLOCK_SIZE, LOG2_NUM_EXPERTS, false>;
  size_t shared_size = sizeof(typename BlockRadixRank::TempStorage);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = shared_size;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  auto kernel =
      &fusedBuildExpertMapsSortFirstTokenKernel<BLOCK_SIZE, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;

  int device = 0;
  int max_smem_per_block = 0;
  check_cuda_error(cudaGetDevice(&device));
  check_cuda_error(
      cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  if (shared_size >= static_cast<size_t>(max_smem_per_block)) {
    // This should mean that
    // cudaFuncSetAttribute(cutlass::Kernel<GemmKernel>,
    // cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size) wouldn't work.
    return false;
  }

  check_cuda_error(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_size));
  check_cuda_error(
      cudaLaunchKernelEx(&config, kernel, token_selected_experts, permuted_row_to_unpermuted_row,
                         unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens,
                         experts_per_token, start_expert, end_expert, num_experts_per_node));

  return true;
}

template <int EXPERTS_PER_TOKEN, int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(
    int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token,
    int const start_expert, int const end_expert, bool enable_pdl, cudaStream_t stream) {
  int const block_size = num_tokens;
  if (num_tokens > 256) {
    TLLM_LOG_TRACE(
        "Number of tokens %d is greater than 256, which is not supported for fused moe prologues",
        num_tokens);
    return false;
  }

  auto func = &fusedBuildExpertMapsSortFirstTokenDispatch<32, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  if (block_size > 32 && block_size <= 64) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<64, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  } else if (block_size > 64 && block_size <= 128) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<128, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  } else if (block_size > 128 && block_size <= 256) {
    func = &fusedBuildExpertMapsSortFirstTokenDispatch<256, EXPERTS_PER_TOKEN, LOG2_NUM_EXPERTS>;
  }

  return func(token_selected_experts, permuted_row_to_unpermuted_row,
              unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens,
              num_experts_per_node, experts_per_token, start_expert, end_expert, enable_pdl,
              stream);
}

template <int LOG2_NUM_EXPERTS>
bool fusedBuildExpertMapsSortFirstTokenBlockSize(
    int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token,
    int const start_expert, int const end_expert, bool enable_pdl, cudaStream_t stream) {
  auto func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
  switch (experts_per_token) {
    case 1: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<1, LOG2_NUM_EXPERTS>;
      break;
    }
    case 2: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<2, LOG2_NUM_EXPERTS>;
      break;
    }
    case 4: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<4, LOG2_NUM_EXPERTS>;
      break;
    }
    case 6: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<6, LOG2_NUM_EXPERTS>;
      break;
    }
    case 8: {
      func = &fusedBuildExpertMapsSortFirstTokenBlockSize<8, LOG2_NUM_EXPERTS>;
      break;
    }
    default: {
      TLLM_LOG_TRACE("Top-K value %d does not have supported fused moe prologues",
                     experts_per_token);
      return false;
    }
  }
  return func(token_selected_experts, permuted_row_to_unpermuted_row,
              unpermuted_row_to_permuted_row, expert_first_token_offset, num_tokens,
              num_experts_per_node, experts_per_token, start_expert, end_expert, enable_pdl,
              stream);
}

bool fusedBuildExpertMapsSortFirstToken(
    int const* token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int64_t const num_tokens, int const num_experts_per_node, int const experts_per_token,
    int const start_expert, int const end_expert, bool enable_pdl, cudaStream_t stream) {
  // We need enough bits to represent [0, num_experts_per_node+1] (inclusive) i.e.
  // num_experts_per_node + 2 values This is floor(log2(num_experts_per_node+1)) + 1
  int expert_log = static_cast<int>(log2(num_experts_per_node + 1)) + 1;
  if (expert_log <= 9) {
    auto funcs = std::array{&fusedBuildExpertMapsSortFirstTokenBlockSize<1>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<2>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<3>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<4>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<5>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<6>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<7>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<8>,
                            &fusedBuildExpertMapsSortFirstTokenBlockSize<9>};

    return funcs[expert_log - 1](token_selected_experts, permuted_row_to_unpermuted_row,
                                 unpermuted_row_to_permuted_row, expert_first_token_offset,
                                 num_tokens, num_experts_per_node, experts_per_token, start_expert,
                                 end_expert, enable_pdl, stream);
  }
  TLLM_LOG_TRACE("Experts per node %d does not have supported fused moe prologues",
                 num_experts_per_node);
  return false;
}

int64_t computeNumTokensPerBlock(int64_t const num_tokens, int64_t const num_experts_per_node) {
  for (int64_t num_tokens_per_block = 32; num_tokens_per_block <= 1024; num_tokens_per_block *= 2) {
    int64_t const num_blocks_per_seq =
        tensorrt_llm::common::ceilDiv(num_tokens, num_tokens_per_block);
    if (num_blocks_per_seq * num_experts_per_node <= num_tokens_per_block) {
      return num_tokens_per_block;
    }
  }
  return 1024;
}

template <int kNumTokensPerBlock>
__global__ void blockExpertPrefixSumKernel(int const* token_selected_experts,
                                           int* blocked_expert_counts,
                                           int* blocked_row_to_unpermuted_row,
                                           int64_t const num_tokens,
                                           int64_t const num_experts_per_token,
                                           int const start_expert_id) {
  using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // target_expert_id and expert_id are offset by start_expert_id
  int const target_expert_id = blockIdx.x;
  int const block_id = blockIdx.y;
  int const num_blocks_per_seq = gridDim.y;
  int const token_id = block_id * kNumTokensPerBlock + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int expanded_token_id = -1;
  if (token_id < num_tokens) {
    for (int i = 0; i < num_experts_per_token; i++) {
      // TODO(enweiz): Fix uncoalesced access with shared memory.
      int const expert_id =
          token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
      if (expert_id == target_expert_id) {
        expanded_token_id = i * num_tokens + token_id;
        break;
      }
    }
  }

  int const has_matched = expanded_token_id >= 0 ? 1 : 0;
  int index;
  BlockScan(temp_storage).ExclusiveSum(has_matched, index);

  if (has_matched) {
    blocked_row_to_unpermuted_row[target_expert_id * num_tokens + block_id * kNumTokensPerBlock +
                                  index] = expanded_token_id;
  }
  if (threadIdx.x == kNumTokensPerBlock - 1) {
    blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id] = index + has_matched;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void blockExpertPrefixSum(int const* token_selected_experts, int* blocked_expert_counts,
                          int* blocked_row_to_unpermuted_row, int64_t const num_tokens,
                          int64_t const num_experts_per_node, int64_t const num_experts_per_token,
                          int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq,
                          int const start_expert_id, bool enable_pdl, cudaStream_t stream) {
  dim3 const blocks(num_experts_per_node, num_blocks_per_seq);
  dim3 const threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  auto func = blockExpertPrefixSumKernel<1024>;
  if (num_tokens_per_block <= 32) {
    func = blockExpertPrefixSumKernel<32>;
  } else if (num_tokens_per_block <= 64) {
    func = blockExpertPrefixSumKernel<64>;
  } else if (num_tokens_per_block <= 128) {
    func = blockExpertPrefixSumKernel<128>;
  } else if (num_tokens_per_block <= 256) {
    func = blockExpertPrefixSumKernel<256>;
  } else if (num_tokens_per_block <= 512) {
    func = blockExpertPrefixSumKernel<512>;
  }
  cudaLaunchKernelEx(&config, func, token_selected_experts, blocked_expert_counts,
                     blocked_row_to_unpermuted_row, num_tokens, num_experts_per_token,
                     start_expert_id);
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumLargeKernel(int const* blocked_expert_counts,
                                                 int* blocked_expert_counts_cumsum,
                                                 int64_t* expert_first_token_offset,
                                                 int64_t const num_experts_per_node,
                                                 int64_t const num_blocks_per_seq,
                                                 int64_t const num_elem_per_thread) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int offset = threadIdx.x * num_elem_per_thread;
  int cnt = 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Note: Because of limited registers, cannot store thread-level prefix sum or enable #pragma
  // unroll
  for (int i = 0; i < num_elem_per_thread; i++) {
    // TODO(enweiz): Fix uncoalesced access with shared memory.
    if (offset + i < num_experts_per_node * num_blocks_per_seq) {
      cnt += blocked_expert_counts[offset + i];
    }
  }

  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  for (int i = 0; i < num_elem_per_thread; i++) {
    if (offset + i < num_experts_per_node * num_blocks_per_seq) {
      blocked_expert_counts_cumsum[offset + i] = cumsum;
      if ((offset + i) % num_blocks_per_seq == 0) {
        expert_first_token_offset[(offset + i) / num_blocks_per_seq] = cumsum;
      }
      cumsum += blocked_expert_counts[offset + i];
      if ((offset + i) == num_experts_per_node * num_blocks_per_seq - 1) {
        expert_first_token_offset[num_experts_per_node] = cumsum;
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertPrefixSumKernel(int const* blocked_expert_counts,
                                            int* blocked_expert_counts_cumsum,
                                            int64_t* expert_first_token_offset,
                                            int64_t const num_experts_per_node,
                                            int64_t const num_blocks_per_seq) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int const cnt = threadIdx.x < num_experts_per_node * num_blocks_per_seq
                      ? blocked_expert_counts[threadIdx.x]
                      : 0;
  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  if (threadIdx.x < num_experts_per_node * num_blocks_per_seq) {
    blocked_expert_counts_cumsum[threadIdx.x] = cumsum;
    if (threadIdx.x % num_blocks_per_seq == 0) {
      expert_first_token_offset[threadIdx.x / num_blocks_per_seq] = cumsum;
    }
    if (threadIdx.x == num_experts_per_node * num_blocks_per_seq - 1) {
      expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void globalExpertPrefixSum(int const* blocked_expert_counts, int* blocked_expert_counts_cumsum,
                           int64_t* expert_first_token_offset, int64_t const num_experts_per_node,
                           int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq,
                           bool enable_pdl, cudaStream_t stream) {
  int64_t const num_elements = num_experts_per_node * num_blocks_per_seq;

  cudaLaunchConfig_t config;
  config.gridDim = 1;
  config.blockDim = 1024;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  if (num_elements <= 1024) {
    auto func = globalExpertPrefixSumKernel<1024>;
    if (num_elements <= 32) {
      func = globalExpertPrefixSumKernel<32>;
      config.blockDim = 32;
    } else if (num_elements <= 64) {
      func = globalExpertPrefixSumKernel<64>;
      config.blockDim = 64;
    } else if (num_elements <= 128) {
      func = globalExpertPrefixSumKernel<128>;
      config.blockDim = 128;
    } else if (num_elements <= 256) {
      func = globalExpertPrefixSumKernel<256>;
      config.blockDim = 256;
    } else if (num_elements <= 512) {
      func = globalExpertPrefixSumKernel<512>;
      config.blockDim = 512;
    }
    cudaLaunchKernelEx(&config, func, blocked_expert_counts, blocked_expert_counts_cumsum,
                       expert_first_token_offset, num_experts_per_node, num_blocks_per_seq);
  } else {
    auto func = globalExpertPrefixSumLargeKernel<1024>;
    int64_t const num_elem_per_thread = tensorrt_llm::common::ceilDiv(num_elements, 1024);
    cudaLaunchKernelEx(&config, func, blocked_expert_counts, blocked_expert_counts_cumsum,
                       expert_first_token_offset, num_experts_per_node, num_blocks_per_seq,
                       num_elem_per_thread);
  }
}

__global__ void mergeExpertPrefixSumKernel(int const* blocked_expert_counts,
                                           int const* blocked_expert_counts_cumsum,
                                           int const* blocked_row_to_unpermuted_row,
                                           int* permuted_token_selected_experts,
                                           int* permuted_row_to_unpermuted_row,
                                           int* unpermuted_row_to_permuted_row,
                                           int const num_tokens) {
  int const target_expert_id = blockIdx.x;
  int const block_id = blockIdx.y;
  int const num_blocks_per_seq = gridDim.y;
  int const token_id = block_id * blockDim.x + threadIdx.x;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int const cnt = blocked_expert_counts[target_expert_id * num_blocks_per_seq + block_id];
  int const offset = blocked_expert_counts_cumsum[target_expert_id * num_blocks_per_seq + block_id];
  if (threadIdx.x < cnt) {
    int const unpermuted_row =
        blocked_row_to_unpermuted_row[target_expert_id * num_tokens + token_id];
    int const permuted_row = offset + threadIdx.x;
    permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
    permuted_token_selected_experts[permuted_row] = target_expert_id;
    unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void mergeExpertPrefixSum(int const* blocked_expert_counts, int const* blocked_expert_counts_cumsum,
                          int const* blocked_row_to_unpermuted_row,
                          int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
                          int* unpermuted_row_to_permuted_row, int64_t const num_tokens,
                          int64_t const num_experts_per_node, int64_t const num_tokens_per_block,
                          int64_t const num_blocks_per_seq, bool enable_pdl, cudaStream_t stream) {
  dim3 const blocks(num_experts_per_node, num_blocks_per_seq);
  dim3 const threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config, mergeExpertPrefixSumKernel, blocked_expert_counts,
                     blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row,
                     permuted_token_selected_experts, permuted_row_to_unpermuted_row,
                     unpermuted_row_to_permuted_row, num_tokens);
}

template <int kNumTokensPerBlock, typename TokenIndexT>
__global__ void blockExpertAdapterPrefixSumKernel(
    int const* token_selected_experts, TokenIndexT const* token_lora_indices,
    int* blocked_expert_adapter_counts, int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens, int64_t const num_experts_per_token, int const start_expert_id,
    int const num_lora_adapters) {
  using BlockScan = cub::BlockScan<int, kNumTokensPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int const target_expert_id = blockIdx.x;
  int const target_adapter_bucket = blockIdx.y;
  int const block_id = blockIdx.z;
  int const num_blocks_per_seq = gridDim.z;
  int const token_id = block_id * kNumTokensPerBlock + threadIdx.x;
  int const adapter_bucket_count = num_lora_adapters + 1;
  int const target_lora_idx = target_adapter_bucket - 1;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int expanded_token_id = -1;
  if (token_id < num_tokens) {
    int64_t const raw_lora_idx = static_cast<int64_t>(token_lora_indices[token_id]);
    int const lora_bucket = (raw_lora_idx >= 0 && raw_lora_idx < num_lora_adapters)
                                ? static_cast<int>(raw_lora_idx)
                                : -1;
    if (lora_bucket == target_lora_idx) {
      for (int i = 0; i < num_experts_per_token; i++) {
        int const expert_id =
            token_selected_experts[token_id * num_experts_per_token + i] - start_expert_id;
        if (expert_id == target_expert_id) {
          expanded_token_id = i * num_tokens + token_id;
          break;
        }
      }
    }
  }

  int const has_matched = expanded_token_id >= 0 ? 1 : 0;
  int index;
  BlockScan(temp_storage).ExclusiveSum(has_matched, index);

  int const bucket =
      (target_expert_id * adapter_bucket_count + target_adapter_bucket) * num_blocks_per_seq +
      block_id;
  if (has_matched) {
    blocked_row_to_unpermuted_row[static_cast<int64_t>(target_expert_id) * adapter_bucket_count *
                                      num_tokens +
                                  static_cast<int64_t>(target_adapter_bucket) * num_tokens +
                                  block_id * kNumTokensPerBlock + index] = expanded_token_id;
  }
  if (threadIdx.x == kNumTokensPerBlock - 1) {
    blocked_expert_adapter_counts[bucket] = index + has_matched;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <typename TokenIndexT>
void blockExpertAdapterPrefixSum(
    int const* token_selected_experts, TokenIndexT const* token_lora_indices,
    int* blocked_expert_adapter_counts, int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens, int64_t const num_experts_per_node,
    int64_t const num_experts_per_token, int64_t const num_tokens_per_block,
    int64_t const num_blocks_per_seq, int const start_expert_id, int const num_lora_adapters,
    bool enable_pdl, cudaStream_t stream) {
  dim3 const blocks(num_experts_per_node, num_lora_adapters + 1, num_blocks_per_seq);
  dim3 const threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  auto func = blockExpertAdapterPrefixSumKernel<1024, TokenIndexT>;
  if (num_tokens_per_block <= 32) {
    func = blockExpertAdapterPrefixSumKernel<32, TokenIndexT>;
  } else if (num_tokens_per_block <= 64) {
    func = blockExpertAdapterPrefixSumKernel<64, TokenIndexT>;
  } else if (num_tokens_per_block <= 128) {
    func = blockExpertAdapterPrefixSumKernel<128, TokenIndexT>;
  } else if (num_tokens_per_block <= 256) {
    func = blockExpertAdapterPrefixSumKernel<256, TokenIndexT>;
  } else if (num_tokens_per_block <= 512) {
    func = blockExpertAdapterPrefixSumKernel<512, TokenIndexT>;
  }
  cudaLaunchKernelEx(&config, func, token_selected_experts, token_lora_indices,
                     blocked_expert_adapter_counts, blocked_row_to_unpermuted_row, num_tokens,
                     num_experts_per_token, start_expert_id, num_lora_adapters);
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertAdapterPrefixSumLargeKernel(int const* blocked_expert_adapter_counts,
                                                        int* blocked_expert_adapter_counts_cumsum,
                                                        int64_t* expert_first_token_offset,
                                                        int64_t const num_experts_per_node,
                                                        int64_t const num_blocks_per_seq,
                                                        int const adapter_bucket_count,
                                                        int64_t const num_elem_per_thread) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  int offset = threadIdx.x * num_elem_per_thread;
  int cnt = 0;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int64_t const num_elements =
      num_experts_per_node * static_cast<int64_t>(adapter_bucket_count) * num_blocks_per_seq;
  for (int i = 0; i < num_elem_per_thread; i++) {
    if (offset + i < num_elements) {
      cnt += blocked_expert_adapter_counts[offset + i];
    }
  }

  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  for (int i = 0; i < num_elem_per_thread; i++) {
    int const element = offset + i;
    if (element < num_elements) {
      blocked_expert_adapter_counts_cumsum[element] = cumsum;
      if (element % (adapter_bucket_count * num_blocks_per_seq) == 0) {
        expert_first_token_offset[element / (adapter_bucket_count * num_blocks_per_seq)] = cumsum;
      }
      cumsum += blocked_expert_adapter_counts[element];
      if (element == num_elements - 1) {
        expert_first_token_offset[num_experts_per_node] = cumsum;
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <int kNumThreadsPerBlock>
__global__ void globalExpertAdapterPrefixSumKernel(int const* blocked_expert_adapter_counts,
                                                   int* blocked_expert_adapter_counts_cumsum,
                                                   int64_t* expert_first_token_offset,
                                                   int64_t const num_experts_per_node,
                                                   int64_t const num_blocks_per_seq,
                                                   int const adapter_bucket_count) {
  using BlockScan = cub::BlockScan<int, kNumThreadsPerBlock>;
  __shared__ typename BlockScan::TempStorage temp_storage;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int64_t const num_elements =
      num_experts_per_node * static_cast<int64_t>(adapter_bucket_count) * num_blocks_per_seq;
  int const cnt = threadIdx.x < num_elements ? blocked_expert_adapter_counts[threadIdx.x] : 0;
  int cumsum;
  BlockScan(temp_storage).ExclusiveSum(cnt, cumsum);

  if (threadIdx.x < num_elements) {
    blocked_expert_adapter_counts_cumsum[threadIdx.x] = cumsum;
    if (threadIdx.x % (adapter_bucket_count * num_blocks_per_seq) == 0) {
      expert_first_token_offset[threadIdx.x / (adapter_bucket_count * num_blocks_per_seq)] = cumsum;
    }
    if (threadIdx.x == num_elements - 1) {
      expert_first_token_offset[num_experts_per_node] = cumsum + cnt;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void globalExpertAdapterPrefixSum(int const* blocked_expert_adapter_counts,
                                  int* blocked_expert_adapter_counts_cumsum,
                                  int64_t* expert_first_token_offset,
                                  int64_t const num_experts_per_node,
                                  int64_t const num_tokens_per_block,
                                  int64_t const num_blocks_per_seq, int const adapter_bucket_count,
                                  bool enable_pdl, cudaStream_t stream) {
  int64_t const num_elements =
      num_experts_per_node * static_cast<int64_t>(adapter_bucket_count) * num_blocks_per_seq;

  cudaLaunchConfig_t config;
  config.gridDim = 1;
  config.blockDim = 1024;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  if (num_elements <= 1024) {
    auto func = globalExpertAdapterPrefixSumKernel<1024>;
    if (num_elements <= 32) {
      func = globalExpertAdapterPrefixSumKernel<32>;
      config.blockDim = 32;
    } else if (num_elements <= 64) {
      func = globalExpertAdapterPrefixSumKernel<64>;
      config.blockDim = 64;
    } else if (num_elements <= 128) {
      func = globalExpertAdapterPrefixSumKernel<128>;
      config.blockDim = 128;
    } else if (num_elements <= 256) {
      func = globalExpertAdapterPrefixSumKernel<256>;
      config.blockDim = 256;
    } else if (num_elements <= 512) {
      func = globalExpertAdapterPrefixSumKernel<512>;
      config.blockDim = 512;
    }
    cudaLaunchKernelEx(&config, func, blocked_expert_adapter_counts,
                       blocked_expert_adapter_counts_cumsum, expert_first_token_offset,
                       num_experts_per_node, num_blocks_per_seq, adapter_bucket_count);
  } else {
    auto func = globalExpertAdapterPrefixSumLargeKernel<1024>;
    int64_t const num_elem_per_thread = tensorrt_llm::common::ceilDiv(num_elements, 1024);
    cudaLaunchKernelEx(&config, func, blocked_expert_adapter_counts,
                       blocked_expert_adapter_counts_cumsum, expert_first_token_offset,
                       num_experts_per_node, num_blocks_per_seq, adapter_bucket_count,
                       num_elem_per_thread);
  }
}

__global__ void mergeExpertAdapterPrefixSumKernel(
    int const* blocked_expert_adapter_counts, int const* blocked_expert_adapter_counts_cumsum,
    int const* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row, int const num_tokens,
    int const adapter_bucket_count) {
  int const target_expert_id = blockIdx.x;
  int const target_adapter_bucket = blockIdx.y;
  int const block_id = blockIdx.z;
  int const num_blocks_per_seq = gridDim.z;
  int const token_id = block_id * blockDim.x + threadIdx.x;
  int const bucket =
      (target_expert_id * adapter_bucket_count + target_adapter_bucket) * num_blocks_per_seq +
      block_id;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int const cnt = blocked_expert_adapter_counts[bucket];
  int const offset = blocked_expert_adapter_counts_cumsum[bucket];
  if (threadIdx.x < cnt) {
    int const unpermuted_row =
        blocked_row_to_unpermuted_row[static_cast<int64_t>(target_expert_id) *
                                          adapter_bucket_count * num_tokens +
                                      static_cast<int64_t>(target_adapter_bucket) * num_tokens +
                                      token_id];
    int const permuted_row = offset + threadIdx.x;
    permuted_row_to_unpermuted_row[permuted_row] = unpermuted_row;
    permuted_token_selected_experts[permuted_row] = target_expert_id;
    unpermuted_row_to_permuted_row[unpermuted_row] = permuted_row;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

void mergeExpertAdapterPrefixSum(
    int const* blocked_expert_adapter_counts, int const* blocked_expert_adapter_counts_cumsum,
    int const* blocked_row_to_unpermuted_row, int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int64_t const num_tokens, int64_t const num_experts_per_node,
    int64_t const num_tokens_per_block, int64_t const num_blocks_per_seq,
    int const adapter_bucket_count, bool enable_pdl, cudaStream_t stream) {
  dim3 const blocks(num_experts_per_node, adapter_bucket_count, num_blocks_per_seq);
  dim3 const threads(num_tokens_per_block);

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  cudaLaunchKernelEx(&config, mergeExpertAdapterPrefixSumKernel, blocked_expert_adapter_counts,
                     blocked_expert_adapter_counts_cumsum, blocked_row_to_unpermuted_row,
                     permuted_token_selected_experts, permuted_row_to_unpermuted_row,
                     unpermuted_row_to_permuted_row, num_tokens, adapter_bucket_count);
}

// threeStepBuildExpertMapsSortFirstToken uses three kernels to achieve the sort of
// token_selected_experts

// 1. blockExpertPrefixSumKernel launches [num_experts_per_node, num_blocks_per_seq] CTAs; each CTA
// has num_tokens_per_block threads. blocked_row_to_unpermuted_row points to a 2D buffer of size
// [num_experts_per_node, num_tokens], which can be viewed as [num_experts_per_node,
// num_blocks_per_seq] blocks, and each block has num_tokens_per_block tokens. Note that each CTA
// corresponds to a block in blocked_row_to_unpermuted_row. Within each CTA, the threads leverage
// cub::BlockScan to compute the offsets of tokens that activate the target expert. If a thread's
// token activates the target expert, the thread stores its unpermuted_row to the buffer block with
// the offset. In addition, the kernel also stores the expert counts for each block to another 2D
// buffer blocked_expert_counts of size [num_experts_per_node, num_blocks_per_seq].

// 2. globalExpertPrefixSumKernel launches 1 CTA; that CTA has num_experts_per_node *
// num_blocks_per_seq threads. The kernel views blocked_expert_counts as a 1D buffer, and leverages
// cub::BlockScan to compute the prefix sum of the expert counts for each block. The prefix sum is
// stored to blocked_expert_counts_cumsum.

// 3. mergeExpertPrefixSumKernel launches [num_experts_per_node, num_blocks_per_seq] CTAs; each CTA
// has num_tokens_per_block threads. Each CTA obtains the block-level offset from
// blocked_expert_counts_cumsum, and thus compacts blocked_row_to_unpermuted_row to
// permuted_row_to_unpermuted_row. In addition, with the block-level offsets, the kernel fills
// permuted_token_selected_experts.

// computeNumTokensPerBlock decides num_tokens_per_block. Note that both blockExpertPrefixSumKernel
// and globalExpertPrefixSumKernel leverage cub::BlockScan, and their CTA sizes are
// num_tokens_per_block and num_experts_per_node * num_blocks_per_seq, respectively.
// computeNumTokensPerBlock tries to find a minimum CTA size for both kernels, so that the
// block-leval cub::BlockScan can be efficient.

void threeStepBuildExpertMapsSortFirstToken(
    int const* token_selected_experts, int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset, int* blocked_expert_counts,
    int* blocked_expert_counts_cumsum, int* blocked_row_to_unpermuted_row, int64_t const num_tokens,
    int64_t const num_experts_per_node, int64_t const num_experts_per_token,
    int const start_expert_id, bool enable_pdl, cudaStream_t stream) {
  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_tokens, num_experts_per_node);
  int64_t const num_blocks_per_seq =
      tensorrt_llm::common::ceilDiv(num_tokens, num_tokens_per_block);

  blockExpertPrefixSum(token_selected_experts, blocked_expert_counts, blocked_row_to_unpermuted_row,
                       num_tokens, num_experts_per_node, num_experts_per_token,
                       num_tokens_per_block, num_blocks_per_seq, start_expert_id, enable_pdl,
                       stream);
  sync_check_cuda_error(stream);

  globalExpertPrefixSum(blocked_expert_counts, blocked_expert_counts_cumsum,
                        expert_first_token_offset, num_experts_per_node, num_tokens_per_block,
                        num_blocks_per_seq, enable_pdl, stream);
  sync_check_cuda_error(stream);

  mergeExpertPrefixSum(blocked_expert_counts, blocked_expert_counts_cumsum,
                       blocked_row_to_unpermuted_row, permuted_token_selected_experts,
                       permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row, num_tokens,
                       num_experts_per_node, num_tokens_per_block, num_blocks_per_seq, enable_pdl,
                       stream);
}

template <typename TokenIndexT>
void threeStepBuildExpertAdapterMapsSortFirstTokenImpl(
    int const* token_selected_experts, TokenIndexT const* token_lora_indices,
    int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t* expert_first_token_offset,
    int* blocked_expert_adapter_counts, int* blocked_expert_adapter_counts_cumsum,
    int* blocked_row_to_unpermuted_row, int64_t const num_tokens,
    int64_t const num_experts_per_node, int64_t const num_experts_per_token,
    int const start_expert_id, int const num_lora_adapters, bool enable_pdl, cudaStream_t stream) {
  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_tokens, num_experts_per_node);
  int64_t const num_blocks_per_seq =
      tensorrt_llm::common::ceilDiv(num_tokens, num_tokens_per_block);
  int const adapter_bucket_count = num_lora_adapters + 1;

  blockExpertAdapterPrefixSum(token_selected_experts, token_lora_indices,
                              blocked_expert_adapter_counts, blocked_row_to_unpermuted_row,
                              num_tokens, num_experts_per_node, num_experts_per_token,
                              num_tokens_per_block, num_blocks_per_seq, start_expert_id,
                              num_lora_adapters, enable_pdl, stream);
  sync_check_cuda_error(stream);

  globalExpertAdapterPrefixSum(blocked_expert_adapter_counts, blocked_expert_adapter_counts_cumsum,
                               expert_first_token_offset, num_experts_per_node,
                               num_tokens_per_block, num_blocks_per_seq, adapter_bucket_count,
                               enable_pdl, stream);
  sync_check_cuda_error(stream);

  mergeExpertAdapterPrefixSum(blocked_expert_adapter_counts, blocked_expert_adapter_counts_cumsum,
                              blocked_row_to_unpermuted_row, permuted_token_selected_experts,
                              permuted_row_to_unpermuted_row, unpermuted_row_to_permuted_row,
                              num_tokens, num_experts_per_node, num_tokens_per_block,
                              num_blocks_per_seq, adapter_bucket_count, enable_pdl, stream);
}

void threeStepBuildExpertAdapterMapsSortFirstToken(
    int const* token_selected_experts, void const* token_lora_indices,
    bool token_lora_indices_is_int64, int* permuted_token_selected_experts,
    int* permuted_row_to_unpermuted_row, int* unpermuted_row_to_permuted_row,
    int64_t* expert_first_token_offset, int* blocked_expert_adapter_counts,
    int* blocked_expert_adapter_counts_cumsum, int* blocked_row_to_unpermuted_row,
    int64_t const num_tokens, int64_t const num_experts_per_node,
    int64_t const num_experts_per_token, int const start_expert_id, int const num_lora_adapters,
    bool enable_pdl, cudaStream_t stream) {
  if (token_lora_indices_is_int64) {
    threeStepBuildExpertAdapterMapsSortFirstTokenImpl(
        token_selected_experts, static_cast<int64_t const*>(token_lora_indices),
        permuted_token_selected_experts, permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row, expert_first_token_offset, blocked_expert_adapter_counts,
        blocked_expert_adapter_counts_cumsum, blocked_row_to_unpermuted_row, num_tokens,
        num_experts_per_node, num_experts_per_token, start_expert_id, num_lora_adapters, enable_pdl,
        stream);
  } else {
    threeStepBuildExpertAdapterMapsSortFirstTokenImpl(
        token_selected_experts, static_cast<int32_t const*>(token_lora_indices),
        permuted_token_selected_experts, permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row, expert_first_token_offset, blocked_expert_adapter_counts,
        blocked_expert_adapter_counts_cumsum, blocked_row_to_unpermuted_row, num_tokens,
        num_experts_per_node, num_experts_per_token, start_expert_id, num_lora_adapters, enable_pdl,
        stream);
  }
}

template <typename TokenIndexT>
__global__ void countExpertAdapterRowsKernel(int const* permuted_row_to_unpermuted_row,
                                             int64_t const* expert_first_token_offset,
                                             TokenIndexT const* token_lora_indices,
                                             int* adapter_bucket_counts, int64_t const num_tokens,
                                             int const adapter_bucket_count) {
  int const expert = blockIdx.x;
  int64_t const begin = expert_first_token_offset[expert];
  int64_t const end = expert_first_token_offset[expert + 1];
  for (int64_t row = begin + threadIdx.x; row < end; row += blockDim.x) {
    int const source_index = permuted_row_to_unpermuted_row[row] % num_tokens;
    int64_t const raw_lora_idx = static_cast<int64_t>(token_lora_indices[source_index]);
    int const bucket = (raw_lora_idx >= 0 && raw_lora_idx < adapter_bucket_count - 1)
                           ? static_cast<int>(raw_lora_idx) + 1
                           : 0;
    atomicAdd(adapter_bucket_counts + expert * adapter_bucket_count + bucket, 1);
  }
}

__global__ void buildExpertAdapterRowStartsKernel(int const* adapter_bucket_counts,
                                                  int* adapter_bucket_offsets,
                                                  int* adapter_bucket_cursors,
                                                  int64_t const* expert_first_token_offset,
                                                  int const adapter_bucket_count) {
  int const expert = blockIdx.x;
  if (threadIdx.x != 0) {
    return;
  }
  int running = static_cast<int>(expert_first_token_offset[expert]);
  for (int bucket = 0; bucket < adapter_bucket_count; ++bucket) {
    int const idx = expert * adapter_bucket_count + bucket;
    adapter_bucket_offsets[idx] = running;
    running += adapter_bucket_counts[idx];
    adapter_bucket_cursors[idx] = 0;
  }
}

template <typename TokenIndexT>
__global__ void scatterExpertAdapterRowsKernel(
    int const* old_permuted_row_to_unpermuted_row, int* new_permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int* permuted_token_selected_experts,
    int64_t const* expert_first_token_offset, TokenIndexT const* token_lora_indices,
    int* adapter_bucket_cursors, int const* adapter_bucket_offsets, int64_t const num_tokens,
    int const adapter_bucket_count) {
  int const expert = blockIdx.x;
  int64_t const begin = expert_first_token_offset[expert];
  int64_t const end = expert_first_token_offset[expert + 1];
  for (int64_t old_row = begin + threadIdx.x; old_row < end; old_row += blockDim.x) {
    int const unpermuted_row = old_permuted_row_to_unpermuted_row[old_row];
    int const source_index = unpermuted_row % num_tokens;
    int64_t const raw_lora_idx = static_cast<int64_t>(token_lora_indices[source_index]);
    int const bucket = (raw_lora_idx >= 0 && raw_lora_idx < adapter_bucket_count - 1)
                           ? static_cast<int>(raw_lora_idx) + 1
                           : 0;
    int const idx = expert * adapter_bucket_count + bucket;
    int const offset = atomicAdd(adapter_bucket_cursors + idx, 1);
    int const new_row = adapter_bucket_offsets[idx] + offset;
    new_permuted_row_to_unpermuted_row[new_row] = unpermuted_row;
    permuted_token_selected_experts[new_row] = expert;
    unpermuted_row_to_permuted_row[unpermuted_row] = new_row;
  }
}

__global__ void copyRegroupedRowsKernel(int const* src, int* dst, int64_t num_rows) {
  for (int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; row < num_rows;
       row += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    dst[row] = src[row];
  }
}

template <typename TokenIndexT>
void regroupExpertRowsByLoraAdapterImpl(
    int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t const* expert_first_token_offset,
    TokenIndexT const* token_lora_indices, int* adapter_bucket_counts, int* adapter_bucket_offsets,
    int* temp_permuted_row_to_unpermuted_row, int64_t num_tokens, int64_t expanded_num_rows,
    int num_experts_per_node, int num_lora_adapters, bool copy_regrouped_rows,
    cudaStream_t stream) {
  int const adapter_bucket_count = num_lora_adapters + 1;
  int const bucket_count = num_experts_per_node * adapter_bucket_count;
  TLLM_CUDA_CHECK(cudaMemsetAsync(adapter_bucket_counts, 0, bucket_count * sizeof(int), stream));

  constexpr int threads = 256;
  countExpertAdapterRowsKernel<<<num_experts_per_node, threads, 0, stream>>>(
      permuted_row_to_unpermuted_row, expert_first_token_offset, token_lora_indices,
      adapter_bucket_counts, num_tokens, adapter_bucket_count);
  buildExpertAdapterRowStartsKernel<<<num_experts_per_node, 1, 0, stream>>>(
      adapter_bucket_counts, adapter_bucket_offsets, adapter_bucket_counts,
      expert_first_token_offset, adapter_bucket_count);
  scatterExpertAdapterRowsKernel<<<num_experts_per_node, threads, 0, stream>>>(
      permuted_row_to_unpermuted_row, temp_permuted_row_to_unpermuted_row,
      unpermuted_row_to_permuted_row, permuted_token_selected_experts, expert_first_token_offset,
      token_lora_indices, adapter_bucket_counts, adapter_bucket_offsets, num_tokens,
      adapter_bucket_count);
  if (copy_regrouped_rows) {
    int const copy_blocks =
        static_cast<int>(std::min<int64_t>((expanded_num_rows + threads - 1) / threads, 65535));
    copyRegroupedRowsKernel<<<copy_blocks, threads, 0, stream>>>(
        temp_permuted_row_to_unpermuted_row, permuted_row_to_unpermuted_row, expanded_num_rows);
  }
}

bool regroupExpertRowsByLoraAdapter(
    int* permuted_token_selected_experts, int* permuted_row_to_unpermuted_row,
    int* unpermuted_row_to_permuted_row, int64_t const* expert_first_token_offset,
    void const* token_lora_indices, bool token_lora_indices_is_int64, int* adapter_bucket_counts,
    int* adapter_bucket_offsets, int* temp_permuted_row_to_unpermuted_row, int64_t num_tokens,
    int64_t expanded_num_rows, int num_experts_per_node, int num_lora_adapters,
    int64_t routing_bucket_capacity, bool copy_regrouped_rows, cudaStream_t stream) {
  int const adapter_bucket_count = num_lora_adapters + 1;
  if (token_lora_indices == nullptr || num_lora_adapters <= 0 ||
      static_cast<int64_t>(num_experts_per_node) * adapter_bucket_count > routing_bucket_capacity) {
    return false;
  }
  if (token_lora_indices_is_int64) {
    regroupExpertRowsByLoraAdapterImpl(
        permuted_token_selected_experts, permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row, expert_first_token_offset,
        static_cast<int64_t const*>(token_lora_indices), adapter_bucket_counts,
        adapter_bucket_offsets, temp_permuted_row_to_unpermuted_row, num_tokens, expanded_num_rows,
        num_experts_per_node, num_lora_adapters, copy_regrouped_rows, stream);
  } else {
    regroupExpertRowsByLoraAdapterImpl(
        permuted_token_selected_experts, permuted_row_to_unpermuted_row,
        unpermuted_row_to_permuted_row, expert_first_token_offset,
        static_cast<int32_t const*>(token_lora_indices), adapter_bucket_counts,
        adapter_bucket_offsets, temp_permuted_row_to_unpermuted_row, num_tokens, expanded_num_rows,
        num_experts_per_node, num_lora_adapters, copy_regrouped_rows, stream);
  }
  return true;
}

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices,
                                                      int64_t const arr_length, T const target) {
  int64_t low = 0, high = arr_length - 1, target_location = -1;
  while (low <= high) {
    int64_t mid = (low + high) / 2;

    if (sorted_indices[mid] >= target) {
      high = mid - 1;
    } else {
      low = mid + 1;
      target_location = mid;
    }
  }
  return target_location + 1;
}

template <class T>
using sizeof_bits = cutlass::sizeof_bits<
    typename cutlass_kernels::TllmToCutlassTypeAdapter<std::remove_cv_t<T>>::type>;

// Function to safely offset an pointer that may contain sub-byte types (FP4/INT4)
template <class T>
__host__ __device__ constexpr T* safe_inc_ptr(T* ptr, size_t offset) {
  constexpr int adjustment = (sizeof_bits<T>::value < 8) ? (8 / sizeof_bits<T>::value) : 1;
  assert(offset % adjustment == 0 && "Attempt to offset index to sub-byte");
  return ptr + offset / adjustment;
}

__host__ __device__ constexpr int64_t getOffsetWeightSF(
    int64_t expert_id, int64_t gemm_n, int64_t gemm_k,
    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  auto function = [=](int64_t min_n_dim_alignment, int64_t min_k_dim_alignment,
                      int64_t block_size) {
    int64_t padded_gemm_n =
        TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_n, min_n_dim_alignment);
    int64_t padded_gemm_k =
        TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_k, min_k_dim_alignment);
    assert(gemm_k % block_size == 0);
    return expert_id * padded_gemm_n * padded_gemm_k / block_size;
  };
  switch (scaling_type) {
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE:
      return 0;  // No scaling factors, no offset
  }

  assert(false && "Unrecognized scaling type");
  return 0;
}

__host__ __device__ constexpr int64_t getOffsetActivationSF(
    int64_t expert_id, int64_t token_offset, int64_t gemm_k,
    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  auto function = [=](int64_t min_n_dim_alignment, int64_t min_k_dim_alignment,
                      int64_t block_size) {
    // This formulation ensures that:
    // `sf_offset[i + 1] - sf_offset[i] >= padded(token_offset[i + 1] - token_offset[i])`
    // is true for all possible token distributions.
    int64_t padded_sf_start_offset = TmaWarpSpecializedGroupedGemmInput::alignToSfDim(
        token_offset + expert_id * (min_n_dim_alignment - 1), min_n_dim_alignment);
    int64_t padded_gemm_k =
        TmaWarpSpecializedGroupedGemmInput::alignToSfDim(gemm_k, min_k_dim_alignment);
    assert(gemm_k % block_size == 0);
    assert(padded_gemm_k % block_size == 0);
    return padded_sf_start_offset * padded_gemm_k / block_size;
  };
  switch (scaling_type) {
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX,
                      TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4:
      return function(TmaWarpSpecializedGroupedGemmInput::MinNDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4,
                      TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize);
    case TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE:
      return 0;  // No scaling factors, no offset
  }

  assert(false && "Unrecognized scaling type");
  return 0;
}

template <class GemmOutputType, class QuantizedType, class ComputeElem, int VecSize,
          bool DISABLE_FP4_QUANT_FAST_MATH = false, typename NVFP4_4OVER6_CONFIG = std::false_type>
__device__ auto quantizePackedFPXValue(
    ComputeElem& post_act_val, float global_scale_val, int64_t num_tokens_before_expert,
    int64_t expert_id, int64_t token_id, int64_t elem_idx, int64_t num_cols,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType scaling_type) {
  constexpr bool is_fp8 = std::is_same_v<QuantizedType, __nv_fp8_e4m3>;
  static constexpr int NumThreadsPerSF = VecSize / CVT_ELTS_PER_THREAD;
  // Quantize the input to FP4
  static_assert(std::is_same_v<GemmOutputType, __nv_bfloat16> ||
                std::is_same_v<GemmOutputType, half>);
  static_assert(ComputeElem::kElements == CVT_ELTS_PER_THREAD);
  PackedVec<GemmOutputType> packed_vec{};
  for (int i = 0; i < CVT_ELTS_PER_THREAD / 2; i++) {
    packed_vec.elts[i].x = static_cast<GemmOutputType>(post_act_val[i * 2 + 0]);
    packed_vec.elts[i].y = static_cast<GemmOutputType>(post_act_val[i * 2 + 1]);
  }

  // We need to offset into the scaling factors for just this expert
  auto act_sf_expert = act_sf_flat + getOffsetActivationSF(expert_id, num_tokens_before_expert,
                                                           num_cols, scaling_type);

  // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this
  // expert
  auto sf_out =
      cvt_quant_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF>(
          std::nullopt /* batchIdx */, token_id - num_tokens_before_expert, elem_idx,
          std::nullopt /* numRows */, num_cols / VecSize, act_sf_expert,
          QuantizationSFLayout::SWIZZLED_128x4);

  // Do the conversion and set the output and scaling factor
  auto func = [&]() {
    if constexpr (is_fp8) {
      return [](PackedVec<GemmOutputType>& vec, float /* ignored */, uint8_t* SFout) -> uint64_t {
        static_assert(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize == VecSize);
        return cvt_warp_fp16_to_mxfp8<GemmOutputType, VecSize, CVT_ELTS_PER_THREAD>(vec, SFout);
      };
    } else {
      if (scaling_type == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4) {
        return &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, CVT_ELTS_PER_THREAD, false,
                                     DISABLE_FP4_QUANT_FAST_MATH, NVFP4_4OVER6_CONFIG>;
      }
      return &cvt_warp_fp16_to_fp4<GemmOutputType, VecSize, CVT_ELTS_PER_THREAD, true,
                                   DISABLE_FP4_QUANT_FAST_MATH>;
    }
  }();

  return func(packed_vec, global_scale_val, sf_out);
}

template <int VecSize, int ElementsPerThread>
__device__ void writeSF(int64_t num_tokens_before_expert, int64_t expert_id,
                        int64_t source_token_id, int64_t token_id, int64_t elem_idx,
                        int64_t num_cols,
                        TmaWarpSpecializedGroupedGemmInput::ElementSF* act_sf_flat,
                        TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf,
                        bool const swizzled_input_sf = true) {
  static constexpr int NumThreadsPerSF = VecSize / ElementsPerThread;

  // We need to offset into the scaling factors for just this expert
  auto act_sf_expert =
      act_sf_flat + getOffsetActivationSF(
                        expert_id, num_tokens_before_expert, num_cols,
                        (VecSize == TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize)
                            ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                            : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);

  // Use `token - num_tokens_before_expert` because we want this to be relative to the start of this
  // expert
  auto sf_out =
      cvt_quant_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF, NumThreadsPerSF>(
          std::nullopt /* batchIdx */, token_id - num_tokens_before_expert, elem_idx,
          std::nullopt /* numRows */, num_cols / VecSize, act_sf_expert,
          QuantizationSFLayout::SWIZZLED_128x4);
  if (sf_out) {
    if (input_sf) {
      if (swizzled_input_sf) {
        auto const sf_in =
            cvt_quant_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF,
                                        NumThreadsPerSF>(
                std::nullopt /* batchIdx */, source_token_id, elem_idx, std::nullopt /* numRows */,
                num_cols / VecSize,
                const_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(input_sf),
                QuantizationSFLayout::SWIZZLED_128x4);
        *sf_out = *sf_in;
      } else {
        auto const sf_in =
            cvt_quant_get_sf_out_offset<TmaWarpSpecializedGroupedGemmInput::ElementSF,
                                        NumThreadsPerSF>(
                std::nullopt /* batchIdx */, source_token_id, elem_idx, std::nullopt /* numRows */,
                num_cols / VecSize,
                const_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF*>(input_sf),
                QuantizationSFLayout::LINEAR);
        *sf_out = *sf_in;
      }
    } else {
      *sf_out = 0x00;
    }
  }
}

// ====================== Compute FP8 dequant scale only ===============================
__global__ void computeFP8DequantScaleKernel(float const** alpha_scale_ptr_array,
                                             int64_t const num_experts_per_node,
                                             float const* fp8_dequant) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

  assert(fp8_dequant != nullptr);
  alpha_scale_ptr_array[expert] = fp8_dequant + expert;
}

float const** computeFP8DequantScale(float const** alpha_scale_ptr_array,
                                     int const num_experts_per_node, float const* fp8_dequant,
                                     cudaStream_t stream) {
  if (!fp8_dequant) {
    return nullptr;
  }

  int const threads = std::min(1024, num_experts_per_node);
  int const blocks = (num_experts_per_node + threads - 1) / threads;

  computeFP8DequantScaleKernel<<<blocks, threads, 0, stream>>>(alpha_scale_ptr_array,
                                                               num_experts_per_node, fp8_dequant);

  return alpha_scale_ptr_array;
}

template <class BSConfig>
__device__ void setupFP4BlockScalingFactors(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int expert, int gemm_m, int gemm_n, int gemm_k,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* weight_block_scale,
    int64_t num_tokens_before_expert) {
  assert(layout_info.fpX_block_scaling_factors_stride_act);
  assert(layout_info.fpX_block_scaling_factors_stride_weight);

  auto stride_act_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(
      layout_info.fpX_block_scaling_factors_stride_act);
  auto stride_weight_ptr = reinterpret_cast<typename BSConfig::LayoutSF*>(
      layout_info.fpX_block_scaling_factors_stride_weight);
  if (layout_info.swap_ab) {
    // M & N swapped for transpose
    stride_act_ptr[expert] = BSConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
    stride_weight_ptr[expert] = BSConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, (int)1));
  } else {
    stride_act_ptr[expert] = BSConfig::tile_atom_to_shape_SFA(
        cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, (int)1));
    stride_weight_ptr[expert] = BSConfig::tile_atom_to_shape_SFB(
        cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, (int)1));
  }

  // This assert validates our current assumption that A&B can be safely transposed without needing
  // to modify
  assert(
      BSConfig::tile_atom_to_shape_SFB(
          cute::make_shape((int)gemm_n, (int)gemm_m, (int)gemm_k, 1)) ==
      BSConfig::tile_atom_to_shape_SFA(cute::make_shape((int)gemm_m, (int)gemm_n, (int)gemm_k, 1)));

  auto scaling_type =
      std::is_same_v<BSConfig, TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig>
          ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
          : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
  layout_info.fpX_block_scaling_factors_act[expert] =
      fp4_act_flat + getOffsetActivationSF(expert, num_tokens_before_expert, gemm_k, scaling_type);

  layout_info.fpX_block_scaling_factors_weight[expert] =
      weight_block_scale + getOffsetWeightSF(expert, gemm_n, gemm_k, scaling_type);
}

__device__ void computeTmaWarpSpecializedInputStrides(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int gemm_m, int gemm_n, int gemm_k,
    int64_t out_idx) {
  if (layout_info.swap_ab) {
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideB*>(
        layout_info.stride_act)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideB{},
                                         cute::make_shape(gemm_m, gemm_k, 1));
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideA*>(
        layout_info.stride_weight)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideA{},
                                         cute::make_shape(gemm_n, gemm_k, 1));
  } else {
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideA*>(
        layout_info.stride_act)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideA{},
                                         cute::make_shape(gemm_m, gemm_k, 1));
    reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideB*>(
        layout_info.stride_weight)[out_idx] =
        cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideB{},
                                         cute::make_shape(gemm_n, gemm_k, 1));
  }
  if (layout_info.stride_c) {
    // TODO Enable 1xN bias matrix as C
    assert(false && "CUTLASS does not support a 1xN bias");
  }
  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE) {
    if (layout_info.swap_ab) {
      reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideD_T*>(
          layout_info.stride_d)[out_idx] =
          cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideD_T{},
                                           cute::make_shape(gemm_n, gemm_m, 1));
    } else {
      reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::StrideD*>(
          layout_info.stride_d)[out_idx] =
          cutlass::make_cute_packed_stride(TmaWarpSpecializedGroupedGemmInput::StrideD{},
                                           cute::make_shape(gemm_m, gemm_n, 1));
    }
  }
  if (layout_info.int4_groupwise_params.enabled) {
    layout_info.int4_groupwise_params.stride_s_a[out_idx] = cutlass::make_cute_packed_stride(
        TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::StrideSFA{},
        cute::make_shape(
            gemm_n,
            gemm_k /
                (layout_info.int4_groupwise_params.use_wfp4a16
                     ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size
                     : TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size),
            1));
  }
}

template <class T, class WeightType, class OutputType, class ScaleBiasType>
__device__ void computeTmaWarpSpecializedInputPointers(
    TmaWarpSpecializedGroupedGemmInput& layout_info, int64_t gemm_m, int64_t gemm_n, int64_t gemm_k,
    int num_tokens_before_expert, int64_t expert, T const* in, WeightType const* weights,
    TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const* w4a8_weight_scale,
    ScaleBiasType const* bias, OutputType* output, float const* router_scales,
    int const* permuted_row_to_unpermuted_row, int64_t const out_idx) {
  // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
  layout_info.ptr_act[out_idx] = safe_inc_ptr(in, num_tokens_before_expert * gemm_k);

  // Each expert's weight matrix is a constant size NxK, get the matrix at index `expert`
  layout_info.ptr_weight[out_idx] = safe_inc_ptr(weights, expert * (gemm_n * gemm_k));

  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE) {
    // The output prior to this contains N elements per token, with `num_tokens_before_expert`
    // tokens
    layout_info.ptr_d[out_idx] = safe_inc_ptr(output, num_tokens_before_expert * gemm_n);
  }
  if (layout_info.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
    layout_info.fused_finalize_epilogue.ptr_source_token_index[expert] =
        permuted_row_to_unpermuted_row + num_tokens_before_expert;
    layout_info.fused_finalize_epilogue.ptr_router_scales[expert] =
        router_scales + num_tokens_before_expert;
    if (layout_info.fused_finalize_epilogue.ptr_bias != nullptr) {
      layout_info.fused_finalize_epilogue.ptr_bias[expert] = bias + gemm_n * expert;
    }
  }
  if (layout_info.int4_groupwise_params.enabled) {
    // The group size of wfp4a16 is multiplied by 2 because each scale uses 1 byte instead of 2
    // bytes
    layout_info.int4_groupwise_params.ptr_s_a[out_idx] = safe_inc_ptr(
        w4a8_weight_scale,
        expert *
            (gemm_n * gemm_k /
             (layout_info.int4_groupwise_params.use_wfp4a16
                  ? TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::wfp4a16_group_size * 2
                  : TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::int4_group_size)));
  }
}

// TODO Some of this setup could be cached
template <class T, class WeightType, class OutputType, class ScaleBiasType>
__global__ void computeStridesTmaWarpSpecializedKernel(
    int64_t const* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput layout_info1,
    TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens,
    int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k,
    int64_t const num_experts_per_node, T const* gemm1_in, T const* gemm2_in,
    WeightType const* weights1, WeightType const* weights2, float const* alpha_scale_flat1,
    float const* alpha_scale_flat2,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2, QuantParams quant_params,
    ScaleBiasType const* bias1, ScaleBiasType const* bias2, OutputType* gemm1_output,
    OutputType* gemm2_output, float const* router_scales,
    int const* permuted_row_to_unpermuted_row) {
  // First, compute the global tid. We only need 1 thread per expert.
  int const expert = blockIdx.x * blockDim.x + threadIdx.x;
  if (expert >= num_experts_per_node) {
    return;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  // Both gemms use the same token offset
  auto const num_tokens_before_expert = expert_first_token_offset[expert];
  auto const num_tokens_including_expert = expert_first_token_offset[expert + 1];
  auto const num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
  auto const gemm_m = num_tokens_to_expert;

  // M and N transposed since we are using the #tokens as the N dimension
  layout_info1.shape_info.problem_shapes[expert] =
      TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
          layout_info1.swap_ab ? gemm1_n : gemm_m, layout_info1.swap_ab ? gemm_m : gemm1_n,
          gemm1_k);
  layout_info2.shape_info.problem_shapes[expert] =
      TmaWarpSpecializedGroupedGemmInput::ProblemShape::UnderlyingProblemShape(
          layout_info2.swap_ab ? gemm2_n : gemm_m, layout_info2.swap_ab ? gemm_m : gemm2_n,
          gemm2_k);

  if (layout_info1.int4_groupwise_params.enabled) {
    layout_info1.int4_groupwise_params.shape.problem_shapes[expert] =
        TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::
            UnderlyingProblemShape(layout_info1.swap_ab ? gemm1_n : gemm_m,
                                   layout_info1.swap_ab ? gemm_m : gemm1_n, gemm1_k);
  }

  if (layout_info2.int4_groupwise_params.enabled) {
    layout_info2.int4_groupwise_params.shape.problem_shapes[expert] =
        TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::ProblemShapeInt::
            UnderlyingProblemShape(layout_info2.swap_ab ? gemm2_n : gemm_m,
                                   layout_info2.swap_ab ? gemm_m : gemm2_n, gemm2_k);
  }

  // Skip expensive stride/pointer/SF setup for experts with no assigned tokens.
  // All problem shapes (including int4_groupwise) are initialized above so CUTLASS
  // can correctly traverse the problem list. The remaining work (alpha scales,
  // block scaling factors, strides, pointers) is only needed for active experts.
  // For decode (1 token, top_k=8, 128 experts), this skips ~120 of 128 experts.
  if (gemm_m == 0) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaTriggerProgrammaticLaunchCompletion();
#endif
    return;
  }

  if (alpha_scale_flat1 && alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array[expert] = alpha_scale_flat1 + expert;
    layout_info2.alpha_scale_ptr_array[expert] = alpha_scale_flat2 + expert;
  }

  auto setupIfSelected = [&](auto bs_config, auto quant_type) {
    if (quant_type.fc1.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(
          layout_info1, expert, gemm_m, gemm1_n, gemm1_k, fp4_act_flat1,
          quant_type.fc1.weight_block_scale, num_tokens_before_expert);
    }
    if (quant_type.fc2.weight_block_scale) {
      setupFP4BlockScalingFactors<decltype(bs_config)>(
          layout_info2, expert, gemm_m, gemm2_n, gemm2_k, fp4_act_flat2,
          quant_type.fc2.weight_block_scale, num_tokens_before_expert);
    }
  };

  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaledConfig{}, quant_params.fp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{},
                  quant_params.fp8_mxfp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{},
                  quant_params.mxfp8_mxfp4);
  setupIfSelected(TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaledConfig{},
                  quant_params.mxfp8_mxfp8);

  assert(gemm_m <= INT32_MAX);
  assert(gemm1_n > 0 && gemm1_n <= INT32_MAX);
  assert(gemm1_k > 0 && gemm1_k <= INT32_MAX);
  assert(gemm2_n > 0 && gemm2_n <= INT32_MAX);
  assert(gemm2_k > 0 && gemm2_k <= INT32_MAX);
  computeTmaWarpSpecializedInputStrides(layout_info1, gemm_m, gemm1_n, gemm1_k, expert);
  computeTmaWarpSpecializedInputStrides(layout_info2, gemm_m, gemm2_n, gemm2_k, expert);

  computeTmaWarpSpecializedInputPointers(
      layout_info1, gemm_m, gemm1_n, gemm1_k, num_tokens_before_expert, expert, gemm1_in, weights1,
      reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
          quant_params.groupwise.fc1.weight_scales),
      bias1, gemm1_output, nullptr, nullptr, expert);
  computeTmaWarpSpecializedInputPointers(
      layout_info2, gemm_m, gemm2_n, gemm2_k, num_tokens_before_expert, expert, gemm2_in, weights2,
      reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams::SFA const*>(
          quant_params.groupwise.fc2.weight_scales),
      bias2, gemm2_output, router_scales, permuted_row_to_unpermuted_row, expert);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// ========================== Permutation things =======================================

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input) {
  cutlass::NumericArrayConverter<typename U::Element, typename T::Element, U::kElements> converter;
  return converter(input);
}

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with
// finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since
// we will have to duplicate some rows in the input matrix to match the dimensions. Duplicates will
// always get routed to separate experts in the end.

// Note that the permuted_row_to_unpermuted_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ...
// (k-1)*rows_in_input all map to row 0 in the original matrix. Thus, to know where to read in the
// source matrix, we simply take the modulus of the expanded index.

constexpr static int EXPAND_THREADS_PER_BLOCK = 256;
constexpr static int EXPAND_SCATTER_MAX_K = 16;

template <class ActivationsType>
__global__ void expandInputRowsScatterKernel(ActivationsType const* unpermuted_input,
                                             ActivationsType* permuted_output,
                                             float const* unpermuted_scales, float* permuted_scales,
                                             int const* unpermuted_row_to_permuted_row,
                                             int const* token_selected_experts,
                                             int64_t const num_tokens, int64_t const hidden_size,
                                             int64_t const k, int const num_experts_per_node,
                                             int const start_expert_id) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  constexpr int64_t ELEM_PER_THREAD = 128 / sizeof_bits<ActivationsType>::value;
  static_assert(ELEM_PER_THREAD > 0, "Expected a valid vector width");

  int64_t const num_elems_in_col = hidden_size / ELEM_PER_THREAD;
  assert(hidden_size % ELEM_PER_THREAD == 0);

  using DataElem = cutlass::Array<ActivationsType, ELEM_PER_THREAD>;
  auto const* input_v = reinterpret_cast<DataElem const*>(unpermuted_input);
  auto* output_v = reinterpret_cast<DataElem*>(permuted_output);

  __shared__ int permuted_rows[EXPAND_SCATTER_MAX_K];

  for (int64_t source_row = blockIdx.x; source_row < num_tokens; source_row += gridDim.x) {
    if (threadIdx.x < k) {
      int const k_idx = threadIdx.x;
      int const expert_id = token_selected_experts[source_row * k + k_idx] - start_expert_id;
      int permuted_row = -1;
      if (expert_id >= 0 && expert_id < num_experts_per_node) {
        int64_t const expanded_original_row = source_row + static_cast<int64_t>(k_idx) * num_tokens;
        permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];
        int64_t const expanded_rows = num_tokens * k;
        if (permuted_row < 0 || static_cast<int64_t>(permuted_row) >= expanded_rows) {
          permuted_row = -1;
        }
      }
      permuted_rows[k_idx] = permuted_row;
      if (permuted_scales && permuted_row >= 0) {
        int64_t const source_k_idx = source_row * k + k_idx;
        permuted_scales[permuted_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
      }
    }
    __syncthreads();

    auto const* source_row_ptr = input_v + source_row * num_elems_in_col;
    for (int64_t elem_index = threadIdx.x; elem_index < num_elems_in_col;
         elem_index += EXPAND_THREADS_PER_BLOCK) {
      auto const in_vec = source_row_ptr[elem_index];
      CUTLASS_PRAGMA_UNROLL
      for (int k_idx = 0; k_idx < EXPAND_SCATTER_MAX_K; ++k_idx) {
        if (k_idx >= k) {
          break;
        }
        int const permuted_row = permuted_rows[k_idx];
        if (permuted_row >= 0) {
          output_v[static_cast<int64_t>(permuted_row) * num_elems_in_col + elem_index] = in_vec;
        }
      }
    }
    __syncthreads();
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <class InputActivationsType, class ExpandedActivationsType,
          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType BlockScalingType,
          bool PRE_QUANT_AWQ, bool DISABLE_FP4_QUANT_FAST_MATH = false,
          typename NVFP4_4OVER6_CONFIG = std::false_type>
__global__ void expandInputRowsKernel(
    InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,
    float const* unpermuted_scales, float* permuted_scales,
    int const* permuted_row_to_unpermuted_row, int64_t const num_tokens, int64_t const hidden_size,
    int64_t const k, float const* fc1_act_global_scale, bool use_per_expert_act_scale,
    int64_t const* expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, bool const swizzled_input_sf,
    int64_t const num_experts_per_node, InputActivationsType const* prequant_scales = nullptr) {
  static_assert(BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE ||
                    !PRE_QUANT_AWQ,
                "AWQ and Block Scaling are mutually exclusive");
  constexpr bool is_mxfp8 =
      std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX &&
      !PRE_QUANT_AWQ;
  constexpr bool is_mxfp8_input = is_mxfp8 && std::is_same_v<InputActivationsType, __nv_fp8_e4m3>;
  constexpr bool need_mxfp8_quant = is_mxfp8 && !is_mxfp8_input;

#ifdef ENABLE_FP4
  constexpr bool is_nvfp4 =
      std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1> &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4 &&
      !PRE_QUANT_AWQ;
  constexpr bool is_nvfp4_input = is_nvfp4 && std::is_same_v<InputActivationsType, __nv_fp4_e2m1>;
  constexpr bool need_nvfp4_quant = is_nvfp4 && !is_nvfp4_input;
#else
  constexpr bool is_nvfp4 = false;
  constexpr bool is_nvfp4_input = false;
  constexpr bool need_nvfp4_quant = false;
#endif

  static_assert(need_nvfp4_quant || need_mxfp8_quant || PRE_QUANT_AWQ ||
                    std::is_same_v<InputActivationsType, ExpandedActivationsType>,
                "Only NVFP4, MXFP8 and WINT4_AFP8 supports outputting a different format as part "
                "of the expansion");
  static_assert(
      !IsNVFP44Over6Config<NVFP4_4OVER6_CONFIG>::value ||
          BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4,
      "NVFP4 4over6 requires NVFP4 block scaling");

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  constexpr int VecSize = is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                                   : TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;

  constexpr int64_t ELEM_PER_THREAD = (is_nvfp4 || is_mxfp8)
                                          ? CVT_ELTS_PER_THREAD
                                          : (128 / sizeof_bits<InputActivationsType>::value);

  // This should be VecSize * 4 elements
  // We assume at least VecSize alignment or the quantization will fail
  constexpr int64_t min_k_dim_alignment =
      is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4
               : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX;
  int64_t const padded_hidden_size =
      TmaWarpSpecializedGroupedGemmInput::alignToSfDim(hidden_size, min_k_dim_alignment);

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];
  for (int64_t permuted_row = blockIdx.x; permuted_row < num_valid_tokens;
       permuted_row += gridDim.x) {
    int64_t const unpermuted_row = permuted_row_to_unpermuted_row[permuted_row];

    // Load 128-bits per thread

    constexpr int64_t ELEM_PER_BYTE = is_nvfp4_input ? 2 : 1;
    using DataElem = std::conditional_t<
        is_nvfp4_input, uint32_t,
        std::conditional_t<is_mxfp8_input, uint64_t,
                           cutlass::Array<InputActivationsType, ELEM_PER_THREAD>>>;
    using OutputElem = std::conditional_t<
        is_nvfp4, uint32_t,
        std::conditional_t<is_mxfp8, uint64_t,
                           cutlass::Array<ExpandedActivationsType, ELEM_PER_THREAD>>>;

    // Duplicate and permute rows
    int64_t const source_k_rank = unpermuted_row / num_tokens;
    int64_t const source_row = unpermuted_row % num_tokens;

    auto const* source_row_ptr = reinterpret_cast<DataElem const*>(
        unpermuted_input + source_row * hidden_size / ELEM_PER_BYTE);
    // Cast first to handle when this is FP4
    auto* dest_row_ptr = reinterpret_cast<OutputElem*>(permuted_output) +
                         permuted_row * hidden_size / ELEM_PER_THREAD;

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = EXPAND_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = hidden_size / ELEM_PER_THREAD;
    assert(hidden_size % ELEM_PER_THREAD == 0);
    assert(hidden_size % VecSize == 0);

    if constexpr (is_nvfp4 || is_mxfp8) {
      static_assert(ELEM_PER_THREAD == 8, "Expecting 8 elements per thread for quantized types");
      int64_t expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node,
                                                   (int64_t)permuted_row + 1) -
                       1;

      assert(!fc1_act_global_scale || is_nvfp4 && "Global scale is only supported for NVFP4");
      size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
      float global_scale_val = fc1_act_global_scale ? fc1_act_global_scale[act_scale_idx] : 1.0f;
      int64_t num_tokens_before_expert = expert_first_token_offset[expert];

      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        auto in_vec = source_row_ptr[elem_index];
        if constexpr (need_nvfp4_quant || need_mxfp8_quant) {
          auto res =
              quantizePackedFPXValue<InputActivationsType, ExpandedActivationsType, DataElem,
                                     VecSize, DISABLE_FP4_QUANT_FAST_MATH, NVFP4_4OVER6_CONFIG>(
                  in_vec, global_scale_val, num_tokens_before_expert, expert, permuted_row,
                  elem_index, padded_hidden_size, fc1_act_sf_flat,
                  is_nvfp4 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                           : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
          static_assert(sizeof(res) == sizeof(*dest_row_ptr),
                        "Quantized value must be the same size as the output");
          dest_row_ptr[elem_index] = res;
        } else {
          assert(act_scale_idx == 0 &&
                 "Cannot use per-expert act scale for pre-quantized activations");
          writeSF<VecSize, ELEM_PER_THREAD>(num_tokens_before_expert, expert, source_row,
                                            permuted_row, elem_index, padded_hidden_size,
                                            fc1_act_sf_flat, input_sf, swizzled_input_sf);
          dest_row_ptr[elem_index] = in_vec;
        }
      }

      // Pad zeros in the extra SFs along the K dimension, we do this to ensure there are no nan
      // values in the padded SF atom Use VecSize per thread since we are just writing out zeros so
      // every thread can process a whole vector
      size_t padding_start_offset = hidden_size / VecSize + start_offset;
      size_t padding_elems_in_col = padded_hidden_size / VecSize;
      for (int64_t elem_index = padding_start_offset; elem_index < padding_elems_in_col;
           elem_index += stride) {
        writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1, permuted_row,
                                  elem_index, padded_hidden_size, fc1_act_sf_flat,
                                  /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
      }
    } else if constexpr (PRE_QUANT_AWQ) {
      static_assert(!is_nvfp4 && !is_mxfp8, "NVFP4 and MXFP8 are not supported for AWQ");
      static_assert(!std::is_same_v<InputActivationsType, ExpandedActivationsType>,
                    "Input and output types must be different for AWQ");
      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        auto frag_elems = source_row_ptr[elem_index];

        CUTLASS_PRAGMA_UNROLL
        for (int e = 0; e < ELEM_PER_THREAD; e++) {
          frag_elems[e] = frag_elems[e] * prequant_scales[elem_index * ELEM_PER_THREAD + e];
        }

        dest_row_ptr[elem_index] = arrayConvert<DataElem, OutputElem>(frag_elems);
      }
    } else {
      for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
        dest_row_ptr[elem_index] = source_row_ptr[elem_index];
      }
    }

    if (permuted_scales && threadIdx.x == 0) {
      int64_t const source_k_idx = source_row * k + source_k_rank;
      permuted_scales[permuted_row] = unpermuted_scales ? unpermuted_scales[source_k_idx] : 1.0f;
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // N-dim SF padding (zeroing extra token rows beyond tokens_to_expert up to MinNDimAlignment)
  // is intentionally omitted. The CUTLASS grouped GEMM sets gemm_m = tokens_to_expert per expert
  // and never reads scale factors for rows beyond that. The N-dim padding rows don't correspond
  // to any valid MMA tiles, so their content doesn't affect correctness.
  // K-dim SF padding (above, inside the per-token loop) is still required because MMA tiles may
  // straddle the inter_size boundary within valid rows.
}

template <class InputActivationsType, class ExpandedActivationsType>
void expandInputRowsKernelLauncher(
    InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,
    float const* unpermuted_scales, float* permuted_scales,
    int const* permuted_row_to_unpermuted_row, int const* unpermuted_row_to_permuted_row,
    int const* token_selected_experts, int64_t const num_rows, int64_t const hidden_size,
    int const k, int const num_experts_per_node, int const start_expert_id,
    QuantParams const& quant_params, bool use_per_expert_act_scale,
    int64_t* expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, bool const swizzled_input_sf,
    void const* prequant_scales, bool enable_pdl, cudaStream_t stream) {
#ifdef ENABLE_FP4
  TLLM_CHECK_WITH_INFO(
      (std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1> && fc1_act_sf_flat) ||
          !use_per_expert_act_scale,
      "Per-expert act scale for FC1 is only supported for NVFP4 activations");
#endif

  static int64_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
  // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
  // N-dim SF padding has been removed (CUTLASS grouped GEMM never reads beyond
  // tokens_to_expert), so the grid is driven purely by the expanded token count.
  int64_t const blocks = std::min(smCount * 8, std::max(num_rows * k, int64_t{1}));
  int64_t const threads = EXPAND_THREADS_PER_BLOCK;

  if constexpr (std::is_same_v<InputActivationsType, ExpandedActivationsType> &&
                std::is_same_v<InputActivationsType, __nv_bfloat16>) {
    constexpr int64_t ScatterElemsPerThread = 128 / sizeof_bits<InputActivationsType>::value;
    bool const use_scatter_expand =
        prequant_scales == nullptr && !use_per_expert_act_scale &&
        unpermuted_row_to_permuted_row != nullptr && token_selected_experts != nullptr && k > 1 &&
        k <= EXPAND_SCATTER_MAX_K && num_rows >= 32768 && hidden_size % ScatterElemsPerThread == 0;
    if (use_scatter_expand) {
      cudaLaunchConfig_t config;
      config.gridDim = std::min(smCount * 8, std::max(num_rows, int64_t{1}));
      config.blockDim = threads;
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      cudaLaunchAttribute attrs[1];
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
      config.numAttrs = 1;
      config.attrs = attrs;
      cudaLaunchKernelEx(&config, &expandInputRowsScatterKernel<InputActivationsType>,
                         unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
                         unpermuted_row_to_permuted_row, token_selected_experts, num_rows,
                         hidden_size, k, num_experts_per_node, start_expert_id);
      return;
    }
  }

  auto func = [&]() {
#ifdef ENABLE_FP8
    // Always MXFP8
    if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> &&
                  !std::is_same_v<InputActivationsType, __nv_fp8_e4m3>) {
      TLLM_CHECK_WITH_INFO(quant_params.mxfp8_mxfp4.fc1.weight_block_scale ||
                               quant_params.mxfp8_mxfp8.fc1.weight_block_scale || prequant_scales,
                           "MXFP8 block scaling or prequant_scales parameters not provided");
      return prequant_scales
                 ? &expandInputRowsKernel<
                       InputActivationsType, ExpandedActivationsType,
                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, true>
                 : &expandInputRowsKernel<
                       InputActivationsType, ExpandedActivationsType,
                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX, false>;
    }
    // Could be either regular FP8 or MXFP8
    else if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp8_e4m3> &&
                       std::is_same_v<InputActivationsType, __nv_fp8_e4m3>) {
      TLLM_CHECK_WITH_INFO(!prequant_scales, "FP8 is not supported for AWQ");
      return (quant_params.mxfp8_mxfp4.fc1.weight_block_scale ||
              quant_params.mxfp8_mxfp8.fc1.weight_block_scale)
                 ? &expandInputRowsKernel<
                       InputActivationsType, ExpandedActivationsType,
                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX, false>
                 : &expandInputRowsKernel<
                       InputActivationsType, ExpandedActivationsType,
                       TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE, false>;
    } else
#endif
#ifdef ENABLE_FP4
        if constexpr (std::is_same_v<ExpandedActivationsType, __nv_fp4_e2m1>) {
      TLLM_CHECK_WITH_INFO(quant_params.fp4.fc1.weight_block_scale,
                           "NVFP4 block scaling is expected for FP4xFP4");
      TLLM_CHECK_WITH_INFO(!prequant_scales, "NVFP4 is not supported for AWQ");
      return dispatchNVFP44Over6Config(
          [&](auto disableFP4QuantFastMathTag, auto nvfp4_4over6_config_tag) {
            return &expandInputRowsKernel<
                InputActivationsType, ExpandedActivationsType,
                TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4, false,
                decltype(disableFP4QuantFastMathTag)::value, decltype(nvfp4_4over6_config_tag)>;
          });
    } else
#endif
    {
      TLLM_CHECK_WITH_INFO(!prequant_scales,
                           "w4afp8 Prequant scales provided for non-FP8 data type");
      return &expandInputRowsKernel<InputActivationsType, ExpandedActivationsType,
                                    TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE,
                                    false>;
    }
  }();

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, func, unpermuted_input, permuted_output, unpermuted_scales,
                     permuted_scales, permuted_row_to_unpermuted_row, num_rows, hidden_size, k,
                     quant_params.fp4.fc1.act_global_scale, use_per_expert_act_scale,
                     expert_first_token_offset, fc1_act_sf_flat, input_sf, swizzled_input_sf,
                     num_experts_per_node,
                     reinterpret_cast<InputActivationsType const*>(prequant_scales));
}

#define INSTANTIATE_EXPAND_INPUT_ROWS(InputActivationsType, ExpandedActivationsType)               \
  template void expandInputRowsKernelLauncher<InputActivationsType, ExpandedActivationsType>(      \
      InputActivationsType const* unpermuted_input, ExpandedActivationsType* permuted_output,      \
      float const* unpermuted_scales, float* permuted_scales,                                      \
      int const* permuted_row_to_unpermuted_row, int const* unpermuted_row_to_permuted_row,        \
      int const* token_selected_experts, int64_t const num_rows, int64_t const hidden_size,        \
      int const k, int const num_experts_per_node, int const start_expert_id,                      \
      QuantParams const& quant_params, bool use_per_expert_act_scale,                              \
      int64_t* expert_first_token_offset,                                                          \
      TmaWarpSpecializedGroupedGemmInput::ElementSF* fc1_act_sf_flat,                              \
      TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf, bool const swizzled_input_sf, \
      void const* prequant_scales, bool enable_pdl, cudaStream_t stream)

// Instantiate the data types that are used by the external pytorch op
// INSTANTIATE_EXPAND_INPUT_ROWS(float, float);
// INSTANTIATE_EXPAND_INPUT_ROWS(half, half);
// #ifdef ENABLE_BF16
// INSTANTIATE_EXPAND_INPUT_ROWS(__nv_bfloat16, __nv_bfloat16);
// #endif

enum class ScaleMode : int {
  NO_SCALE = 0,
  DEFAULT = 1,
};

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip
// connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingKernel(
    GemmOutputType const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias, float const* scales, int const* unpermuted_row_to_permuted_row,
    int const* token_selected_experts, int64_t const padded_cols, int64_t const unpadded_cols,
    int64_t const experts_per_token, int const num_experts_per_node, int const start_expert_id) {
  assert(padded_cols % 4 == 0);
  assert(unpadded_cols % 4 == 0);
  assert(unpadded_cols <= padded_cols);
  int64_t const original_row = blockIdx.x;
  int64_t const num_rows = gridDim.x;
  auto const offset = original_row * unpadded_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;
  (void)num_experts_per_node;
  (void)start_expert_id;

  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

  int64_t const start_offset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const num_elems_in_padded_col = padded_cols / FINALIZE_ELEM_PER_THREAD;
  int64_t const num_elems_in_orig_col = unpadded_cols / FINALIZE_ELEM_PER_THREAD;

  using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
  using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
  auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_orig_col; elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
      int64_t const k_offset = original_row * experts_per_token + k_idx;
      int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;
      if (expert_id < 0 || expert_id >= num_experts_per_node) {
        continue;
      }

      int64_t const expanded_original_row = original_row + k_idx * num_rows;
      int64_t const expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];

      int64_t const expanded_rows = num_rows * experts_per_token;
      if (expanded_permuted_row < 0 || expanded_permuted_row >= expanded_rows) {
        continue;
      }

      float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_padded_col;

      ComputeElem expert_result =
          arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
      if (bias) {
        auto const* bias_ptr = bias_v + expert_id * num_elems_in_padded_col;
        expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
      }

      thread_output = thread_output + row_scale * expert_result;
    }

    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip
// connection.
template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingNoFillingKernel(
    GemmOutputType const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias, float const* scales, int const* const unpermuted_row_to_permuted_row,
    int const* permuted_row_to_unpermuted_row, int const* token_selected_experts,
    int64_t const* expert_first_token_offset, int64_t const num_rows, int64_t const padded_cols,
    int64_t const unpadded_cols, int64_t const experts_per_token, int const num_experts_per_node,
    int const start_expert_id) {
  assert(padded_cols % 4 == 0);
  assert(unpadded_cols % 4 == 0);
  assert(unpadded_cols <= padded_cols);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];
  for (int64_t expanded_permuted_row = blockIdx.x; expanded_permuted_row < num_valid_tokens;
       expanded_permuted_row += gridDim.x) {
    int64_t unpermuted_row = permuted_row_to_unpermuted_row[expanded_permuted_row];

    // Duplicate and permute rows
    int64_t const source_k_rank = unpermuted_row / num_rows;
    int64_t const source_row = unpermuted_row % num_rows;

    // If the expert is the first selected (valid) one of the corresponding token on the current EP
    // rank, do reduction; otherwise, skip.
    bool is_first_selected_expert = true;
    for (int k_idx = 0; k_idx < source_k_rank; ++k_idx) {
      int const expert_id =
          token_selected_experts[source_row * experts_per_token + k_idx] - start_expert_id;
      if (expert_id >= 0 && expert_id < num_experts_per_node) {
        is_first_selected_expert = false;
        break;
      }
    }
    if (!is_first_selected_expert) {
      continue;
    }

    OutputType* reduced_row_ptr = reduced_unpermuted_output + source_row * unpadded_cols;

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD =
        128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_padded_col = padded_cols / FINALIZE_ELEM_PER_THREAD;
    int64_t const num_elems_in_orig_col = unpadded_cols / FINALIZE_ELEM_PER_THREAD;

    using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
    using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
    auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
    auto const* expanded_permuted_rows_v =
        reinterpret_cast<InputElem const*>(expanded_permuted_rows);
    auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

    for (int elem_index = start_offset; elem_index < num_elems_in_padded_col;
         elem_index += stride) {
      if (elem_index >= num_elems_in_orig_col) continue;  // Skip writing beyond original columns

      ComputeElem thread_output;
      thread_output.fill(0);
      for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
        int64_t const k_offset = source_row * experts_per_token + k_idx;
        int64_t const expert_id = token_selected_experts[k_offset] - start_expert_id;
        if (expert_id < 0 || expert_id >= num_experts_per_node) {
          continue;
        }

        int64_t const expanded_permuted_row_from_k_idx =
            unpermuted_row_to_permuted_row[source_row + k_idx * num_rows];

        float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];

        auto const* expanded_permuted_rows_row_ptr =
            expanded_permuted_rows_v + expanded_permuted_row_from_k_idx * num_elems_in_padded_col;

        ComputeElem expert_result =
            arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);

        if (bias) {
          auto const* bias_ptr = bias_v + expert_id * num_elems_in_padded_col;
          expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
        }

        thread_output = thread_output + row_scale * expert_result;
      }
      OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
      reduced_row_ptr_v[elem_index] = output_elem;
    }
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(
    GemmOutputType const* expanded_permuted_rows, OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias, float const* final_scales, int const* unpermuted_row_to_permuted_row,
    int const* permuted_row_to_unpermuted_row, int const* token_selected_experts,
    int64_t const* expert_first_token_offset, int64_t const num_rows, int64_t const padded_cols,
    int64_t const unpadded_cols, int64_t const experts_per_token,
    int64_t const num_experts_per_node, MOEParallelismConfig parallelism_config,
    bool const enable_alltoall, bool enable_pdl, cudaStream_t stream) {
  // Only add bias on rank 0 for tensor parallelism
  bool const is_rank_0 = parallelism_config.tp_rank == 0;
  ScaleBiasType const* bias_ptr = is_rank_0 ? bias : nullptr;
  int const start_expert_id = num_experts_per_node * parallelism_config.ep_rank;

  cudaLaunchConfig_t config;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  if (parallelism_config.ep_size > 1 && enable_alltoall) {
    // If all-to-all comm is enabled, finalizeMoeRouting doesn't need to fill the invalid output
    // tokens with zeros.
    static int const smCount = tensorrt_llm::common::getMultiProcessorCount();
    // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
    int64_t const blocks = smCount * 8;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
    config.blockDim = threads;
    auto func = final_scales
                    ? &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::DEFAULT>
                    : &finalizeMoeRoutingNoFillingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::NO_SCALE>;
    cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr,
                       final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
                       token_selected_experts, expert_first_token_offset, num_rows, padded_cols,
                       unpadded_cols, experts_per_token, num_experts_per_node, start_expert_id);
  } else {
    // If all-gather reduce-scatter is used, finalizeMoeRouting must fill invalid output tokens with
    // zeros.
    int64_t const blocks = num_rows;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;
    config.gridDim = blocks;
    config.blockDim = threads;
    auto func = final_scales ? &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::DEFAULT>
                             : &finalizeMoeRoutingKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                         ScaleMode::NO_SCALE>;
    cudaLaunchKernelEx(&config, func, expanded_permuted_rows, reduced_unpermuted_output, bias_ptr,
                       final_scales, unpermuted_row_to_permuted_row, token_selected_experts,
                       padded_cols, unpadded_cols, experts_per_token, num_experts_per_node,
                       start_expert_id);
  }
}

template <typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void finalizeMoeRoutingWithLoraKernel(
    GemmOutputType const* expanded_permuted_rows, ScaleBiasType const* expanded_lora_rows,
    int32_t const* expanded_lora_ranks, OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias, float const* scales, int const* unpermuted_row_to_permuted_row,
    int const* token_selected_experts, int64_t const padded_cols, int64_t const unpadded_cols,
    int64_t const experts_per_token, int const num_experts_per_node, int const start_expert_id) {
  assert(padded_cols % 4 == 0);
  assert(unpadded_cols % 4 == 0);
  assert(unpadded_cols <= padded_cols);
  int64_t const original_row = blockIdx.x;
  int64_t const num_rows = gridDim.x;
  auto const offset = original_row * unpadded_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;

  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

  int64_t const start_offset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const num_elems_in_padded_col = padded_cols / FINALIZE_ELEM_PER_THREAD;
  int64_t const num_elems_in_orig_col = unpadded_cols / FINALIZE_ELEM_PER_THREAD;

  using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
  using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
  auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto const* expanded_lora_rows_v = reinterpret_cast<BiasElem const*>(expanded_lora_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

  constexpr int kMaxCachedExpertsPerToken = 16;
  __shared__ int64_t cached_permuted_rows[kMaxCachedExpertsPerToken];
  __shared__ int cached_expert_ids[kMaxCachedExpertsPerToken];
  __shared__ int cached_has_lora[kMaxCachedExpertsPerToken];
  __shared__ float cached_scales[kMaxCachedExpertsPerToken];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  bool const use_cached_routes = experts_per_token <= kMaxCachedExpertsPerToken;
  if (use_cached_routes) {
    int64_t const expanded_rows = num_rows * experts_per_token;
    for (int k_idx = threadIdx.x; k_idx < experts_per_token; k_idx += blockDim.x) {
      int64_t const k_offset = original_row * experts_per_token + k_idx;
      int const expert_id = token_selected_experts[k_offset] - start_expert_id;
      int64_t expanded_permuted_row = -1;
      int has_lora = 0;
      if (expert_id >= 0 && expert_id < num_experts_per_node) {
        int64_t const expanded_original_row = original_row + k_idx * num_rows;
        expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];
        if (expanded_permuted_row < 0 || expanded_permuted_row >= expanded_rows) {
          expanded_permuted_row = -1;
        } else if (expanded_lora_rows && expanded_lora_ranks &&
                   expanded_lora_ranks[expanded_permuted_row] > 0) {
          has_lora = 1;
        }
      }
      cached_permuted_rows[k_idx] = expanded_permuted_row;
      cached_expert_ids[k_idx] = expert_id;
      cached_has_lora[k_idx] = has_lora;
      cached_scales[k_idx] = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
    }
    __syncthreads();
  }

#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_orig_col; elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
      int64_t expanded_permuted_row;
      int expert_id;
      int has_lora;
      float row_scale;
      if (use_cached_routes) {
        expanded_permuted_row = cached_permuted_rows[k_idx];
        expert_id = cached_expert_ids[k_idx];
        has_lora = cached_has_lora[k_idx];
        row_scale = cached_scales[k_idx];
        if (expanded_permuted_row < 0) {
          continue;
        }
      } else {
        int64_t const k_offset = original_row * experts_per_token + k_idx;
        expert_id = token_selected_experts[k_offset] - start_expert_id;
        if (expert_id < 0 || expert_id >= num_experts_per_node) {
          continue;
        }

        int64_t const expanded_original_row = original_row + k_idx * num_rows;
        expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];

        int64_t const expanded_rows = num_rows * experts_per_token;
        if (expanded_permuted_row < 0 || expanded_permuted_row >= expanded_rows) {
          continue;
        }

        row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
        has_lora = expanded_lora_rows && expanded_lora_ranks &&
                   expanded_lora_ranks[expanded_permuted_row] > 0;
      }

      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_padded_col;

      ComputeElem expert_result =
          arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
      if (has_lora) {
        auto const* expanded_lora_rows_row_ptr =
            expanded_lora_rows_v + expanded_permuted_row * num_elems_in_padded_col;
        expert_result = expert_result +
                        arrayConvert<BiasElem, ComputeElem>(expanded_lora_rows_row_ptr[elem_index]);
      }
      if (bias) {
        auto const* bias_ptr = bias_v + expert_id * num_elems_in_padded_col;
        expert_result = expert_result + arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
      }

      thread_output = thread_output + row_scale * expert_result;
    }

    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingWithLoraKernelLauncher(
    GemmOutputType const* expanded_permuted_rows, ScaleBiasType const* expanded_lora_rows,
    int32_t const* expanded_lora_ranks, OutputType* reduced_unpermuted_output,
    ScaleBiasType const* bias, float const* final_scales, int const* unpermuted_row_to_permuted_row,
    int const* token_selected_experts, int64_t const num_rows, int64_t const padded_cols,
    int64_t const unpadded_cols, int64_t const experts_per_token,
    int64_t const num_experts_per_node, MOEParallelismConfig parallelism_config, bool enable_pdl,
    cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(parallelism_config.ep_size == 1,
                       "LoRA finalize fusion does not support expert parallelism");
  ScaleBiasType const* bias_ptr = parallelism_config.tp_rank == 0 ? bias : nullptr;
  int const start_expert_id = num_experts_per_node * parallelism_config.ep_rank;

  cudaLaunchConfig_t config;
  config.gridDim = num_rows;
  config.blockDim = FINALIZE_THREADS_PER_BLOCK;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  auto func = final_scales ? &finalizeMoeRoutingWithLoraKernel<OutputType, GemmOutputType,
                                                               ScaleBiasType, ScaleMode::DEFAULT>
                           : &finalizeMoeRoutingWithLoraKernel<OutputType, GemmOutputType,
                                                               ScaleBiasType, ScaleMode::NO_SCALE>;
  cudaLaunchKernelEx(&config, func, expanded_permuted_rows, expanded_lora_rows, expanded_lora_ranks,
                     reduced_unpermuted_output, bias_ptr, final_scales,
                     unpermuted_row_to_permuted_row, token_selected_experts, padded_cols,
                     unpadded_cols, experts_per_token, num_experts_per_node, start_expert_id);
}

template <typename OutputType, class GemmOutputType, class ScaleBiasType, typename TokenIndexT,
          ScaleMode SCALE_MODE, bool kUsePdl>
__global__ void finalizeMoeRoutingWithTokenLoraKernel(
    GemmOutputType const* expanded_permuted_rows, ScaleBiasType const* expanded_lora_rows,
    OutputType* reduced_unpermuted_output, float const* scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts,
    TokenIndexT const* token_lora_indices, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int64_t const padded_cols,
    int64_t const unpadded_cols, int64_t const experts_per_token, int const num_experts_per_node,
    int const start_expert_id, int const num_lora_adapters, int64_t const col_begin,
    int64_t const col_end) {
  assert(padded_cols % 4 == 0);
  assert(unpadded_cols % 4 == 0);
  assert(unpadded_cols <= padded_cols);
  int64_t const original_row = blockIdx.x;
  int64_t const num_rows = gridDim.x;
  auto const offset = original_row * unpadded_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;

  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<GemmOutputType>::value);

  int64_t const start_offset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const num_elems_in_padded_col = padded_cols / FINALIZE_ELEM_PER_THREAD;
  int64_t const num_elems_in_orig_col = unpadded_cols / FINALIZE_ELEM_PER_THREAD;
  int64_t const slice_col_begin = col_begin > 0 ? col_begin : 0;
  int64_t const slice_col_end =
      col_end > 0 ? (col_end < unpadded_cols ? col_end : unpadded_cols) : unpadded_cols;
  int64_t const elem_begin = slice_col_begin / FINALIZE_ELEM_PER_THREAD;
  int64_t const elem_end = slice_col_end / FINALIZE_ELEM_PER_THREAD;
  int64_t const elem_limit = elem_end < num_elems_in_orig_col ? elem_end : num_elems_in_orig_col;

  using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
  using LoraElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
  auto const* expanded_lora_rows_v = reinterpret_cast<LoraElem const*>(expanded_lora_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (kUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  constexpr int kMaxCachedExpertsPerToken = 16;
  __shared__ int cached_token_has_lora;
  __shared__ int64_t cached_permuted_rows[kMaxCachedExpertsPerToken];
  __shared__ float cached_scales[kMaxCachedExpertsPerToken];
  int64_t const expanded_rows = num_rows * experts_per_token;
  bool const use_cached_routes = experts_per_token <= kMaxCachedExpertsPerToken;
  bool token_has_lora = false;
  if (use_cached_routes) {
    if (threadIdx.x == 0) {
      int has_lora = 0;
      int64_t const raw_lora_idx = static_cast<int64_t>(token_lora_indices[original_row]);
      if (raw_lora_idx >= 0 && raw_lora_idx < num_lora_adapters) {
        int const adapter = static_cast<int>(raw_lora_idx);
        int32_t const raw_rank = adapter_lora_ranks[adapter];
        auto const* lora_a = adapter_lora_weight_ptrs[adapter * 2];
        auto const* lora_b = adapter_lora_weight_ptrs[adapter * 2 + 1];
        has_lora = raw_rank > 0 && lora_a != nullptr && lora_b != nullptr;
      }
      cached_token_has_lora = has_lora;
    }
    for (int k_idx = threadIdx.x; k_idx < experts_per_token; k_idx += blockDim.x) {
      int64_t const k_offset = original_row * experts_per_token + k_idx;
      int const expert_id = token_selected_experts[k_offset] - start_expert_id;
      int64_t expanded_permuted_row = -1;
      if (expert_id >= 0 && expert_id < num_experts_per_node) {
        int64_t const expanded_original_row = original_row + k_idx * num_rows;
        expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];
        if (expanded_permuted_row < 0 || expanded_permuted_row >= expanded_rows) {
          expanded_permuted_row = -1;
        }
      }
      cached_permuted_rows[k_idx] = expanded_permuted_row;
      cached_scales[k_idx] = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
    }
    __syncthreads();
    token_has_lora = cached_token_has_lora != 0;
  } else {
    int64_t const raw_lora_idx = static_cast<int64_t>(token_lora_indices[original_row]);
    if (raw_lora_idx >= 0 && raw_lora_idx < num_lora_adapters) {
      int const adapter = static_cast<int>(raw_lora_idx);
      int32_t const raw_rank = adapter_lora_ranks[adapter];
      auto const* lora_a = adapter_lora_weight_ptrs[adapter * 2];
      auto const* lora_b = adapter_lora_weight_ptrs[adapter * 2 + 1];
      token_has_lora = raw_rank > 0 && lora_a != nullptr && lora_b != nullptr;
    }
  }

#pragma unroll
  for (int elem_index = elem_begin + start_offset; elem_index < elem_limit; elem_index += stride) {
    ComputeElem thread_output;
    thread_output.fill(0);
    for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
      int64_t expanded_permuted_row;
      float row_scale;
      if (use_cached_routes) {
        expanded_permuted_row = cached_permuted_rows[k_idx];
        if (expanded_permuted_row < 0) {
          continue;
        }
        row_scale = cached_scales[k_idx];
      } else {
        int64_t const k_offset = original_row * experts_per_token + k_idx;
        int const expert_id = token_selected_experts[k_offset] - start_expert_id;
        if (expert_id < 0 || expert_id >= num_experts_per_node) {
          continue;
        }

        int64_t const expanded_original_row = original_row + k_idx * num_rows;
        expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];
        if (expanded_permuted_row < 0 || expanded_permuted_row >= expanded_rows) {
          continue;
        }
        row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
      }

      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_padded_col;
      ComputeElem expert_result =
          arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
      if (token_has_lora) {
        auto const* expanded_lora_rows_row_ptr =
            expanded_lora_rows_v + expanded_permuted_row * num_elems_in_padded_col;
        expert_result = expert_result +
                        arrayConvert<LoraElem, ComputeElem>(expanded_lora_rows_row_ptr[elem_index]);
      }
      thread_output = thread_output + row_scale * expert_result;
    }

    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (kUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <typename TokenIndexT, int kThreads, int kElemPerThread, bool kTrustLoraMetadata,
          bool kUsePdl>
__global__ void finalizeMoeRoutingWithTokenLoraBf16Top8Kernel(
    __nv_bfloat16 const* expanded_permuted_rows, __nv_bfloat16 const* expanded_lora_rows,
    __nv_bfloat16* reduced_unpermuted_output, float const* scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts,
    TokenIndexT const* token_lora_indices, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int const num_lora_adapters) {
  static_assert(kElemPerThread == 8 || kElemPerThread == 16);
  constexpr int64_t kHiddenSize = 6144;
  constexpr int kExpertsPerToken = 8;
  constexpr int kNumExpertsPerNode = 256;
  constexpr int64_t kVecs = kHiddenSize / kElemPerThread;

  int64_t const original_row = blockIdx.x;
  int64_t const num_rows = gridDim.x;
  int64_t const expanded_rows = num_rows * kExpertsPerToken;

  using Elem = cutlass::Array<__nv_bfloat16, kElemPerThread>;
  using ComputeElem = cutlass::Array<float, kElemPerThread>;
  auto const* expanded_permuted_rows_v = reinterpret_cast<Elem const*>(expanded_permuted_rows);
  auto const* expanded_lora_rows_v = reinterpret_cast<Elem const*>(expanded_lora_rows);
  auto* output_v = reinterpret_cast<Elem*>(reduced_unpermuted_output + original_row * kHiddenSize);

  __shared__ int64_t cached_permuted_rows[kExpertsPerToken];
  __shared__ float cached_scales[kExpertsPerToken];
  __shared__ int cached_token_has_lora;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (kUsePdl) {
    cudaGridDependencySynchronize();
  }
#endif

  if (threadIdx.x == 0) {
    int has_lora = 0;
    int64_t const raw_lora_idx = static_cast<int64_t>(token_lora_indices[original_row]);
    if (raw_lora_idx >= 0 && raw_lora_idx < num_lora_adapters) {
      if constexpr (kTrustLoraMetadata) {
        has_lora = 1;
      } else {
        int const adapter = static_cast<int>(raw_lora_idx);
        int32_t const raw_rank = adapter_lora_ranks[adapter];
        auto const* lora_a = adapter_lora_weight_ptrs[adapter * 2];
        auto const* lora_b = adapter_lora_weight_ptrs[adapter * 2 + 1];
        has_lora = raw_rank > 0 && lora_a != nullptr && lora_b != nullptr;
      }
    }
    cached_token_has_lora = has_lora;
  }

  for (int k_idx = threadIdx.x; k_idx < kExpertsPerToken; k_idx += kThreads) {
    int64_t const k_offset = original_row * kExpertsPerToken + k_idx;
    int const expert_id = token_selected_experts[k_offset];
    int64_t expanded_permuted_row = -1;
    if (expert_id >= 0 && expert_id < kNumExpertsPerNode) {
      int64_t const expanded_original_row = original_row + k_idx * num_rows;
      expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];
      if (expanded_permuted_row < 0 || expanded_permuted_row >= expanded_rows) {
        expanded_permuted_row = -1;
      }
    }
    cached_permuted_rows[k_idx] = expanded_permuted_row;
    cached_scales[k_idx] = scales[k_offset];
  }
  __syncthreads();

  bool const token_has_lora = cached_token_has_lora != 0;
#pragma unroll
  for (int64_t elem_index = threadIdx.x; elem_index < kVecs; elem_index += kThreads) {
    ComputeElem thread_output;
    thread_output.fill(0);
#pragma unroll
    for (int k_idx = 0; k_idx < kExpertsPerToken; ++k_idx) {
      int64_t const expanded_permuted_row = cached_permuted_rows[k_idx];
      if (expanded_permuted_row < 0) {
        continue;
      }
      auto const* expanded_permuted_rows_row_ptr =
          expanded_permuted_rows_v + expanded_permuted_row * kVecs;
      ComputeElem expert_result =
          arrayConvert<Elem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
      if (token_has_lora) {
        auto const* expanded_lora_rows_row_ptr =
            expanded_lora_rows_v + expanded_permuted_row * kVecs;
        expert_result =
            expert_result + arrayConvert<Elem, ComputeElem>(expanded_lora_rows_row_ptr[elem_index]);
      }
      thread_output = thread_output + cached_scales[k_idx] * expert_result;
    }
    output_v[elem_index] = arrayConvert<ComputeElem, Elem>(thread_output);
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  if constexpr (kUsePdl) {
    cudaTriggerProgrammaticLaunchCompletion();
  }
#endif
}

template <typename TokenIndexT, int kThreads, int kElemPerThread, bool kTrustLoraMetadata>
void launchFinalizeMoeRoutingWithTokenLoraBf16Top8(
    __nv_bfloat16 const* expanded_permuted_rows, __nv_bfloat16 const* expanded_lora_rows,
    __nv_bfloat16* reduced_unpermuted_output, float const* final_scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts,
    TokenIndexT const* token_lora_indices, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int64_t const num_rows,
    int const num_lora_adapters, bool enable_pdl, cudaStream_t stream) {
  cudaLaunchConfig_t config;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  config.gridDim = num_rows;
  config.blockDim = kThreads;
  auto func =
      enable_pdl
          ? &finalizeMoeRoutingWithTokenLoraBf16Top8Kernel<TokenIndexT, kThreads, kElemPerThread,
                                                           kTrustLoraMetadata, true>
          : &finalizeMoeRoutingWithTokenLoraBf16Top8Kernel<TokenIndexT, kThreads, kElemPerThread,
                                                           kTrustLoraMetadata, false>;
  cudaLaunchKernelEx(&config, func, expanded_permuted_rows, expanded_lora_rows,
                     reduced_unpermuted_output, final_scales, unpermuted_row_to_permuted_row,
                     token_selected_experts, token_lora_indices, adapter_lora_ranks,
                     adapter_lora_weight_ptrs, num_lora_adapters);
}

template <class OutputType, class GemmOutputType, class ScaleBiasType, typename TokenIndexT>
bool maybeLaunchFinalizeMoeRoutingWithTokenLoraFastPath(
    GemmOutputType const* expanded_permuted_rows, ScaleBiasType const* expanded_lora_rows,
    OutputType* reduced_unpermuted_output, float const* final_scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts,
    TokenIndexT const* token_lora_indices, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int64_t const num_rows, int64_t const padded_cols,
    int64_t const unpadded_cols, int64_t const experts_per_token,
    int64_t const num_experts_per_node, MOEParallelismConfig parallelism_config,
    int const num_lora_adapters, bool enable_pdl, cudaStream_t stream) {
  if constexpr (std::is_same_v<OutputType, __nv_bfloat16> &&
                std::is_same_v<GemmOutputType, __nv_bfloat16> &&
                std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
    if (final_scales == nullptr || padded_cols != 6144 || unpadded_cols != 6144 ||
        experts_per_token != 8 || num_experts_per_node != 256 || parallelism_config.ep_size != 1 ||
        parallelism_config.ep_rank != 0 || num_lora_adapters <= 0) {
      return false;
    }
    if (enable_pdl) {
      return false;
    }
    auto const* base = reinterpret_cast<__nv_bfloat16 const*>(expanded_permuted_rows);
    auto const* lora = reinterpret_cast<__nv_bfloat16 const*>(expanded_lora_rows);
    auto* output = reinterpret_cast<__nv_bfloat16*>(reduced_unpermuted_output);
    launchFinalizeMoeRoutingWithTokenLoraBf16Top8<TokenIndexT, 256, 8, false>(
        base, lora, output, final_scales, unpermuted_row_to_permuted_row,
        token_selected_experts, token_lora_indices, adapter_lora_ranks, adapter_lora_weight_ptrs,
        num_rows, num_lora_adapters, enable_pdl, stream);
    return true;
  }
  return false;
}

template <class OutputType, class GemmOutputType, class ScaleBiasType, typename TokenIndexT>
void finalizeMoeRoutingWithTokenLoraKernelLauncher(
    GemmOutputType const* expanded_permuted_rows, ScaleBiasType const* expanded_lora_rows,
    OutputType* reduced_unpermuted_output, float const* final_scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts,
    TokenIndexT const* token_lora_indices, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int64_t const num_rows, int64_t const padded_cols,
    int64_t const unpadded_cols, int64_t const experts_per_token,
    int64_t const num_experts_per_node, MOEParallelismConfig parallelism_config,
    int const num_lora_adapters, bool enable_pdl, cudaStream_t stream, int64_t const col_begin = 0,
    int64_t const col_end = -1) {
  TLLM_CHECK_WITH_INFO(parallelism_config.ep_size == 1,
                       "LoRA token finalize fusion does not support expert parallelism");
  int const start_expert_id = num_experts_per_node * parallelism_config.ep_rank;
  if (start_expert_id == 0 &&
      maybeLaunchFinalizeMoeRoutingWithTokenLoraFastPath(
          expanded_permuted_rows, expanded_lora_rows, reduced_unpermuted_output, final_scales,
          unpermuted_row_to_permuted_row, token_selected_experts, token_lora_indices,
          adapter_lora_ranks, adapter_lora_weight_ptrs, num_rows, padded_cols, unpadded_cols,
          experts_per_token, num_experts_per_node, parallelism_config, num_lora_adapters,
          enable_pdl, stream)) {
    return;
  }

  cudaLaunchConfig_t config;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  config.gridDim = num_rows;
  config.blockDim = FINALIZE_THREADS_PER_BLOCK;
  auto func =
      final_scales
          ? (enable_pdl
                 ? &finalizeMoeRoutingWithTokenLoraKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                          TokenIndexT, ScaleMode::DEFAULT, true>
                 : &finalizeMoeRoutingWithTokenLoraKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                          TokenIndexT, ScaleMode::DEFAULT, false>)
          : (enable_pdl
                 ? &finalizeMoeRoutingWithTokenLoraKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                          TokenIndexT, ScaleMode::NO_SCALE, true>
                 : &finalizeMoeRoutingWithTokenLoraKernel<OutputType, GemmOutputType, ScaleBiasType,
                                                          TokenIndexT, ScaleMode::NO_SCALE, false>);
  cudaLaunchKernelEx(&config, func, expanded_permuted_rows, expanded_lora_rows,
                     reduced_unpermuted_output, final_scales, unpermuted_row_to_permuted_row,
                     token_selected_experts, token_lora_indices, adapter_lora_ranks,
                     adapter_lora_weight_ptrs, padded_cols, unpadded_cols, experts_per_token,
                     num_experts_per_node, start_expert_id, num_lora_adapters, col_begin, col_end);
}

template <typename OutputType, class ScaleBiasType, ScaleMode SCALE_MODE>
__global__ void addFinalizedLoraToOutputKernel(
    ScaleBiasType const* expanded_lora_rows, int32_t const* expanded_lora_ranks,
    OutputType* reduced_unpermuted_output, float const* scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts,
    int64_t const padded_cols, int64_t const unpadded_cols, int64_t const experts_per_token,
    int const num_experts_per_node, int const start_expert_id) {
  assert(padded_cols % 4 == 0);
  assert(unpadded_cols % 4 == 0);
  assert(unpadded_cols <= padded_cols);
  int64_t const original_row = blockIdx.x;
  int64_t const num_rows = gridDim.x;
  auto const offset = original_row * unpadded_cols;
  OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;

  constexpr int64_t FINALIZE_ELEM_PER_THREAD =
      128 / std::min(sizeof_bits<OutputType>::value, sizeof_bits<ScaleBiasType>::value);

  int64_t const start_offset = threadIdx.x;
  int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
  int64_t const num_elems_in_padded_col = padded_cols / FINALIZE_ELEM_PER_THREAD;
  int64_t const num_elems_in_orig_col = unpadded_cols / FINALIZE_ELEM_PER_THREAD;

  using LoraElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
  auto const* expanded_lora_rows_v = reinterpret_cast<LoraElem const*>(expanded_lora_rows);
  auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

  constexpr int kMaxCachedExpertsPerToken = 16;
  __shared__ int64_t cached_permuted_rows[kMaxCachedExpertsPerToken];
  __shared__ int cached_has_lora[kMaxCachedExpertsPerToken];
  __shared__ float cached_scales[kMaxCachedExpertsPerToken];
  __shared__ int cached_any_lora;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  bool const use_cached_routes = experts_per_token <= kMaxCachedExpertsPerToken;
  if (use_cached_routes) {
    if (threadIdx.x == 0) {
      cached_any_lora = 0;
    }
    __syncthreads();

    int64_t const expanded_rows = num_rows * experts_per_token;
    for (int k_idx = threadIdx.x; k_idx < experts_per_token; k_idx += blockDim.x) {
      int64_t const k_offset = original_row * experts_per_token + k_idx;
      int const expert_id = token_selected_experts[k_offset] - start_expert_id;
      int64_t expanded_permuted_row = -1;
      int has_lora = 0;
      if (expert_id >= 0 && expert_id < num_experts_per_node) {
        int64_t const expanded_original_row = original_row + k_idx * num_rows;
        expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];
        if (expanded_permuted_row < 0 || expanded_permuted_row >= expanded_rows) {
          expanded_permuted_row = -1;
        } else if (expanded_lora_ranks && expanded_lora_ranks[expanded_permuted_row] > 0) {
          has_lora = 1;
          atomicOr(&cached_any_lora, 1);
        }
      }
      cached_permuted_rows[k_idx] = expanded_permuted_row;
      cached_has_lora[k_idx] = has_lora;
      cached_scales[k_idx] = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
    }
    __syncthreads();
    if (!cached_any_lora) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
      cudaTriggerProgrammaticLaunchCompletion();
#endif
      return;
    }
  }

#pragma unroll
  for (int elem_index = start_offset; elem_index < num_elems_in_orig_col; elem_index += stride) {
    ComputeElem thread_output =
        arrayConvert<OutputElem, ComputeElem>(reduced_row_ptr_v[elem_index]);
    for (int k_idx = 0; k_idx < experts_per_token; ++k_idx) {
      int64_t expanded_permuted_row;
      int has_lora;
      float row_scale;
      if (use_cached_routes) {
        expanded_permuted_row = cached_permuted_rows[k_idx];
        has_lora = cached_has_lora[k_idx];
        row_scale = cached_scales[k_idx];
        if (expanded_permuted_row < 0 || !has_lora) {
          continue;
        }
      } else {
        int64_t const k_offset = original_row * experts_per_token + k_idx;
        int const expert_id = token_selected_experts[k_offset] - start_expert_id;
        if (expert_id < 0 || expert_id >= num_experts_per_node) {
          continue;
        }

        int64_t const expanded_original_row = original_row + k_idx * num_rows;
        expanded_permuted_row = unpermuted_row_to_permuted_row[expanded_original_row];

        int64_t const expanded_rows = num_rows * experts_per_token;
        if (expanded_permuted_row < 0 || expanded_permuted_row >= expanded_rows ||
            expanded_lora_ranks[expanded_permuted_row] <= 0) {
          continue;
        }

        row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
      }

      auto const* expanded_lora_rows_row_ptr =
          expanded_lora_rows_v + expanded_permuted_row * num_elems_in_padded_col;
      thread_output = thread_output + row_scale * arrayConvert<LoraElem, ComputeElem>(
                                                      expanded_lora_rows_row_ptr[elem_index]);
    }

    OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
    reduced_row_ptr_v[elem_index] = output_elem;
  }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <class OutputType, class ScaleBiasType>
void addFinalizedLoraToOutputKernelLauncher(
    ScaleBiasType const* expanded_lora_rows, int32_t const* expanded_lora_ranks,
    OutputType* reduced_unpermuted_output, float const* final_scales,
    int const* unpermuted_row_to_permuted_row, int const* token_selected_experts,
    int64_t const num_rows, int64_t const padded_cols, int64_t const unpadded_cols,
    int64_t const experts_per_token, int64_t const num_experts_per_node,
    MOEParallelismConfig parallelism_config, bool enable_pdl, cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(parallelism_config.ep_size == 1,
                       "LoRA finalize fusion does not support expert parallelism");
  int const start_expert_id = num_experts_per_node * parallelism_config.ep_rank;

  cudaLaunchConfig_t config;
  config.gridDim = num_rows;
  config.blockDim = FINALIZE_THREADS_PER_BLOCK;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;

  auto func = final_scales
                  ? &addFinalizedLoraToOutputKernel<OutputType, ScaleBiasType, ScaleMode::DEFAULT>
                  : &addFinalizedLoraToOutputKernel<OutputType, ScaleBiasType, ScaleMode::NO_SCALE>;
  cudaLaunchKernelEx(&config, func, expanded_lora_rows, expanded_lora_ranks,
                     reduced_unpermuted_output, final_scales, unpermuted_row_to_permuted_row,
                     token_selected_experts, padded_cols, unpadded_cols, experts_per_token,
                     num_experts_per_node, start_expert_id);
}

#define INSTANTIATE_FINALIZE_MOE_ROUTING(OutputT, GemmOutputT, ScaleBiasT)                  \
  template void finalizeMoeRoutingKernelLauncher<OutputT, GemmOutputT, ScaleBiasT>(         \
      GemmOutputT const* expanded_permuted_rows, OutputT* reduced_unpermuted_output,        \
      ScaleBiasT const* bias, float const* final_scales,                                    \
      int const* unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row, \
      int const* expert_for_source_row, int64_t const* expert_first_token_offset,           \
      int64_t const num_rows, int64_t const padded_cols, int64_t const actual_cols,         \
      int64_t const experts_per_token, int64_t const num_experts_per_node,                  \
      MOEParallelismConfig parallelism_config, bool const enable_alltoall, bool enable_pdl, \
      cudaStream_t stream);

// // Instantiate the data types that are used by the external pytorch op
// INSTANTIATE_FINALIZE_MOE_ROUTING(half, half, half);
// INSTANTIATE_FINALIZE_MOE_ROUTING(float, float, float);
// #ifdef ENABLE_BF16
// INSTANTIATE_FINALIZE_MOE_ROUTING(__nv_bfloat16, __nv_bfloat16, __nv_bfloat16);
// #endif

// ============================== Activation Adaptors =================================
template <template <class> class ActFn>
struct IdentityAdaptor {
  constexpr static bool IS_GLU = false;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();

  template <class T>
  __device__ T operator()(T const& x) const {
    ActFn<T> fn{};
    return fn(x);
  }
};

template <template <class> class ActFn>
struct GLUAdaptor {
  constexpr static bool IS_GLU = true;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();

  template <class T>
  __device__ T operator()(T const& gate, T const& linear) const {
    ActFn<T> fn{};
    return fn(gate) * linear;
  }
};

struct SwigluBiasAdaptor {
  constexpr static bool IS_GLU = true;
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();

  template <class T>
  __device__ T operator()(T const& gate, T const& linear) const {
    cutlass::epilogue::thread::Sigmoid<T> fn{};
    T linear_clamped = cutlass::maximum<T>{}(cutlass::minimum<T>{}(linear, limit), -limit);
    T gate_clamped = cutlass::minimum<T>{}(gate, limit);
    return gate_clamped * fn(gate_clamped * alpha) * (linear_clamped + beta);
  }
};

// ============================== Gated Activation =================================
constexpr static int ACTIVATION_THREADS_PER_BLOCK = 256;

template <class ActivationOutputType, class GemmOutputType, class ActFn>
__global__ void doGatedActivationKernel(ActivationOutputType* output,
                                        GemmOutputType const* gemm_result,
                                        int64_t const* expert_first_token_offset,
                                        int64_t inter_size, int64_t num_experts_per_node,
                                        ActivationParams activation_type) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  if (token >= expert_first_token_offset[num_experts_per_node]) {
    return;
  }

  output = output + token * inter_size;
  gemm_result = gemm_result + token * inter_size * 2;

  constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 128 / sizeof_bits<ActivationOutputType>::value;

  using OutputElem = cutlass::Array<ActivationOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
  auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
  auto output_vec = reinterpret_cast<OutputElem*>(output);
  int64_t const start_offset = tid;
  int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
  assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
  int64_t const inter_size_vec = inter_size / ACTIVATION_ELEM_PER_THREAD;

  float gate_alpha = 1.0f;
  float gate_bias = 0.0f;
  float gate_limit = std::numeric_limits<float>::infinity();
  if (activation_type.swiglu_alpha || activation_type.swiglu_beta || activation_type.swiglu_limit) {
    int expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node,
                                             (int64_t)token + 1) -
                 1;
    gate_alpha = activation_type.swiglu_alpha ? activation_type.swiglu_alpha[expert] : 1.0f;
    gate_bias = activation_type.swiglu_beta ? activation_type.swiglu_beta[expert] : 0.0f;
    gate_limit = activation_type.swiglu_limit ? activation_type.swiglu_limit[expert]
                                              : std::numeric_limits<float>::infinity();
  }

  ActFn fn{};
  fn.alpha = gate_alpha;
  fn.beta = gate_bias;
  fn.limit = gate_limit;
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto linear_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
    // BF16 isn't supported, use FP32 for activation function
    auto gate_value =
        arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + inter_size_vec]);
    auto gate_act = fn(gate_value, linear_value);
    output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(gate_act);
  }
}

template <typename ActivationOutputType, typename GemmOutputType>
void doGatedActivation(ActivationOutputType* output, GemmOutputType const* gemm_result,
                       int64_t const* expert_first_token_offset, int64_t inter_size,
                       int64_t num_tokens, int64_t num_experts_per_node,
                       ActivationParams activation_type, cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

  auto* fn = (activation_type == ActivationType::Swiglu)
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType,
                                            GLUAdaptor<cutlass::epilogue::thread::SiLu>>
             : activation_type == ActivationType::Geglu
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType,
                                            GLUAdaptor<cutlass::epilogue::thread::GELU>>
             : activation_type == ActivationType::SwigluBias
                 ? &doGatedActivationKernel<ActivationOutputType, GemmOutputType, SwigluBiasAdaptor>
                 : nullptr;
  TLLM_CHECK_WITH_INFO(fn != nullptr, "Invalid activation type");
  fn<<<blocks, threads, 0, stream>>>(output, gemm_result, expert_first_token_offset, inter_size,
                                     num_experts_per_node, activation_type);
}

// ============================== Activation =================================

template <class T, class GemmOutputType, class ScaleBiasType, class ActFn,
          TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType BlockScalingType,
          bool DISABLE_FP4_QUANT_FAST_MATH = false, typename NVFP4_4OVER6_CONFIG = std::false_type>
__global__ __launch_bounds__(ACTIVATION_THREADS_PER_BLOCK) void doActivationKernel(
    T* output, GemmOutputType const* gemm_result, float const* fp8_quant,
    ScaleBiasType const* bias_ptr, bool bias_is_broadcast, int64_t const* expert_first_token_offset,
    int num_experts_per_node, int64_t inter_size, float const* fc2_act_global_scale,
    bool use_per_expert_act_scale, TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat,
    int32_t const* bias_row_ranks, ActivationParams activation_params) {
#ifdef ENABLE_FP4
  constexpr bool IsNVFP4 =
      std::is_same_v<T, __nv_fp4_e2m1> &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4;
  constexpr bool IsMXFP8 =
      std::is_same_v<T, __nv_fp8_e4m3> &&
      BlockScalingType == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX;
#else
  constexpr bool IsNVFP4 = cute::dependent_false<T>;
  constexpr bool IsMXFP8 = cute::dependent_false<T>;
#endif

  int64_t const tid = threadIdx.x;
  constexpr bool IsGated = ActFn::IS_GLU;
  size_t gated_size_mul = IsGated ? 2 : 1;
  size_t gated_off = IsGated ? inter_size : 0;

  constexpr int64_t VecSize = IsNVFP4
                                  ? TmaWarpSpecializedGroupedGemmInput::NVFP4BlockScaleVectorSize
                                  : TmaWarpSpecializedGroupedGemmInput::MXFPXBlockScaleVectorSize;
  // Load 128-bits per thread, according to the smallest data type we read/write
  constexpr int64_t ACTIVATION_ELEM_PER_THREAD =
      (IsNVFP4 || IsMXFP8)
          ? CVT_ELTS_PER_THREAD
          : (128 / std::min(sizeof_bits<T>::value, sizeof_bits<GemmOutputType>::value));
  static_assert(!IsNVFP44Over6Config<NVFP4_4OVER6_CONFIG>::value || IsNVFP4,
                "NVFP4 4over6 requires NVFP4 block scaling");

  // This should be VecSize * 4 elements
  // We assume at least VecSize alignment or the quantization will fail
  int64_t const min_k_dim_alignment =
      IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentNVFP4
              : TmaWarpSpecializedGroupedGemmInput::MinKDimAlignmentMXFPX;
  int64_t const padded_inter_size = ceilDiv(inter_size, min_k_dim_alignment) * min_k_dim_alignment;

  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif
  for (int64_t token = blockIdx.x; token < num_valid_tokens; token += gridDim.x) {
    size_t gemm_result_offset = token * inter_size * gated_size_mul;
    size_t output_offset = token * inter_size;
    bool const row_has_bias = bias_ptr && (bias_row_ranks == nullptr || bias_row_ranks[token] > 0);

    int64_t expert = 0;
    float gate_alpha = 1.0f;
    float gate_beta = 0.0f;
    float gate_limit = std::numeric_limits<float>::infinity();
    bool const needs_expert = (row_has_bias && bias_is_broadcast) || IsNVFP4 || IsMXFP8 ||
                              use_per_expert_act_scale || activation_params.swiglu_alpha ||
                              activation_params.swiglu_beta || activation_params.swiglu_limit;
    if (needs_expert) {
      // TODO this is almost certainly faster as a linear scan
      expert =
          findTotalEltsLessThanTarget(expert_first_token_offset, num_experts_per_node, token + 1) -
          1;

      gate_alpha = activation_params.swiglu_alpha ? activation_params.swiglu_alpha[expert] : 1.0f;
      gate_beta = activation_params.swiglu_beta ? activation_params.swiglu_beta[expert] : 0.0f;
      gate_limit = activation_params.swiglu_limit ? activation_params.swiglu_limit[expert]
                                                  : std::numeric_limits<float>::infinity();
    }

    size_t act_scale_idx = use_per_expert_act_scale ? expert : 0;
    float const quant_scale = fp8_quant ? fp8_quant[act_scale_idx] : 1.f;

    // Some globals for FP4
    float global_scale_val = fc2_act_global_scale ? fc2_act_global_scale[act_scale_idx] : 1.0f;
    int64_t num_tokens_before_expert = (IsNVFP4 || IsMXFP8) ? expert_first_token_offset[expert] : 0;

    size_t bias_offset = 0;
    if (row_has_bias) {
      bias_offset = (bias_is_broadcast ? expert * inter_size * gated_size_mul : gemm_result_offset);
    }

    using BiasElem = cutlass::Array<ScaleBiasType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using OutputElem = std::conditional_t<
        IsNVFP4, uint32_t,
        std::conditional_t<IsMXFP8, uint64_t, cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>>>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    // Aliases gemm_result for non-gated, non-fp8 cases
    auto gemm_result_vec =
        reinterpret_cast<GemmResultElem const*>(gemm_result + gemm_result_offset);
    auto output_vec = reinterpret_cast<OutputElem*>(safe_inc_ptr(output, output_offset));
    auto bias_ptr_vec =
        row_has_bias ? reinterpret_cast<BiasElem const*>(bias_ptr + bias_offset) : nullptr;
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

    ActFn fn{};
    fn.alpha = gate_alpha;
    fn.beta = gate_beta;
    fn.limit = gate_limit;
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      auto fc1_value =
          arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + gated_off_vec]);
      if (bias_ptr_vec) {
        fc1_value = fc1_value +
                    arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index + gated_off_vec]);
      }

      auto gate_act = [&]() {
        if constexpr (IsGated) {
          auto linear_value =
              arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
          if (bias_ptr_vec) {
            linear_value =
                linear_value + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index]);
          }
          return fn(fc1_value, linear_value);
        } else {
          return fn(fc1_value);
        }
      }();

      auto post_act_val = gate_act * quant_scale;

      if constexpr (IsNVFP4 || IsMXFP8) {
        // We use GemmOutputType as the intermediate compute type as that should always be
        // unquantized
        auto res = quantizePackedFPXValue<GemmOutputType, T, ComputeElem, VecSize,
                                          DISABLE_FP4_QUANT_FAST_MATH, NVFP4_4OVER6_CONFIG>(
            post_act_val, global_scale_val, num_tokens_before_expert, expert, token, elem_index,
            inter_size, fc2_act_sf_flat,
            IsNVFP4 ? TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4
                    : TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
        static_assert(sizeof(res) == sizeof(*output_vec),
                      "Quantized value must be the same size as the output");
        output_vec[elem_index] = res;
      } else {
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(post_act_val);
      }
    }

    // Pad zeros in the extra SFs along the K dimension, we do this to ensure there are no nan
    // values in the padded SF atom
    if constexpr (IsNVFP4 || IsMXFP8) {
      // Use VecSize per thread since we are just writing out zeros so every thread can process a
      // whole vector
      size_t padding_start_offset = inter_size / VecSize + start_offset;
      size_t padding_elems_in_col = padded_inter_size / VecSize;
      for (int64_t elem_index = padding_start_offset; elem_index < padding_elems_in_col;
           elem_index += stride) {
        writeSF<VecSize, VecSize>(num_tokens_before_expert, expert, /*source_row*/ -1, token,
                                  elem_index, padded_inter_size, fc2_act_sf_flat,
                                  /* input_sf */ nullptr);  // Pass nulltpr input_sf so we write 0
      }
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif

  // N-dim SF padding (zeroing extra token rows beyond tokens_to_expert up to MinNDimAlignment)
  // is intentionally omitted. The CUTLASS grouped GEMM sets gemm_m = tokens_to_expert per expert
  // and never reads scale factors for rows beyond that. The N-dim padding rows don't correspond
  // to any valid MMA tiles, so their content doesn't affect correctness.
  // K-dim SF padding (above, inside the per-token loop) is still required because MMA tiles may
  // straddle the inter_size boundary within valid rows.
}

template <int kThreadsPerBlock>
__global__ __launch_bounds__(kThreadsPerBlock) void doActivationSwigluBf16Kernel(
    __nv_bfloat16* output, __nv_bfloat16 const* gemm_result, __nv_bfloat16 const* bias_ptr,
    int64_t const* expert_first_token_offset, int num_experts_per_node, int64_t inter_size,
    int32_t const* bias_row_ranks) {
  int64_t const tid = threadIdx.x;
  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];
  int64_t const inter_pairs = inter_size >> 1;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaGridDependencySynchronize();
#endif

  for (int64_t token = blockIdx.x; token < num_valid_tokens; token += gridDim.x) {
    bool const row_has_bias =
        bias_ptr != nullptr && (bias_row_ranks == nullptr || bias_row_ranks[token] > 0);
    int64_t const row_base = token * inter_size * 2;
    auto const* linear_pairs = reinterpret_cast<__nv_bfloat162 const*>(gemm_result + row_base);
    auto const* gate_pairs =
        reinterpret_cast<__nv_bfloat162 const*>(gemm_result + row_base + inter_size);
    auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output + token * inter_size);
    auto const* bias_linear_pairs =
        row_has_bias ? reinterpret_cast<__nv_bfloat162 const*>(bias_ptr + row_base) : nullptr;
    auto const* bias_gate_pairs =
        row_has_bias ? reinterpret_cast<__nv_bfloat162 const*>(bias_ptr + row_base + inter_size)
                     : nullptr;

    for (int64_t pair_idx = tid; pair_idx < inter_pairs; pair_idx += blockDim.x) {
      float2 linear_value = __bfloat1622float2(linear_pairs[pair_idx]);
      float2 gate_value = __bfloat1622float2(gate_pairs[pair_idx]);
      if (row_has_bias) {
        float2 const linear_bias = __bfloat1622float2(bias_linear_pairs[pair_idx]);
        float2 const gate_bias = __bfloat1622float2(bias_gate_pairs[pair_idx]);
        linear_value.x += linear_bias.x;
        linear_value.y += linear_bias.y;
        gate_value.x += gate_bias.x;
        gate_value.y += gate_bias.y;
      }

      float const act0 = gate_value.x / (1.0f + __expf(-gate_value.x));
      float const act1 = gate_value.y / (1.0f + __expf(-gate_value.y));
      output_pairs[pair_idx] = __floats2bfloat162_rn(act0 * linear_value.x, act1 * linear_value.y);
    }
  }

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  cudaTriggerProgrammaticLaunchCompletion();
#endif
}

template <class T, class GemmOutputType, class ScaleBiasType>
void doActivation(T* output, GemmOutputType const* gemm_result, float const* fp8_quant,
                  ScaleBiasType const* bias, bool bias_is_broadcast,
                  int64_t const* expert_first_token_offset, int num_experts_per_node,
                  int64_t inter_size, int64_t expanded_num_tokens, ActivationParams activation_type,
                  QuantParams const& quant_params, bool use_per_expert_act_scale,
                  TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_act_sf_flat, bool enable_pdl,
                  cudaStream_t stream, int32_t const* bias_row_ranks = nullptr) {
  static int64_t const smCount = tensorrt_llm::common::getMultiProcessorCount();
  // Note: Launching 8 blocks per SM can fully leverage the memory bandwidth (tested on B200).
  // N-dim SF padding has been removed (CUTLASS grouped GEMM never reads beyond
  // tokens_to_expert), so the grid is driven purely by the expanded token count.
  int64_t blocks = std::min(smCount * 8, std::max(expanded_num_tokens, int64_t{1}));
  int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

  if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<GemmOutputType, __nv_bfloat16> &&
                std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
    if (activation_type.activation_type == ActivationType::Swiglu && fp8_quant == nullptr &&
        !bias_is_broadcast && !use_per_expert_act_scale && fc2_act_sf_flat == nullptr &&
        activation_type.swiglu_alpha == nullptr && activation_type.swiglu_beta == nullptr &&
        activation_type.swiglu_limit == nullptr && (inter_size % 2 == 0)) {
      cudaLaunchConfig_t config;
      config.gridDim = blocks;
      config.blockDim = threads;
      config.dynamicSmemBytes = 0;
      config.stream = stream;
      cudaLaunchAttribute attrs[1];
      attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
      attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
      config.numAttrs = 1;
      config.attrs = attrs;
      cudaLaunchKernelEx(&config, doActivationSwigluBf16Kernel<ACTIVATION_THREADS_PER_BLOCK>,
                         reinterpret_cast<__nv_bfloat16*>(output),
                         reinterpret_cast<__nv_bfloat16 const*>(gemm_result),
                         reinterpret_cast<__nv_bfloat16 const*>(bias), expert_first_token_offset,
                         num_experts_per_node, inter_size, bias_row_ranks);
      return;
    }
  }

  auto fn = [&]() {
    auto fn = [&](auto block_scaling_type, auto disableFP4QuantFastMathTag,
                  auto nvfp4_4over6_config_tag) {
      auto fn_list = std::array{
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, IdentityAdaptor<cutlass::epilogue::thread::GELU>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Gelu
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, IdentityAdaptor<cutlass::epilogue::thread::ReLu>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Relu
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, IdentityAdaptor<cutlass::epilogue::thread::SiLu>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Silu
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, GLUAdaptor<cutlass::epilogue::thread::SiLu>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Swiglu
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, GLUAdaptor<cutlass::epilogue::thread::GELU>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Geglu
          &doActivationKernel<T, GemmOutputType, ScaleBiasType, SwigluBiasAdaptor,
                              decltype(block_scaling_type)::value,
                              decltype(disableFP4QuantFastMathTag)::value,
                              decltype(nvfp4_4over6_config_tag)>,  // SwigluBias
          &doActivationKernel<
              T, GemmOutputType, ScaleBiasType, IdentityAdaptor<cutlass::epilogue::thread::Relu2>,
              decltype(block_scaling_type)::value, decltype(disableFP4QuantFastMathTag)::value,
              decltype(nvfp4_4over6_config_tag)>,  // Relu2
          &doActivationKernel<T, GemmOutputType, ScaleBiasType,
                              IdentityAdaptor<cutlass::epilogue::thread::Identity>,
                              decltype(block_scaling_type)::value,
                              decltype(disableFP4QuantFastMathTag)::value,
                              decltype(nvfp4_4over6_config_tag)>  // Identity
      };
      return fn_list[static_cast<int>(activation_type.activation_type)];
    };
#ifdef ENABLE_FP4
    auto NVFP4 = tensorrt_llm::common::ConstExprWrapper<
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NVFP4>{};
    auto MXFPX = tensorrt_llm::common::ConstExprWrapper<
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX>{};
#endif
    auto NONE = tensorrt_llm::common::ConstExprWrapper<
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType,
        TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::NONE>{};
#ifdef ENABLE_FP4
    if constexpr (std::is_same_v<T, __nv_fp4_e2m1>) {
      TLLM_CHECK_WITH_INFO(quant_params.fp4.fc2.weight_block_scale,
                           "NVFP4 block scaling is expected for FP4xFP4");
      return dispatchNVFP44Over6Config(
          [&](auto disableFP4QuantFastMathTag, auto nvfp4_4over6_config_tag) {
            return fn(NVFP4, disableFP4QuantFastMathTag, nvfp4_4over6_config_tag);
          });
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
      return (quant_params.mxfp8_mxfp4.fc2.weight_block_scale ||
              quant_params.mxfp8_mxfp8.fc2.weight_block_scale)
                 ? fn(MXFPX, std::false_type{}, std::false_type{})
                 : fn(NONE, std::false_type{}, std::false_type{});
    } else
#endif
    {
      return fn(NONE, std::false_type{}, std::false_type{});
    }
  }();

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, fn, output, gemm_result, fp8_quant, bias, bias_is_broadcast,
                     expert_first_token_offset, num_experts_per_node, inter_size,
                     quant_params.fp4.fc2.act_global_scale, use_per_expert_act_scale,
                     fc2_act_sf_flat, bias_row_ranks, activation_type);
}

// ============================== Lora Add Bias =================================
constexpr static int LORA_KERNELS_THREADS_PER_BLOCK = 256;

template <class ScaleBiasType, class LoraType, bool IsGated>
__global__ void loraAddBiasKernel(ScaleBiasType* output, LoraType const* lora_result,
                                  ScaleBiasType const* bias, int64_t const* num_valid_tokens_ptr,
                                  int* permuted_token_selected_experts, int64_t inter_size) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  int64_t const num_tokens = gridDim.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  LoraType const* lora_result_1 = lora_result + token * inter_size;
  int expert_id = permuted_token_selected_experts[token];
  if constexpr (IsGated) {
    output = output + token * inter_size * 2;
    bias = bias + expert_id * inter_size * 2;
  } else {
    output = output + token * inter_size;
    bias = bias + expert_id * inter_size;
  }

  constexpr int64_t LORA_ADD_BIAS_ELEM_PER_THREAD = 128 / sizeof_bits<LoraType>::value;

  using DataElem = cutlass::Array<LoraType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
  using BiasElem = cutlass::Array<ScaleBiasType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
  auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
  auto bias_vec = reinterpret_cast<BiasElem const*>(bias);
  auto output_vec = reinterpret_cast<BiasElem*>(output);

  int64_t const start_offset = tid;
  int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % LORA_ADD_BIAS_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_1_vec[elem_index];
    auto bias_value = bias_vec[elem_index];
    output_vec[elem_index] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
  }

  if constexpr (IsGated) {
    auto lora_result_2_vec =
        reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
    int64_t const inter_size_vec = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
      auto lora_value = lora_result_2_vec[elem_index];
      auto bias_value = bias_vec[elem_index + inter_size_vec];
      output_vec[elem_index + inter_size_vec] =
          bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
    }
  }
}

template <class ScaleBiasType, class LoraType>
void loraAddBias(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
                 int64_t const* num_valid_tokens_ptr, int64_t inter_size,
                 int* permuted_token_selected_experts, int64_t num_tokens, bool is_gated_activation,
                 cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

  auto selected_fn = is_gated_activation ? loraAddBiasKernel<ScaleBiasType, LoraType, true>
                                         : loraAddBiasKernel<ScaleBiasType, LoraType, false>;
  selected_fn<<<blocks, threads, 0, stream>>>(output, lora_result, bias, num_valid_tokens_ptr,
                                              permuted_token_selected_experts, inter_size);
}

template <class T>
__global__ void loraReorderKernel(T* output, T const* lora_result,
                                  int64_t const* num_valid_tokens_ptr, int64_t inter_size) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  int64_t const num_tokens = gridDim.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  T const* lora_result_1 = lora_result + token * inter_size;
  output = output + token * inter_size * 2;

  constexpr int64_t LORA_REORDER_ELEM_PER_THREAD = 128 / sizeof_bits<T>::value;

  using DataElem = cutlass::Array<T, LORA_REORDER_ELEM_PER_THREAD>;
  auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
  auto output_vec = reinterpret_cast<DataElem*>(output);

  int64_t const start_offset = tid;
  int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % LORA_REORDER_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / LORA_REORDER_ELEM_PER_THREAD;

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_1_vec[elem_index];
    output_vec[elem_index] = lora_value;
  }

  auto lora_result_2_vec =
      reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
  int64_t const inter_size_vec = inter_size / LORA_REORDER_ELEM_PER_THREAD;
  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto lora_value = lora_result_2_vec[elem_index];
    output_vec[elem_index + inter_size_vec] = lora_value;
  }
}

template <class T>
void loraReorder(T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr,
                 int64_t inter_size, int64_t num_tokens, cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

  loraReorderKernel<T>
      <<<blocks, threads, 0, stream>>>(output, lora_result, num_valid_tokens_ptr, inter_size);
}

// ============================== DEQUANT_FP8 =================================
constexpr static int DEQUANT_KERNELS_THREADS_PER_BLOCK = 256;

template <class OutputType, class InputType>
__global__ void dequantFP8Kernel(OutputType* output, InputType const* input,
                                 int64_t const* num_valid_tokens_ptr, int64_t inter_size,
                                 float const* scale, bool scale_is_dequant) {
  int64_t const tid = threadIdx.x;
  int64_t const token = blockIdx.x;
  if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr) {
    return;
  }

  output = output + token * inter_size;
  input = input + token * inter_size;

  constexpr int64_t DEQUANT_ELEM_PER_THREAD = 128 / sizeof_bits<InputType>::value;

  using DataElem = cutlass::Array<InputType, DEQUANT_ELEM_PER_THREAD>;
  using OutputElem = cutlass::Array<OutputType, DEQUANT_ELEM_PER_THREAD>;
  using ComputeElem = cutlass::Array<float, DEQUANT_ELEM_PER_THREAD>;
  auto input_vec = reinterpret_cast<DataElem const*>(input);
  auto output_vec = reinterpret_cast<OutputElem*>(output);

  int64_t const start_offset = tid;
  int64_t const stride = DEQUANT_KERNELS_THREADS_PER_BLOCK;
  assert(inter_size % DEQUANT_ELEM_PER_THREAD == 0);
  int64_t const num_elems_in_col = inter_size / DEQUANT_ELEM_PER_THREAD;

  ComputeElem deqaunt_scale_value;
  float dequant_scale = scale[0];
  if (!scale_is_dequant) {
    dequant_scale = 1.f / dequant_scale;
  }
  deqaunt_scale_value.fill(dequant_scale);

  for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride) {
    auto input_value = arrayConvert<DataElem, ComputeElem>(input_vec[elem_index]);
    output_vec[elem_index] =
        arrayConvert<ComputeElem, OutputElem>(input_value * deqaunt_scale_value);
  }
}

template <class OutputType, class InputType>
void dequantFP8(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr,
                int64_t inter_size, int64_t num_tokens, float const* scale, bool scale_is_dequant,
                cudaStream_t stream) {
  int64_t const blocks = num_tokens;
  int64_t const threads = DEQUANT_KERNELS_THREADS_PER_BLOCK;

  dequantFP8Kernel<OutputType, InputType><<<blocks, threads, 0, stream>>>(
      output, input, num_valid_tokens_ptr, inter_size, scale, scale_is_dequant);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                   Enable>::CutlassMoeFCRunner()
#ifdef FAST_BUILD
    : blockscale_gemm_runner_{nullptr}
#else
    : blockscale_gemm_runner_{std::make_unique<
          kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunner<__nv_bfloat16, __nv_fp8_e4m3,
                                                                       __nv_bfloat16>>()}
#endif
{
  check_cuda_error(cudaStreamCreateWithFlags(&lora_stream_, cudaStreamNonBlocking));
  check_cuda_error(cudaEventCreateWithFlags(&lora_start_event_, cudaEventDisableTiming));
  check_cuda_error(cudaEventCreateWithFlags(&lora_ready_event_, cudaEventDisableTiming));
  check_cuda_error(cudaEventCreateWithFlags(&lora_fc2_ready_event_, cudaEventDisableTiming));
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                   Enable>::~CutlassMoeFCRunner() {
  if (lora_ready_event_) {
    (void)cudaEventDestroy(lora_ready_event_);
  }
  if (lora_fc2_ready_event_) {
    (void)cudaEventDestroy(lora_fc2_ready_event_);
  }
  if (lora_start_event_) {
    (void)cudaEventDestroy(lora_start_event_);
  }
  if (lora_stream_) {
    (void)cudaStreamDestroy(lora_stream_);
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
std::map<std::string, std::pair<size_t, size_t>>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                   Enable>::getWorkspaceDeviceBufferSizes(int64_t const num_rows,
                                                          int64_t const hidden_size,
                                                          int64_t const inter_size,
                                                          int const num_experts_per_node,
                                                          int const experts_per_token,
                                                          ActivationType activation_type,
                                                          bool use_lora,
                                                          bool use_deepseek_fp8_block_scale,
                                                          bool use_mxfp8_act_scaling,
                                                          bool min_latency_mode, bool use_awq) {
  size_t num_moe_inputs =
      min_latency_mode ? num_experts_per_node * num_rows : experts_per_token * num_rows;
  size_t const permuted_elems = num_moe_inputs * hidden_size;
  size_t const interbuf_elems = num_moe_inputs * inter_size;
  size_t glu_inter_elems = 0;
  bool is_gated_activation = isGatedActivation(activation_type);
  if (is_gated_activation) {
    glu_inter_elems = interbuf_elems * 2;
  } else if (mayHaveDifferentGEMMOutputType()) {
    // In this case we are using activation quantization, and some intermediate buffers will be
    // unquantized We need to have separate memory for these as we can no longer alias the output
    // buffer for reuse
    glu_inter_elems = interbuf_elems;
  }

  bool using_tma_ws = moe_gemm_runner_.supportsTmaWarpSpecialized();

  size_t const gemm_output_dtype = sizeof(UnfusedGemmOutputType);

  constexpr float dtype_size = act_fp4 ? 0.5f : (use_w4afp8 ? 2.0f : sizeof(T));

  size_t const permuted_row_to_unpermuted_row_size =
      min_latency_mode ? 0 : num_moe_inputs * sizeof(int);
  size_t const permuted_token_selected_experts_size =
      min_latency_mode ? 0 : num_moe_inputs * sizeof(int);

  int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_rows, num_experts_per_node);
  int64_t const num_blocks_per_seq = tensorrt_llm::common::ceilDiv(num_rows, num_tokens_per_block);
  size_t const blocked_expert_counts_size =
      min_latency_mode ? 0 : num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size =
      min_latency_mode ? 0 : num_experts_per_node * num_rows * sizeof(int);

  size_t const permuted_data_size = permuted_elems * dtype_size;
  size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
  size_t const permuted_token_final_scales_size =
      mayHaveFinalizeFused() ? num_moe_inputs * sizeof(float) : 0;
  size_t const glu_inter_size =
      glu_inter_elems * gemm_output_dtype;  // May be an intermediate type for quantization
  size_t const fc1_result_size =
      interbuf_elems * dtype_size;  // Activation quantizes so back to dtype_size
  size_t const fc2_result_size =
      min_latency_mode ? 0
                       : num_moe_inputs * hidden_size *
                             gemm_output_dtype;  // May be an intermediate type for quantization

  // If topk is greater than num_experts_per_node (i.e. large EP value), then we don't need to
  // allocate for the whole tokens*topk
  auto act_sf_rows =
      min_latency_mode
          ? num_moe_inputs
          : std::min(num_moe_inputs, static_cast<size_t>(num_rows * num_experts_per_node));
  size_t const sf_size =
      getScalingType() == TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX
          ? sizeof(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF)
          : sizeof(TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF);

  size_t const fc1_fp4_act_scale_size =
      getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size, getScalingType()) *
      sf_size;
  size_t const fc2_fp4_act_scale_size =
      getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size, getScalingType()) *
      sf_size;
  size_t const fp4_act_scale_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);

  size_t const tma_ws_size = using_tma_ws ? TmaWarpSpecializedGroupedGemmInput::workspaceSize(
                                                num_experts_per_node, getScalingType())
                                          : 0;

  size_t const gemm_workspace_size = moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);

  // lora related
  size_t const lora_input_size =
      (use_lora && use_fp8) ? std::max(permuted_elems, interbuf_elems) * sizeof(ScaleBiasType) : 0;
  size_t const lora_fc1_result_size =
      use_lora ? (is_gated_activation ? 2 * interbuf_elems * sizeof(ScaleBiasType)
                                      : interbuf_elems * sizeof(ScaleBiasType))
               : 0;
  size_t const lora_add_bias_size = use_lora ? lora_fc1_result_size : 0;
  size_t const lora_fc2_result_size = use_lora ? permuted_elems * sizeof(ScaleBiasType) : 0;

  // We do some overlapping of the large workspace buffers. Although we could overlap some of the
  // other buffers, they are small enough (i.e no factor of hidden size) they will only be a couple
  // MiB at most, so we don't bother in the case of fused activation we overlap permuted_data and
  // fc2_result in the case of unfused activation we overlap permuted_data and fc1_result we need to
  // calculate the max possible size, so use the max of all three
  size_t overlapped_gemm1_gemm2_inputs_size = std::max(permuted_data_size, fc2_result_size);
  // When glu_inter_elems is 0 we are always fused, otherwise we may need the un-fused case
  if (glu_inter_elems > 0) {
    overlapped_gemm1_gemm2_inputs_size =
        std::max(overlapped_gemm1_gemm2_inputs_size, fc1_result_size);
  }

  size_t const alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float*);

  // if we have glu_inter we overlap it with fc2_result, otherwise we use fc1_result by itself
  size_t overlapped_gemm1_gemm2_outputs_size = fc1_result_size;
  if (glu_inter_elems > 0) {
    overlapped_gemm1_gemm2_outputs_size =
        std::max(std::max(glu_inter_size, fc2_result_size), overlapped_gemm1_gemm2_outputs_size);
  }

  size_t smoothed_act_size = use_awq
                                 ? std::max(permuted_elems, interbuf_elems) * sizeof(T) * 2
                                 : 0;  // Extra workspace required by AWQ for smoothing activations
  size_t deepseek_fc_workspace_size = 0;
  if (use_deepseek_fp8_block_scale) {
    size_t factor = is_gated_activation ? 2 : 1;
    size_t blockscale_fc1_output_size = factor * interbuf_elems * gemm_output_dtype;
    size_t blockscale_fc2_output_size = permuted_elems * gemm_output_dtype;
    overlapped_gemm1_gemm2_inputs_size =
        std::max(std::max(permuted_data_size, fc1_result_size), blockscale_fc2_output_size);
    overlapped_gemm1_gemm2_outputs_size = blockscale_fc1_output_size;

    auto* blockscale_gemm_runner = getDeepSeekBlockScaleGemmRunner();
    TLLM_CHECK(blockscale_gemm_runner != nullptr);
    auto deepseek_fc1_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
        num_rows, factor * inter_size, hidden_size, experts_per_token, num_experts_per_node);
    auto deepseek_fc2_workspace_size = blockscale_gemm_runner->getWorkspaceSize(
        num_rows, hidden_size, inter_size, experts_per_token, num_experts_per_node);
    deepseek_fc_workspace_size = std::max(deepseek_fc1_workspace_size, deepseek_fc2_workspace_size);
  }

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                                                        \
  do {                                                                              \
    auto aligned_size =                                                             \
        tensorrt_llm::common::alignSize(size, tensorrt_llm::common::kCudaMemAlign); \
    out_map[#name] = std::pair{aligned_size, map_offset};                           \
    map_offset += aligned_size;                                                     \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(permuted_row_to_unpermuted_row);
  ADD(permuted_token_selected_experts);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(expert_first_token_offset);
  ADD(permuted_token_final_scales);
  ADD(overlapped_gemm1_gemm2_inputs);
  ADD(overlapped_gemm1_gemm2_outputs);
  ADD_NAME(alpha_scale_ptr_array_fc1, alpha_scale_ptr_array_size);
  ADD_NAME(alpha_scale_ptr_array_fc2, alpha_scale_ptr_array_size);
  ADD(fp4_act_scale);
  ADD_NAME(tma_ws_gemm1_workspace, tma_ws_size);
  ADD_NAME(tma_ws_gemm2_workspace, tma_ws_size);
  ADD(gemm_workspace);
  ADD(lora_input);
  ADD(lora_fc1_result);
  ADD(lora_add_bias);
  ADD(lora_fc2_result);
  ADD(deepseek_fc_workspace);
  ADD(smoothed_act);

  return out_map;

#undef ADD_NAME
#undef ADD
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
size_t CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                          Enable>::getWorkspaceSize(int64_t const num_rows,
                                                    int64_t const hidden_size,
                                                    int64_t const inter_size, int const num_experts,
                                                    int const experts_per_token,
                                                    ActivationType activation_type,
                                                    MOEParallelismConfig parallelism_config,
                                                    bool use_lora,
                                                    bool use_deepseek_fp8_block_scale,
                                                    bool use_mxfp8_act_scaling,
                                                    bool min_latency_mode, bool use_awq) {
  int const ep_size = parallelism_config.ep_size;
  TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0,
                       "Number of experts must be a multiple of ep size");
  auto sizes_map = getWorkspaceDeviceBufferSizes(
      num_rows, hidden_size, inter_size, num_experts / ep_size, experts_per_token, activation_type,
      use_lora, use_deepseek_fp8_block_scale, use_mxfp8_act_scaling, min_latency_mode, use_awq);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(),
                 [](auto& v) { return v.second.first; });
  size_t size = tensorrt_llm::common::calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  TLLM_LOG_TRACE("Mixture Of Experts Plugin requires workspace of %2f MiB", size / 1024.f / 1024.f);
  return size;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Enable>::configureWsPtrs(char* ws_ptr, int64_t const num_rows,
                                                 int64_t const hidden_size,
                                                 int64_t const inter_size,
                                                 int const num_experts_per_node,
                                                 int const experts_per_token,
                                                 ActivationType activation_type,
                                                 MOEParallelismConfig parallelism_config,
                                                 bool use_lora, bool use_deepseek_fp8_block_scale,
                                                 bool use_mxfp8_act_scaling, bool min_latency_mode,
                                                 bool use_awq) {
  auto workspaces = getWorkspaceDeviceBufferSizes(
      num_rows, hidden_size, inter_size, num_experts_per_node, experts_per_token, activation_type,
      use_lora, use_deepseek_fp8_block_scale, use_mxfp8_act_scaling, min_latency_mode, use_awq);

  auto getWsPtr = [&](auto type, std::string const& name) {
    return workspaces.at(name).first
               ? reinterpret_cast<decltype(type)*>(ws_ptr + workspaces.at(name).second)
               : nullptr;
  };

  permuted_row_to_unpermuted_row_ = getWsPtr(int{}, "permuted_row_to_unpermuted_row");
  permuted_token_selected_experts_ = getWsPtr(int{}, "permuted_token_selected_experts");
  blocked_expert_counts_ = getWsPtr(int{}, "blocked_expert_counts");
  blocked_expert_counts_cumsum_ = getWsPtr(int{}, "blocked_expert_counts_cumsum");
  blocked_row_to_unpermuted_row_ = getWsPtr(int{}, "blocked_row_to_unpermuted_row");

  expert_first_token_offset_ = getWsPtr(int64_t{}, "expert_first_token_offset");

  // We check if the provided config uses fused finalize and disable it if it does not
  bool gemm2_using_finalize_fusion =
      gemm2_config_->epilogue_fusion_type ==
      cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
  permuted_token_final_scales_ =
      gemm2_using_finalize_fusion ? getWsPtr(float{}, "permuted_token_final_scales") : nullptr;

  bool const is_gated_activation = isGatedActivation(activation_type);
  bool const gemm1_using_fused_moe = moe_gemm_runner_.isFusedGatedActivation(
      *gemm1_config_, activation_type, inter_size, hidden_size);
  bool const gemm1_using_tma_ws = moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_);
  bool const tma_ws_has_glu =
      gemm1_using_tma_ws && (mayHaveDifferentGEMMOutputType() || is_gated_activation);
  // We always use fused path if we can
  bool const non_tma_ws_has_glu = !gemm1_using_fused_moe && is_gated_activation;
  bool const has_glu_inter_result = tma_ws_has_glu || non_tma_ws_has_glu || use_fp8;

  // Always same value, but overlapped with either fc1_result_ or fc2_result_
  permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
  // Always same value, ignored if not needed
  glu_inter_result_ =
      has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs") : nullptr;

  // fc1 and fc2 alias one of the above pointers, but it depends on if actfn is fused/unfused which
  // is overlapped NOTE: It is important to get the overlapped pointers correct as the wrong order
  // will cause the buffer to be used as an input and output for the same gemm, which will cause
  // corruption
  fc1_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
  fc2_result_ = has_glu_inter_result ? getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs")
                                     : getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");

  if (use_deepseek_fp8_block_scale) {
    permuted_data_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
    fc1_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
    glu_inter_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_outputs");
    fc2_result_ = getWsPtr(T{}, "overlapped_gemm1_gemm2_inputs");
  }

  alpha_scale_ptr_array_fc1_ = getWsPtr((float const*)(nullptr), "alpha_scale_ptr_array_fc1");
  alpha_scale_ptr_array_fc2_ = getWsPtr((float const*)(nullptr), "alpha_scale_ptr_array_fc2");

  // NOTE: We alias these, but if we fuse the quantization for GEMM2 into GEMM1 they will need
  // separated
  fc1_fp4_act_scale_ = nullptr;
  fc2_fp4_act_scale_ = nullptr;
  if (use_block_scaling) {
    fc1_fp4_act_scale_ = getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fp4_act_scale");
    fc2_fp4_act_scale_ = getWsPtr(TmaWarpSpecializedGroupedGemmInput::ElementSF{}, "fp4_act_scale");
    TLLM_CHECK(fc1_fp4_act_scale_ != nullptr);
    TLLM_CHECK(fc2_fp4_act_scale_ != nullptr);
  }

  tma_ws_grouped_gemm1_input_ = {};
  tma_ws_grouped_gemm2_input_ = {};
  if (moe_gemm_runner_.supportsTmaWarpSpecialized()) {
    tma_ws_grouped_gemm1_input_.configureWorkspace(
        getWsPtr(int8_t{}, "tma_ws_gemm1_workspace"), num_experts_per_node,
        getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
        getScalingType());
    tma_ws_grouped_gemm2_input_.configureWorkspace(
        getWsPtr(int8_t{}, "tma_ws_gemm2_workspace"), num_experts_per_node,
        getWsPtr(int8_t{}, "gemm_workspace"), workspaces.at("gemm_workspace").first,
        getScalingType());
  }

  lora_fc1_result_ = {};
  lora_add_bias_ = {};
  lora_fc2_result_ = {};

  if (use_lora) {
    lora_input_ = getWsPtr(ScaleBiasType{}, "lora_input");
    lora_fc1_result_ = getWsPtr(ScaleBiasType{}, "lora_fc1_result");
    lora_add_bias_ = getWsPtr(ScaleBiasType{}, "lora_add_bias");
    lora_fc2_result_ = getWsPtr(ScaleBiasType{}, "lora_fc2_result");
    TLLM_CHECK_WITH_INFO(!use_fp8 || lora_input_ != nullptr,
                         "LoRA input must not be nullptr if FP8 is enabled");
    TLLM_CHECK(lora_fc1_result_ != nullptr);
    TLLM_CHECK(lora_add_bias_ != nullptr);
    TLLM_CHECK(lora_fc2_result_ != nullptr);
  }

  if (use_deepseek_fp8_block_scale) {
    auto* blockscale_gemm_runner = getDeepSeekBlockScaleGemmRunner();
    TLLM_CHECK(blockscale_gemm_runner != nullptr);
    blockscale_gemm_runner->configureWorkspace(getWsPtr(char{}, "deepseek_fc_workspace"));
  }

  if (use_awq) {
    smoothed_act_ = getWsPtr(int8_t{}, "smoothed_act");
  }
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType,
          bool IsMXFPX, class Enable>
kernels::fp8_blockscale_gemm::CutlassFp8BlockScaleGemmRunnerInterface*
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, IsMXFPX,
                   Enable>::getDeepSeekBlockScaleGemmRunner() const {
  TLLM_CHECK_WITH_INFO(
      (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<OutputType, __nv_bfloat16>),
      "Block scale GEMM runner only supports BF16 A/output");
  TLLM_CHECK_WITH_INFO((std::is_same_v<WeightType, __nv_fp8_e4m3>),
                       "Block scale GEMM runner only supports FP8 weights.");
  return blockscale_gemm_runner_.get();
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, IsMXFPX, Enable>::
    BlockScaleFC1(DeepSeekBlockScaleGemmRunner& gemm_runner, T const* const input, T* const output,
                  void* const gemm_output, int64_t const* const expert_first_token_offset,
                  WeightType const* const fc1_expert_weights,
                  ScaleBiasType const* const fc1_expert_biases, float const* const fc2_fp8_quant,
                  int64_t const num_rows, int64_t const expanded_num_rows,
                  int64_t const hidden_size, int64_t const inter_size,
                  int const num_experts_per_node, ActivationParams fc1_activation_type,
                  QuantParams& quant_params, bool enable_pdl, cudaStream_t stream) {
  bool const is_gated_activation = isGatedActivation(fc1_activation_type);

  int shape_n = is_gated_activation ? inter_size * 2 : inter_size;
  int shape_k = hidden_size;

  // NOTE: we assume gemm_runner.configureWorkspace has already been called.
  gemm_runner.moeGemm(gemm_output, input, fc1_expert_weights, expert_first_token_offset,
                      num_experts_per_node, shape_n, shape_k, stream, nullptr,
                      quant_params.fp8_block_scaling.fc1_scales_ptrs);

  sync_check_cuda_error(stream);
  constexpr bool bias_is_broadcast = true;
  constexpr bool use_per_expert_act_scale = false;
  doActivation<T, UnfusedGemmOutputType>(
      output, static_cast<UnfusedGemmOutputType const*>(gemm_output), fc2_fp8_quant,
      fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
      inter_size, expanded_num_rows, fc1_activation_type, quant_params, use_per_expert_act_scale,
      nullptr, enable_pdl, stream);

  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, IsMXFPX, Enable>::
    BlockScaleFC2(
        DeepSeekBlockScaleGemmRunner& gemm_runner, T const* const input, void* const gemm_output,
        OutputType* const final_output, int64_t const* const expert_first_token_offset,
        WeightType const* const fc2_expert_weights, ScaleBiasType const* const fc2_expert_biases,
        float const* const unpermuted_final_scales, int const* const unpermuted_row_to_permuted_row,
        int const* const permuted_row_to_unpermuted_row, int const* const token_selected_experts,
        int64_t const* const num_valid_tokens_ptr, int64_t const num_rows,
        int64_t const expanded_num_rows, int64_t const hidden_size,
        int64_t const unpadded_hidden_size, int64_t const inter_size,
        int64_t const num_experts_per_node, int64_t const k,
        MOEParallelismConfig parallelism_config, bool const enable_alltoall,
        QuantParams& quant_params, bool enable_pdl, cudaStream_t stream) {
  int shape_n = hidden_size;
  int shape_k = inter_size;

  // NOTE: we assume gemm_runner.configureWorkspace has already been called.
  gemm_runner.moeGemm(gemm_output, input, fc2_expert_weights, expert_first_token_offset,
                      num_experts_per_node, shape_n, shape_k, stream, nullptr,
                      quant_params.fp8_block_scaling.fc2_scales_ptrs);

  sync_check_cuda_error(stream);

  finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
      static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
      unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
      token_selected_experts, expert_first_token_offset, num_rows, hidden_size,
      unpadded_hidden_size, k, num_experts_per_node, parallelism_config, enable_alltoall,
      enable_pdl, stream);
}

template <class T, class WeightType, class OutputType, class InputType, class ScaleBiasType,
          bool IsMXFPX, class Enable>
T const* CutlassMoeFCRunner<T, WeightType, OutputType, InputType, ScaleBiasType, IsMXFPX,
                            Enable>::applyPrequantScale(void* smoothed_act,
                                                        void const* permuted_data,
                                                        void const* prequant_scales,
                                                        int64_t const* num_valid_tokens_ptr,
                                                        int64_t const expanded_num_rows,
                                                        int64_t const seq_len, bool const use_awq,
                                                        cudaStream_t stream) {
  T const* gemm_input;
  bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
  if (use_prequant_scale_kernel) {
    TLLM_CHECK_WITH_INFO((!std::is_same_v<T, WeightType>),
                         "Prequant scales are only used for different weight/activation type!");
    if constexpr (!std::is_same_v<T, WeightType>) {
      tensorrt_llm::kernels::apply_per_channel_scale_kernel_launcher<UnfusedGemmOutputType, T>(
          reinterpret_cast<T*>(smoothed_act),
          reinterpret_cast<UnfusedGemmOutputType const*>(permuted_data),
          reinterpret_cast<UnfusedGemmOutputType const*>(prequant_scales), expanded_num_rows,
          seq_len, num_valid_tokens_ptr, stream);
    }
    gemm_input = reinterpret_cast<T const*>(smoothed_act);
  } else {
    gemm_input = reinterpret_cast<T const*>(permuted_data);
  }
  sync_check_cuda_error(stream);
  return gemm_input;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Enable>::gemm1(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType, IsMXFPX>& gemm_runner,
    DeepSeekBlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input, T* const output,
    void* const intermediate_result, int64_t const* const expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template,
    WeightType const* const fc1_expert_weights, ScaleBiasType const* const fc1_expert_biases,
    int64_t const* const num_valid_tokens_ptr, ScaleBiasType const* const fc1_int_scales,
    float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
    TmaWarpSpecializedGroupedGemmInput::ElementSF* fc2_fp4_act_flat, QuantParams quant_params,
    int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size,
    int64_t const inter_size, int const num_experts_per_node, ActivationParams fc1_activation_type,
    float const** alpha_scale_ptr_array, bool bias_is_broadcast, cudaStream_t stream,
    cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode,
    int* num_active_experts_per, int* active_expert_global_ids, bool enable_pdl,
    cudaEvent_t pre_activation_event, int32_t const* fc1_lora_bias_ranks,
    std::function<void()> const* pre_activation_work) {
  if (fp8_blockscale_gemm_runner) {
    TLLM_CHECK(!min_latency_mode);
    Self::BlockScaleFC1(*fp8_blockscale_gemm_runner, input, output, intermediate_result,
                        expert_first_token_offset, fc1_expert_weights, fc1_expert_biases,
                        fc2_fp8_quant, num_rows, expanded_num_rows, hidden_size, inter_size,
                        num_experts_per_node, fc1_activation_type, quant_params, enable_pdl,
                        stream);
    return;
  }

  bool const using_tma_ws_gemm1 = gemm_runner.isTmaWarpSpecialized(config);
  bool const is_gated_activation = isGatedActivation(fc1_activation_type);
  bool const use_ampere_activation_fusion = gemm_runner.isFusedGatedActivation(
      config, fc1_activation_type.activation_type, inter_size, hidden_size);
  size_t const fc1_out_size =
      ((!use_ampere_activation_fusion) && is_gated_activation) ? inter_size * 2 : inter_size;

  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  if (min_latency_mode) {
    TLLM_CHECK_WITH_INFO(using_tma_ws_gemm1,
                         "Only TMA warp specialized GEMM is supported in min latency mode.");
    // TODO: as for bias, need to get the correct expert id according to the active expert global
    // ids
    TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Min latency mode does not support bias.");
    TLLM_CHECK(use_fp4);
  }

  if (using_tma_ws_gemm1) {
    TLLM_CHECK(config.is_tma_warp_specialized);
    TLLM_CHECK(!use_ampere_activation_fusion);
    TLLM_CHECK(!use_fp4 || fc1_fp4_act_flat);
    TLLM_CHECK(!use_fp4 || fc2_fp4_act_flat);

    bool has_different_gemm_output_type = using_tma_ws_gemm1 && !std::is_same_v<T, OutputType>;
    bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
    TLLM_CHECK_WITH_INFO(has_intermediate || input != output,
                         "Input and output buffers are overlapping");
    auto* gemm_output = has_intermediate ? intermediate_result : static_cast<void*>(output);

    auto tma_ws_input = tma_ws_input_template;

    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                     quant_params.groupwise.fc1.alpha, stream);
    }

    auto universal_input =
        GroupedGemmInput<T, WeightType, OutputType, OutputType>{input,
                                                                total_tokens_including_expert,
                                                                /*weights*/ nullptr,
                                                                /*scales*/ nullptr,
                                                                /*zeros*/ nullptr,
                                                                /*biases*/ nullptr,
                                                                /*C*/ nullptr,
                                                                alpha_scale_ptr_array,
                                                                /*occupancy*/ nullptr,
                                                                fc1_activation_type,
                                                                num_rows,
                                                                /*N*/ int64_t(fc1_out_size),
                                                                /*K*/ hidden_size,
                                                                num_experts_per_node,
                                                                quant_params.groupwise.group_size,
                                                                /*bias_is_broadcast*/ true,
                                                                /*use_fused_moe*/ false,
                                                                stream,
                                                                config};
    gemm_runner.moeGemm(universal_input, tma_ws_input);

    if (pre_activation_work != nullptr) {
      // Queue dependent LoRA work while GEMM1 is still running. In debug builds
      // sync_check_cuda_error may synchronize the main stream, so launching this
      // callback before the check is what preserves cross-stream overlap.
      (*pre_activation_work)();
    }
    sync_check_cuda_error(stream);

    // TODO: when bias_is_broadcast is false, fuse bias to gemm
    using GatedActOutputType = std::conditional_t<use_w4afp8, BackBoneType, T>;
    bool use_per_expert_act_scale = use_fp4 ? quant_params.fp4.fc2.use_per_expert_act_scale
                                    : use_wfp4afp8
                                        ? quant_params.fp8_mxfp4.fc2.use_per_expert_act_scale
                                    : use_fp8 ? quant_params.fp8.fc2_use_per_expert_act_scale
                                              : false;

    if (pre_activation_event != nullptr) {
      check_cuda_error(cudaStreamWaitEvent(stream, pre_activation_event, 0));
    }

    doActivation<GatedActOutputType, UnfusedGemmOutputType>(
        reinterpret_cast<GatedActOutputType*>(output),
        static_cast<UnfusedGemmOutputType const*>(gemm_output), fc2_fp8_quant, fc1_expert_biases,
        bias_is_broadcast, expert_first_token_offset, num_experts_per_node, inter_size,
        expanded_num_rows, fc1_activation_type, quant_params, use_per_expert_act_scale,
        fc2_fp4_act_flat, enable_pdl, stream, fc1_lora_bias_ranks);

    sync_check_cuda_error(stream);
  } else if (use_fp8) {
    TLLM_CHECK(!use_ampere_activation_fusion);
    TLLM_CHECK(!config.is_tma_warp_specialized);
    TLLM_CHECK(!use_block_scaling);

    alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                   quant_params.fp8.dequant_fc1, stream);

    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{
        input,
        total_tokens_including_expert,
        fc1_expert_weights,
        /*scales*/ nullptr,
        /*zeros*/ nullptr,
        /*biases*/ nullptr,
        reinterpret_cast<UnfusedGemmOutputType*>(intermediate_result),
        alpha_scale_ptr_array,
        /*occupancy*/ nullptr,
        fc1_activation_type,
        expanded_num_rows,
        /*N*/ int64_t(fc1_out_size),
        /*K*/ hidden_size,
        num_experts_per_node,
        quant_params.groupwise.group_size,
        /*bias_is_broadcast*/ true,
        /*use_fused_moe*/ false,
        stream,
        config};
    gemm_runner.moeGemm(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    bool use_per_expert_act_scale = use_fp8 ? quant_params.fp8.fc2_use_per_expert_act_scale : false;
    doActivation<T, UnfusedGemmOutputType>(
        output, static_cast<UnfusedGemmOutputType const*>(intermediate_result), fc2_fp8_quant,
        fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
        inter_size, expanded_num_rows, fc1_activation_type, quant_params, use_per_expert_act_scale,
        nullptr, enable_pdl, stream);

    sync_check_cuda_error(stream);
  } else if (!is_gated_activation) {
    TLLM_CHECK(!use_ampere_activation_fusion);
    TLLM_CHECK(!config.is_tma_warp_specialized);
    TLLM_CHECK(!use_block_scaling);
    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                     quant_params.groupwise.fc1.alpha, stream);
    }
    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{
        input,
        total_tokens_including_expert,
        fc1_expert_weights,
        /*scales*/ quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_scales)
            : fc1_int_scales,
        /*zeros*/ quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_zeros)
            : nullptr,
        fc1_expert_biases,
        reinterpret_cast<OutputType*>(output),
        alpha_scale_ptr_array,
        /*occupancy*/ nullptr,
        fc1_activation_type,
        expanded_num_rows,
        /*N*/ int64_t(fc1_out_size),
        /*K*/ hidden_size,
        num_experts_per_node,
        quant_params.groupwise.group_size,
        bias_is_broadcast,
        /*use_fused_moe*/ false,
        stream,
        config};
    gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    sync_check_cuda_error(stream);
  } else {
    TLLM_CHECK(!config.is_tma_warp_specialized);
    TLLM_CHECK(is_gated_activation);
    TLLM_CHECK_WITH_INFO(!use_ampere_activation_fusion || input != output,
                         "Input and output buffers are overlapping");
    TLLM_CHECK(!use_block_scaling);
    if (use_w4afp8) {
      alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                     quant_params.groupwise.fc1.alpha, stream);
    }
    // Run the GEMM with activation function overridden with `Identity`, we do the activation
    // separately
    auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{
        input,
        total_tokens_including_expert,
        fc1_expert_weights,
        /*scales*/ quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_scales)
            : fc1_int_scales,
        /*zeros*/ quant_params.groupwise.group_size > 0
            ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc1.weight_zeros)
            : nullptr,
        fc1_expert_biases,
        static_cast<OutputType*>(use_ampere_activation_fusion ? output : intermediate_result),
        alpha_scale_ptr_array,
        /*occupancy*/ nullptr,
        use_ampere_activation_fusion ? fc1_activation_type.activation_type
                                     : ActivationType::Identity,
        expanded_num_rows,
        /*N*/ int64_t(fc1_out_size),
        /*K*/ hidden_size,
        num_experts_per_node,
        quant_params.groupwise.group_size,
        bias_is_broadcast,
        use_ampere_activation_fusion,
        stream,
        config};
    gemm_runner.moeGemmBiasAct(universal_input, TmaWarpSpecializedGroupedGemmInput{});

    sync_check_cuda_error(stream);

    if (!use_ampere_activation_fusion) {
      using GatedActOutputType = std::conditional_t<use_w4afp8, BackBoneType, T>;
      doGatedActivation<GatedActOutputType, UnfusedGemmOutputType>(
          reinterpret_cast<GatedActOutputType*>(output),
          static_cast<UnfusedGemmOutputType const*>(intermediate_result), expert_first_token_offset,
          inter_size, expanded_num_rows, num_experts_per_node, fc1_activation_type, stream);

      sync_check_cuda_error(stream);
    }
  }
}

static __global__ void addLoraBiasRankAwareBf16x2ByRowKernel(
    __nv_bfloat16* gemm_output, __nv_bfloat16 const* fc2_lora, int32_t const* fc2_lora_ranks,
    int64_t const* expert_first_token_offset, int num_experts_per_node, int64_t hidden_size) {
  int64_t const num_valid_tokens = expert_first_token_offset[num_experts_per_node];
  int64_t const hidden_pairs = hidden_size / 2;

  for (int64_t row = blockIdx.x; row < num_valid_tokens; row += gridDim.x) {
    if (fc2_lora_ranks[row] <= 0) {
      continue;
    }

    auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(gemm_output + row * hidden_size);
    auto const* lora_pairs = reinterpret_cast<__nv_bfloat162 const*>(fc2_lora + row * hidden_size);
    for (int64_t col_pair = threadIdx.x; col_pair < hidden_pairs; col_pair += blockDim.x) {
      float2 output_value = __bfloat1622float2(output_pairs[col_pair]);
      float2 lora_value = __bfloat1622float2(lora_pairs[col_pair]);
      output_pairs[col_pair] =
          __floats2bfloat162_rn(output_value.x + lora_value.x, output_value.y + lora_value.y);
    }
  }
}

inline void addLoraBiasRankAwareBf16x2ByRow(__nv_bfloat16* gemm_output,
                                            __nv_bfloat16 const* fc2_lora,
                                            int32_t const* fc2_lora_ranks,
                                            int64_t const* expert_first_token_offset,
                                            int num_experts_per_node, int64_t expanded_num_rows,
                                            int64_t hidden_size, cudaStream_t stream) {
  constexpr int threads = 1024;
  int const blocks =
      static_cast<int>(std::min<int64_t>(std::max<int64_t>(expanded_num_rows, 1), 65535));
  addLoraBiasRankAwareBf16x2ByRowKernel<<<blocks, threads, 0, stream>>>(
      gemm_output, fc2_lora, fc2_lora_ranks, expert_first_token_offset, num_experts_per_node,
      hidden_size);
}

template <bool AddToOutput>
static __global__ void fc2LoraRank16Bf16GroupedOutputKernel(
    __nv_bfloat16* output, __nv_bfloat16 const* low_rank_workspace,
    int64_t const* expert_first_token_offset, int64_t out_hidden_size, int num_lora_adapters,
    int32_t const* adapter_lora_ranks, void const* const* adapter_lora_weight_ptrs,
    int32_t const* permuted_lora_indices, int32_t const* adapter_expert_row_counts,
    int32_t const* adapter_expert_row_indices, int adapter_expert_row_capacity,
    int32_t const* adapter_expert_row_starts, bool rows_grouped_by_adapter) {
  int64_t const out_pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const out_hidden_pairs = out_hidden_size >> 1;
  int const adapter = static_cast<int>(blockIdx.y);
  int const expert = static_cast<int>(blockIdx.z);
  if (out_pair >= out_hidden_pairs || adapter >= num_lora_adapters) {
    return;
  }

  bool const has_row_list = adapter_expert_row_counts != nullptr &&
                            adapter_expert_row_indices != nullptr &&
                            adapter_expert_row_capacity > 0;
  bool use_row_list = false;
  int row_count = 0;
  int row_start = 0;
  int32_t const* row_indices = nullptr;
  if (has_row_list) {
    int const bucket = expert * num_lora_adapters + adapter;
    row_count = adapter_expert_row_counts[bucket];
    if (row_count <= 0) {
      return;
    }
    if (rows_grouped_by_adapter && adapter_expert_row_starts != nullptr) {
      row_start = adapter_expert_row_starts[bucket];
      if (row_start < 0) {
        return;
      }
    } else if (row_count <= adapter_expert_row_capacity) {
      use_row_list = true;
      row_indices =
          adapter_expert_row_indices + static_cast<int64_t>(bucket) * adapter_expert_row_capacity;
    }
  }

  int32_t const raw_lora_rank = adapter_lora_ranks[adapter];
  int const lora_rank = raw_lora_rank <= 0 ? 0 : (raw_lora_rank < 16 ? raw_lora_rank : 16);
  if (lora_rank <= 0) {
    return;
  }

  auto const* lora_b_base =
      reinterpret_cast<__nv_bfloat16 const*>(adapter_lora_weight_ptrs[adapter * 2 + 1]);
  if (lora_b_base == nullptr) {
    return;
  }
  lora_b_base += static_cast<size_t>(expert) * static_cast<size_t>(out_hidden_size) * 16;

  int64_t const out_idx = out_pair << 1;
  auto const* lora_b_row0_pairs =
      reinterpret_cast<__nv_bfloat162 const*>(lora_b_base + static_cast<size_t>(out_idx) * 16);
  auto const* lora_b_row1_pairs =
      reinterpret_cast<__nv_bfloat162 const*>(lora_b_base + static_cast<size_t>(out_idx + 1) * 16);
  __nv_bfloat162 lora_b0_pairs[8];
  __nv_bfloat162 lora_b1_pairs[8];
#pragma unroll
  for (int rank_pair = 0; rank_pair < 8; ++rank_pair) {
    lora_b0_pairs[rank_pair] = lora_b_row0_pairs[rank_pair];
    lora_b1_pairs[rank_pair] = lora_b_row1_pairs[rank_pair];
  }

  auto add_row = [&](int64_t row) {
    auto const* low_rank_pairs =
        reinterpret_cast<__nv_bfloat162 const*>(low_rank_workspace + row * 16);
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    if (lora_rank == 16) {
#pragma unroll
      for (int rank_pair = 0; rank_pair < 8; ++rank_pair) {
        float2 const low_rank_value = __bfloat1622float2(low_rank_pairs[rank_pair]);
        float2 const lora_b0_value = __bfloat1622float2(lora_b0_pairs[rank_pair]);
        float2 const lora_b1_value = __bfloat1622float2(lora_b1_pairs[rank_pair]);
        acc0 += low_rank_value.x * lora_b0_value.x;
        acc0 += low_rank_value.y * lora_b0_value.y;
        acc1 += low_rank_value.x * lora_b1_value.x;
        acc1 += low_rank_value.y * lora_b1_value.y;
      }
    } else {
      int const rank_pairs = lora_rank >> 1;
#pragma unroll
      for (int rank_pair = 0; rank_pair < 8; ++rank_pair) {
        if (rank_pair < rank_pairs) {
          float2 const low_rank_value = __bfloat1622float2(low_rank_pairs[rank_pair]);
          float2 const lora_b0_value = __bfloat1622float2(lora_b0_pairs[rank_pair]);
          float2 const lora_b1_value = __bfloat1622float2(lora_b1_pairs[rank_pair]);
          acc0 += low_rank_value.x * lora_b0_value.x;
          acc0 += low_rank_value.y * lora_b0_value.y;
          acc1 += low_rank_value.x * lora_b1_value.x;
          acc1 += low_rank_value.y * lora_b1_value.y;
        }
      }
      if ((lora_rank & 1) != 0) {
        int const rank_idx = rank_pairs << 1;
        auto const* low_rank = reinterpret_cast<__nv_bfloat16 const*>(low_rank_pairs);
        auto const* lora_b_row0 = reinterpret_cast<__nv_bfloat16 const*>(lora_b0_pairs);
        auto const* lora_b_row1 = reinterpret_cast<__nv_bfloat16 const*>(lora_b1_pairs);
        float const low_rank_value = __bfloat162float(low_rank[rank_idx]);
        acc0 += low_rank_value * __bfloat162float(lora_b_row0[rank_idx]);
        acc1 += low_rank_value * __bfloat162float(lora_b_row1[rank_idx]);
      }
    }

    auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output + row * out_hidden_size);
    if constexpr (AddToOutput) {
      float2 output_value = __bfloat1622float2(output_pairs[out_pair]);
      acc0 += output_value.x;
      acc1 += output_value.y;
    }
    output_pairs[out_pair] = __floats2bfloat162_rn(acc0, acc1);
  };

  if (use_row_list) {
    for (int offset = 0; offset < row_count; ++offset) {
      add_row(row_indices[offset]);
    }
  } else if (rows_grouped_by_adapter && adapter_expert_row_starts != nullptr) {
    for (int offset = 0; offset < row_count; ++offset) {
      add_row(static_cast<int64_t>(row_start) + offset);
    }
  } else {
    int64_t const begin = expert_first_token_offset[expert];
    int64_t const end = expert_first_token_offset[expert + 1];
    for (int64_t row = begin; row < end; ++row) {
      if (permuted_lora_indices[row] != adapter) {
        continue;
      }
      add_row(row);
    }
  }
}

inline void fc2LoraRank16Bf16GroupedOutputAdd(
    __nv_bfloat16* output, __nv_bfloat16 const* low_rank_workspace,
    int64_t const* expert_first_token_offset, int64_t out_hidden_size, int num_experts_per_node,
    int num_lora_adapters, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int32_t const* permuted_lora_indices,
    int32_t const* adapter_expert_row_counts, int32_t const* adapter_expert_row_indices,
    int adapter_expert_row_capacity, int32_t const* adapter_expert_row_starts,
    bool rows_grouped_by_adapter, cudaStream_t stream) {
  constexpr int threads = 256;
  dim3 const grid(static_cast<unsigned int>(((out_hidden_size >> 1) + threads - 1) / threads),
                  static_cast<unsigned int>(num_lora_adapters),
                  static_cast<unsigned int>(num_experts_per_node));
  fc2LoraRank16Bf16GroupedOutputKernel<true><<<grid, threads, 0, stream>>>(
      output, low_rank_workspace, expert_first_token_offset, out_hidden_size, num_lora_adapters,
      adapter_lora_ranks, adapter_lora_weight_ptrs, permuted_lora_indices,
      adapter_expert_row_counts, adapter_expert_row_indices, adapter_expert_row_capacity,
      adapter_expert_row_starts, rows_grouped_by_adapter);
}

inline void fc2LoraRank16Bf16GroupedOutput(
    __nv_bfloat16* output, __nv_bfloat16 const* low_rank_workspace,
    int64_t const* expert_first_token_offset, int64_t out_hidden_size, int num_experts_per_node,
    int num_lora_adapters, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int32_t const* permuted_lora_indices,
    int32_t const* adapter_expert_row_counts, int32_t const* adapter_expert_row_indices,
    int adapter_expert_row_capacity, int32_t const* adapter_expert_row_starts,
    bool rows_grouped_by_adapter, cudaStream_t stream) {
  constexpr int threads = 256;
  dim3 const grid(static_cast<unsigned int>(((out_hidden_size >> 1) + threads - 1) / threads),
                  static_cast<unsigned int>(num_lora_adapters),
                  static_cast<unsigned int>(num_experts_per_node));
  fc2LoraRank16Bf16GroupedOutputKernel<false><<<grid, threads, 0, stream>>>(
      output, low_rank_workspace, expert_first_token_offset, out_hidden_size, num_lora_adapters,
      adapter_lora_ranks, adapter_lora_weight_ptrs, permuted_lora_indices,
      adapter_expert_row_counts, adapter_expert_row_indices, adapter_expert_row_capacity,
      adapter_expert_row_starts, rows_grouped_by_adapter);
}

static __global__ void fc1GatedLoraRank16Bf16GroupedOutputKernel(
    __nv_bfloat16* output, __nv_bfloat16 const* low_rank_workspace,
    int64_t const* expert_first_token_offset, int64_t inter_size, int64_t output_stride,
    int64_t expanded_num_rows, int num_lora_adapters, int32_t const* adapter_gated_lora_ranks,
    void const* const* adapter_gated_lora_weight_ptrs, int32_t const* adapter_fc1_lora_ranks,
    void const* const* adapter_fc1_lora_weight_ptrs,
    void const* const* permuted_gated_lora_weight_ptrs,
    void const* const* permuted_fc1_lora_weight_ptrs, int32_t const* permuted_lora_indices,
    int32_t const* adapter_expert_row_counts, int32_t const* adapter_expert_row_indices,
    int adapter_expert_row_capacity, int32_t const* no_adapter_expert_row_counts,
    int32_t const* no_adapter_expert_row_indices, int no_adapter_expert_row_capacity,
    int32_t const* adapter_expert_row_starts, int32_t const* no_adapter_expert_row_starts,
    bool rows_grouped_by_adapter, int32_t const* fc1_gated_lora_has_missing_rows) {
  int64_t const out_pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const out_hidden_pairs = inter_size >> 1;
  int const adapter = static_cast<int>(blockIdx.y);
  int const expert = static_cast<int>(blockIdx.z);
  if (out_pair >= out_hidden_pairs || adapter >= num_lora_adapters) {
    return;
  }

  bool const has_row_list = adapter_expert_row_counts != nullptr &&
                            adapter_expert_row_indices != nullptr &&
                            adapter_expert_row_capacity > 0;
  bool use_row_list = false;
  int row_count = 0;
  int row_start = 0;
  int32_t const* row_indices = nullptr;
  if (has_row_list) {
    int const bucket = expert * num_lora_adapters + adapter;
    row_count = adapter_expert_row_counts[bucket];
    if (rows_grouped_by_adapter && adapter_expert_row_starts != nullptr) {
      row_start = adapter_expert_row_starts[bucket];
    } else if (row_count <= adapter_expert_row_capacity) {
      use_row_list = true;
      row_indices =
          adapter_expert_row_indices + static_cast<int64_t>(bucket) * adapter_expert_row_capacity;
    }
  }

  int32_t const raw_gated_rank = adapter_gated_lora_ranks[adapter];
  int32_t const raw_fc1_rank = adapter_fc1_lora_ranks[adapter];
  int const gated_rank = raw_gated_rank <= 0 ? 0 : (raw_gated_rank < 16 ? raw_gated_rank : 16);
  int const fc1_rank = raw_fc1_rank <= 0 ? 0 : (raw_fc1_rank < 16 ? raw_fc1_rank : 16);
  if (gated_rank <= 0 && fc1_rank <= 0) {
    return;
  }

  auto const* gated_b_base =
      reinterpret_cast<__nv_bfloat16 const*>(adapter_gated_lora_weight_ptrs[adapter * 2 + 1]);
  auto const* fc1_b_base =
      reinterpret_cast<__nv_bfloat16 const*>(adapter_fc1_lora_weight_ptrs[adapter * 2 + 1]);
  if (gated_rank > 0 && gated_b_base != nullptr) {
    gated_b_base += static_cast<size_t>(expert) * static_cast<size_t>(inter_size) * 16;
  } else {
    gated_b_base = nullptr;
  }
  if (fc1_rank > 0 && fc1_b_base != nullptr) {
    fc1_b_base += static_cast<size_t>(expert) * static_cast<size_t>(inter_size) * 16;
  } else {
    fc1_b_base = nullptr;
  }
  if (gated_b_base == nullptr && fc1_b_base == nullptr) {
    return;
  }

  int64_t const out_idx = out_pair << 1;
  __nv_bfloat162 gated_b0_pairs[8];
  __nv_bfloat162 gated_b1_pairs[8];
  __nv_bfloat162 fc1_b0_pairs[8];
  __nv_bfloat162 fc1_b1_pairs[8];
  if (gated_b_base != nullptr) {
    auto const* gated_b_row0_pairs =
        reinterpret_cast<__nv_bfloat162 const*>(gated_b_base + static_cast<size_t>(out_idx) * 16);
    auto const* gated_b_row1_pairs = reinterpret_cast<__nv_bfloat162 const*>(
        gated_b_base + static_cast<size_t>(out_idx + 1) * 16);
#pragma unroll
    for (int rank_pair = 0; rank_pair < 8; ++rank_pair) {
      gated_b0_pairs[rank_pair] = gated_b_row0_pairs[rank_pair];
      gated_b1_pairs[rank_pair] = gated_b_row1_pairs[rank_pair];
    }
  }
  if (fc1_b_base != nullptr) {
    auto const* fc1_b_row0_pairs =
        reinterpret_cast<__nv_bfloat162 const*>(fc1_b_base + static_cast<size_t>(out_idx) * 16);
    auto const* fc1_b_row1_pairs =
        reinterpret_cast<__nv_bfloat162 const*>(fc1_b_base + static_cast<size_t>(out_idx + 1) * 16);
#pragma unroll
    for (int rank_pair = 0; rank_pair < 8; ++rank_pair) {
      fc1_b0_pairs[rank_pair] = fc1_b_row0_pairs[rank_pair];
      fc1_b1_pairs[rank_pair] = fc1_b_row1_pairs[rank_pair];
    }
  }

  int64_t const begin = expert_first_token_offset[expert];
  int64_t const end = expert_first_token_offset[expert + 1];
  __nv_bfloat16 const* gated_low_rank_workspace = low_rank_workspace;
  __nv_bfloat16 const* fc1_low_rank_workspace = low_rank_workspace + expanded_num_rows * 16;

  auto zero_missing_lora_outputs = [&](int64_t row) {
    if (permuted_lora_indices != nullptr && permuted_lora_indices[row] < 0) {
      auto* gated_output_pairs = reinterpret_cast<__nv_bfloat162*>(output + row * output_stride);
      auto* fc1_output_pairs =
          reinterpret_cast<__nv_bfloat162*>(output + row * output_stride + inter_size);
      gated_output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
      fc1_output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
      return;
    }
    if (permuted_gated_lora_weight_ptrs[row * 2 + 1] == nullptr) {
      auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output + row * output_stride);
      output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
    }
    if (permuted_fc1_lora_weight_ptrs[row * 2 + 1] == nullptr) {
      auto* output_pairs =
          reinterpret_cast<__nv_bfloat162*>(output + row * output_stride + inter_size);
      output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
    }
  };

  if (adapter == 0) {
    bool const has_no_adapter_row_list =
        no_adapter_expert_row_counts != nullptr &&
        ((no_adapter_expert_row_indices != nullptr && no_adapter_expert_row_capacity > 0) ||
         (rows_grouped_by_adapter && no_adapter_expert_row_starts != nullptr)) &&
        (fc1_gated_lora_has_missing_rows == nullptr || *fc1_gated_lora_has_missing_rows == 0);
    bool used_no_adapter_row_list = false;
    if (has_no_adapter_row_list) {
      int const no_adapter_row_count = no_adapter_expert_row_counts[expert];
      if (rows_grouped_by_adapter && no_adapter_expert_row_starts != nullptr &&
          no_adapter_row_count > 0) {
        used_no_adapter_row_list = true;
        int const no_adapter_row_start = no_adapter_expert_row_starts[expert];
        for (int offset = 0; offset < no_adapter_row_count; ++offset) {
          int64_t const row = static_cast<int64_t>(no_adapter_row_start) + offset;
          auto* gated_output_pairs =
              reinterpret_cast<__nv_bfloat162*>(output + row * output_stride);
          auto* fc1_output_pairs =
              reinterpret_cast<__nv_bfloat162*>(output + row * output_stride + inter_size);
          gated_output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
          fc1_output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
        }
      } else if (no_adapter_row_count >= 0 &&
                 no_adapter_row_count <= no_adapter_expert_row_capacity) {
        used_no_adapter_row_list = true;
        auto const* no_adapter_row_indices =
            no_adapter_expert_row_indices +
            static_cast<int64_t>(expert) * no_adapter_expert_row_capacity;
        for (int offset = 0; offset < no_adapter_row_count; ++offset) {
          int64_t const row = no_adapter_row_indices[offset];
          auto* gated_output_pairs =
              reinterpret_cast<__nv_bfloat162*>(output + row * output_stride);
          auto* fc1_output_pairs =
              reinterpret_cast<__nv_bfloat162*>(output + row * output_stride + inter_size);
          gated_output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
          fc1_output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
        }
      }
    }
    if (!used_no_adapter_row_list) {
      for (int64_t row = begin; row < end; ++row) {
        zero_missing_lora_outputs(row);
      }
    }
  }
  if (has_row_list && row_count <= 0) {
    return;
  }

  auto add_row = [&](int64_t row) {
    if (gated_b_base != nullptr) {
      auto const* low_rank_pairs =
          reinterpret_cast<__nv_bfloat162 const*>(gated_low_rank_workspace + row * 16);
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      int const rank_pairs = gated_rank >> 1;
#pragma unroll
      for (int rank_pair = 0; rank_pair < 8; ++rank_pair) {
        if (rank_pair < rank_pairs) {
          float2 const low_rank_value = __bfloat1622float2(low_rank_pairs[rank_pair]);
          float2 const lora_b0_value = __bfloat1622float2(gated_b0_pairs[rank_pair]);
          float2 const lora_b1_value = __bfloat1622float2(gated_b1_pairs[rank_pair]);
          acc0 += low_rank_value.x * lora_b0_value.x;
          acc0 += low_rank_value.y * lora_b0_value.y;
          acc1 += low_rank_value.x * lora_b1_value.x;
          acc1 += low_rank_value.y * lora_b1_value.y;
        }
      }
      if ((gated_rank & 1) != 0) {
        int const rank_idx = rank_pairs << 1;
        auto const* low_rank = reinterpret_cast<__nv_bfloat16 const*>(low_rank_pairs);
        auto const* lora_b_row0 = reinterpret_cast<__nv_bfloat16 const*>(gated_b0_pairs);
        auto const* lora_b_row1 = reinterpret_cast<__nv_bfloat16 const*>(gated_b1_pairs);
        float const low_rank_value = __bfloat162float(low_rank[rank_idx]);
        acc0 += low_rank_value * __bfloat162float(lora_b_row0[rank_idx]);
        acc1 += low_rank_value * __bfloat162float(lora_b_row1[rank_idx]);
      }
      auto* output_pairs = reinterpret_cast<__nv_bfloat162*>(output + row * output_stride);
      output_pairs[out_pair] = __floats2bfloat162_rn(acc0, acc1);
    }

    if (fc1_b_base != nullptr) {
      auto const* low_rank_pairs =
          reinterpret_cast<__nv_bfloat162 const*>(fc1_low_rank_workspace + row * 16);
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      int const rank_pairs = fc1_rank >> 1;
#pragma unroll
      for (int rank_pair = 0; rank_pair < 8; ++rank_pair) {
        if (rank_pair < rank_pairs) {
          float2 const low_rank_value = __bfloat1622float2(low_rank_pairs[rank_pair]);
          float2 const lora_b0_value = __bfloat1622float2(fc1_b0_pairs[rank_pair]);
          float2 const lora_b1_value = __bfloat1622float2(fc1_b1_pairs[rank_pair]);
          acc0 += low_rank_value.x * lora_b0_value.x;
          acc0 += low_rank_value.y * lora_b0_value.y;
          acc1 += low_rank_value.x * lora_b1_value.x;
          acc1 += low_rank_value.y * lora_b1_value.y;
        }
      }
      if ((fc1_rank & 1) != 0) {
        int const rank_idx = rank_pairs << 1;
        auto const* low_rank = reinterpret_cast<__nv_bfloat16 const*>(low_rank_pairs);
        auto const* lora_b_row0 = reinterpret_cast<__nv_bfloat16 const*>(fc1_b0_pairs);
        auto const* lora_b_row1 = reinterpret_cast<__nv_bfloat16 const*>(fc1_b1_pairs);
        float const low_rank_value = __bfloat162float(low_rank[rank_idx]);
        acc0 += low_rank_value * __bfloat162float(lora_b_row0[rank_idx]);
        acc1 += low_rank_value * __bfloat162float(lora_b_row1[rank_idx]);
      }
      auto* output_pairs =
          reinterpret_cast<__nv_bfloat162*>(output + row * output_stride + inter_size);
      output_pairs[out_pair] = __floats2bfloat162_rn(acc0, acc1);
    }
  };

  if (use_row_list) {
    for (int offset = 0; offset < row_count; ++offset) {
      add_row(row_indices[offset]);
    }
  } else if (rows_grouped_by_adapter && adapter_expert_row_starts != nullptr) {
    if (row_count <= 0 || row_start < 0) {
      return;
    }
    for (int offset = 0; offset < row_count; ++offset) {
      add_row(static_cast<int64_t>(row_start) + offset);
    }
  } else {
    for (int64_t row = begin; row < end; ++row) {
      if (permuted_lora_indices[row] == adapter) {
        add_row(row);
      }
    }
  }
}

inline void fc1GatedLoraRank16Bf16GroupedOutput(
    __nv_bfloat16* output, __nv_bfloat16 const* low_rank_workspace,
    int64_t const* expert_first_token_offset, int64_t expanded_num_rows, int64_t inter_size,
    int num_experts_per_node, int num_lora_adapters, int32_t const* adapter_gated_lora_ranks,
    void const* const* adapter_gated_lora_weight_ptrs, int32_t const* adapter_fc1_lora_ranks,
    void const* const* adapter_fc1_lora_weight_ptrs,
    void const* const* permuted_gated_lora_weight_ptrs,
    void const* const* permuted_fc1_lora_weight_ptrs, int32_t const* permuted_lora_indices,
    int32_t const* adapter_expert_row_counts, int32_t const* adapter_expert_row_indices,
    int adapter_expert_row_capacity, int32_t const* no_adapter_expert_row_counts,
    int32_t const* no_adapter_expert_row_indices, int no_adapter_expert_row_capacity,
    int32_t const* adapter_expert_row_starts, int32_t const* no_adapter_expert_row_starts,
    bool rows_grouped_by_adapter, int32_t const* fc1_gated_lora_has_missing_rows,
    cudaStream_t stream) {
  constexpr int threads = 128;
  dim3 const grid(static_cast<unsigned int>(((inter_size >> 1) + threads - 1) / threads),
                  static_cast<unsigned int>(num_lora_adapters),
                  static_cast<unsigned int>(num_experts_per_node));
  fc1GatedLoraRank16Bf16GroupedOutputKernel<<<grid, threads, 0, stream>>>(
      output, low_rank_workspace, expert_first_token_offset, inter_size, inter_size * 2,
      expanded_num_rows, num_lora_adapters, adapter_gated_lora_ranks,
      adapter_gated_lora_weight_ptrs, adapter_fc1_lora_ranks, adapter_fc1_lora_weight_ptrs,
      permuted_gated_lora_weight_ptrs, permuted_fc1_lora_weight_ptrs, permuted_lora_indices,
      adapter_expert_row_counts, adapter_expert_row_indices, adapter_expert_row_capacity,
      no_adapter_expert_row_counts, no_adapter_expert_row_indices, no_adapter_expert_row_capacity,
      adapter_expert_row_starts, no_adapter_expert_row_starts, rows_grouped_by_adapter,
      fc1_gated_lora_has_missing_rows);
}

using namespace nvcuda;

__global__ void loraLowRank16Bf16TensorCoreGroupedKernel(
    __nv_bfloat16 const* input, int64_t input_hidden_size, int64_t expanded_num_rows,
    int64_t num_tokens, int64_t lora_a_expert_stride, int num_lora_adapters,
    int num_experts_per_node, int num_lora_modules, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int32_t const* adapter_expert_row_counts,
    int32_t const* adapter_expert_row_starts, __nv_bfloat16* low_rank_workspace) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  constexpr int kRowsPerWarp = 16;
  constexpr int kWarpsPerBlock = 1;
  constexpr int kRowsPerBlock = kRowsPerWarp * kWarpsPerBlock;
  int const row_block = static_cast<int>(blockIdx.x);
  int const adapter = static_cast<int>(blockIdx.y);
  int const module_expert = static_cast<int>(blockIdx.z);
  int const module = module_expert / num_experts_per_node;
  int const expert = module_expert - module * num_experts_per_node;
  int const bucket = expert * num_lora_adapters + adapter;
  int const warp_id = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;

  int32_t const row_count = adapter_expert_row_counts[bucket];
  int32_t const row_start = adapter_expert_row_starts[bucket];
  if (row_count <= 0 || row_start < 0) {
    return;
  }

  int32_t const raw_rank = adapter_lora_ranks[adapter];
  int const lora_rank = raw_rank <= 0 ? 0 : (raw_rank < 16 ? raw_rank : 16);
  auto const* lora_a_base =
      reinterpret_cast<__nv_bfloat16 const*>(adapter_lora_weight_ptrs[adapter * 2]);
  if (lora_rank <= 0 || lora_a_base == nullptr) {
    return;
  }
  bool const full_rank = lora_rank == 16;
  lora_a_base += static_cast<size_t>(expert) * static_cast<size_t>(lora_a_expert_stride);
  bool const use_direct_b = full_rank;

  __shared__ __nv_bfloat16 a_smem[kWarpsPerBlock][16 * 16];
  __shared__ __nv_bfloat16 b_smem[2][16 * 16];
  __shared__ float c_smem[kWarpsPerBlock][16 * 16];
  for (int tile_idx = row_block; tile_idx * kRowsPerBlock < row_count; tile_idx += gridDim.x) {
    int const tile_row = tile_idx * kRowsPerBlock;
    int const block_valid_rows = min(kRowsPerBlock, row_count - tile_row);
    int const warp_row_offset = warp_id * kRowsPerWarp;
    int const valid_rows = max(0, min(kRowsPerWarp, block_valid_rows - warp_row_offset));
    int64_t const row = static_cast<int64_t>(row_start) + tile_row + warp_row_offset;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    for (int64_t k = 0; k < input_hidden_size; k += 16) {
      if (use_direct_b) {
        for (int idx = lane; idx < 16 * 16; idx += 32) {
          int const local_row = idx >> 4;
          int const local_col = idx & 15;
          a_smem[warp_id][idx] = local_row < valid_rows
                                     ? input[(row + local_row) * input_hidden_size + k + local_col]
                                     : __float2bfloat16(0.0f);
        }
        __syncwarp();
        wmma::load_matrix_sync(a_frag, a_smem[warp_id], 16);
        wmma::load_matrix_sync(b_frag, lora_a_base + k, static_cast<int>(input_hidden_size));
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      } else {
        int const stage = (static_cast<int>(k >> 4) & 1);
        for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
          int const local_k = idx & 15;
          int const rank = idx >> 4;
          b_smem[stage][idx] =
              rank < lora_rank
                  ? lora_a_base[k + local_k + static_cast<int64_t>(rank) * input_hidden_size]
                  : __float2bfloat16(0.0f);
        }
        for (int idx = lane; idx < 16 * 16; idx += 32) {
          int const local_row = idx >> 4;
          int const local_col = idx & 15;
          a_smem[warp_id][idx] = local_row < valid_rows
                                     ? input[(row + local_row) * input_hidden_size + k + local_col]
                                     : __float2bfloat16(0.0f);
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag, a_smem[warp_id], 16);
        wmma::load_matrix_sync(b_frag, b_smem[stage], 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }

    wmma::store_matrix_sync(c_smem[warp_id], c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < valid_rows * 16; idx += 32) {
      int const local_row = idx / 16;
      int const rank = idx - local_row * 16;
      size_t const out_offset =
          (static_cast<size_t>(module) * static_cast<size_t>(expanded_num_rows) +
           static_cast<size_t>(row + local_row)) *
              16 +
          static_cast<size_t>(rank);
      low_rank_workspace[out_offset] =
          (full_rank || rank < lora_rank) ? __float2bfloat16(c_smem[warp_id][local_row * 16 + rank])
                                          : __float2bfloat16(0.0f);
    }
  }
#endif
}

inline void loraLowRank16Bf16TensorCoreGrouped(
    __nv_bfloat16 const* input, int64_t input_hidden_size, int64_t expanded_num_rows,
    int64_t num_tokens, int64_t lora_a_expert_stride, int num_lora_adapters,
    int num_experts_per_node, int num_lora_modules, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int32_t const* adapter_expert_row_counts,
    int32_t const* adapter_expert_row_starts, __nv_bfloat16* low_rank_workspace,
    cudaStream_t stream) {
  constexpr int threads = 32;
  constexpr int rows_per_block = 16;
  int64_t const avg_rows_per_adapter = std::max<int64_t>(
      1, (expanded_num_rows + num_experts_per_node * (num_lora_adapters + 1) - 1) /
             (num_experts_per_node * (num_lora_adapters + 1)));
  int max_row_blocks_per_bucket = static_cast<int>(std::min<int64_t>(
      64, std::max<int64_t>(1, (avg_rows_per_adapter + rows_per_block - 1) / rows_per_block)));
  if (input_hidden_size == 2048 && expanded_num_rows >= 262144 && num_experts_per_node == 256 &&
      num_lora_adapters <= 4 && num_lora_modules == 1) {
    max_row_blocks_per_bucket = 96;
  }
  if (char const* raw = std::getenv("FLASHINFER_CUTLASS_MOE_LORA_LOW_RANK_ROW_BLOCKS")) {
    int const requested = std::atoi(raw);
    if (requested > 0) {
      max_row_blocks_per_bucket = std::min(128, requested);
    }
  }
  dim3 const grid(static_cast<unsigned int>(max_row_blocks_per_bucket),
                  static_cast<unsigned int>(num_lora_adapters),
                  static_cast<unsigned int>(num_lora_modules * num_experts_per_node));
  loraLowRank16Bf16TensorCoreGroupedKernel<<<grid, threads, 0, stream>>>(
      input, input_hidden_size, expanded_num_rows, num_tokens, lora_a_expert_stride,
      num_lora_adapters, num_experts_per_node, num_lora_modules, adapter_lora_ranks,
      adapter_lora_weight_ptrs, adapter_expert_row_counts, adapter_expert_row_starts,
      low_rank_workspace);
}

template <bool UseOriginalInput>
__global__ void loraLowRank16Bf16TensorCoreGroupedDualKernel(
    __nv_bfloat16 const* input, int64_t input_hidden_size, int64_t expanded_num_rows,
    int64_t num_tokens, int64_t lora_a_expert_stride, int num_lora_adapters,
    int num_experts_per_node, int32_t const* adapter_lora_ranks0,
    void const* const* adapter_lora_weight_ptrs0, int32_t const* adapter_lora_ranks1,
    void const* const* adapter_lora_weight_ptrs1, int32_t const* adapter_expert_row_counts,
    int32_t const* adapter_expert_row_starts, __nv_bfloat16* low_rank_workspace0,
    __nv_bfloat16* low_rank_workspace1, int const* input_row_to_unpermuted_row,
    int64_t original_num_rows, bool force_shared_lora_a) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  constexpr int kRowsPerWarp = 16;
  constexpr int kWarpsPerBlock = 1;
  constexpr int kRowsPerBlock = kRowsPerWarp * kWarpsPerBlock;
  int const row_block = static_cast<int>(blockIdx.x);
  int const adapter = static_cast<int>(blockIdx.y);
  int const expert = static_cast<int>(blockIdx.z);
  int const bucket = expert * num_lora_adapters + adapter;
  int const warp_id = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;

  int32_t const row_count = adapter_expert_row_counts[bucket];
  int32_t const row_start = adapter_expert_row_starts[bucket];
  if (row_count <= 0 || row_start < 0) {
    return;
  }

  int32_t const raw_rank0 = adapter_lora_ranks0[adapter];
  int32_t const raw_rank1 = adapter_lora_ranks1[adapter];
  int const lora_rank0 = raw_rank0 <= 0 ? 0 : (raw_rank0 < 16 ? raw_rank0 : 16);
  int const lora_rank1 = raw_rank1 <= 0 ? 0 : (raw_rank1 < 16 ? raw_rank1 : 16);
  bool const full_rank0 = lora_rank0 == 16;
  bool const full_rank1 = lora_rank1 == 16;
  auto const* lora_a_base0 =
      reinterpret_cast<__nv_bfloat16 const*>(adapter_lora_weight_ptrs0[adapter * 2]);
  auto const* lora_a_base1 =
      reinterpret_cast<__nv_bfloat16 const*>(adapter_lora_weight_ptrs1[adapter * 2]);
  bool const use0 = lora_rank0 > 0 && lora_a_base0 != nullptr;
  bool const use1 = lora_rank1 > 0 && lora_a_base1 != nullptr;
  if (!use0 && !use1) {
    return;
  }
  if (use0) {
    lora_a_base0 += static_cast<size_t>(expert) * static_cast<size_t>(lora_a_expert_stride);
  }
  if (use1) {
    lora_a_base1 += static_cast<size_t>(expert) * static_cast<size_t>(lora_a_expert_stride);
  }
  bool const shared_lora_a =
      use0 && use1 && lora_rank0 == lora_rank1 && lora_a_base0 == lora_a_base1;
  bool const use_direct_b =
      (!force_shared_lora_a) && (!use0 || full_rank0) && (!use1 || full_rank1);
  __shared__ __nv_bfloat16 a_smem[kWarpsPerBlock][16 * 16];
  __shared__ __nv_bfloat16 b_smem0[2][16 * 16];
  __shared__ __nv_bfloat16 b_smem1[2][16 * 16];
  __shared__ float c_smem0[kWarpsPerBlock][16 * 16];
  __shared__ float c_smem1[kWarpsPerBlock][16 * 16];

  for (int tile_idx = row_block; tile_idx * kRowsPerBlock < row_count; tile_idx += gridDim.x) {
    int const tile_row = tile_idx * kRowsPerBlock;
    int const block_valid_rows = min(kRowsPerBlock, row_count - tile_row);
    int const warp_row_offset = warp_id * kRowsPerWarp;
    int const valid_rows = max(0, min(kRowsPerWarp, block_valid_rows - warp_row_offset));
    int64_t const row = static_cast<int64_t>(row_start) + tile_row + warp_row_offset;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag0;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag1;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag0;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag1;
    wmma::fill_fragment(c_frag0, 0.0f);
    wmma::fill_fragment(c_frag1, 0.0f);

    for (int64_t k = 0; k < input_hidden_size; k += 16) {
      if (use_direct_b) {
        for (int idx = lane; idx < 16 * 16; idx += 32) {
          int const local_row = idx >> 4;
          int const local_col = idx & 15;
          if (local_row < valid_rows) {
            int64_t input_row = row + local_row;
            if constexpr (UseOriginalInput) {
              input_row =
                  static_cast<int64_t>(input_row_to_unpermuted_row[input_row]) % original_num_rows;
            }
            a_smem[warp_id][idx] = input[input_row * input_hidden_size + k + local_col];
          } else {
            a_smem[warp_id][idx] = __float2bfloat16(0.0f);
          }
        }
        __syncwarp();
        wmma::load_matrix_sync(a_frag, a_smem[warp_id], 16);
        if (use0) {
          wmma::load_matrix_sync(b_frag0, lora_a_base0 + k, static_cast<int>(input_hidden_size));
          wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
        }
        if (use1 && !shared_lora_a) {
          wmma::load_matrix_sync(b_frag1, lora_a_base1 + k, static_cast<int>(input_hidden_size));
          wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
        }
      } else {
        int const stage = (static_cast<int>(k >> 4) & 1);
        for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
          int const local_k = idx & 15;
          int const rank = idx >> 4;
          if (use0) {
            if (full_rank0) {
              b_smem0[stage][idx] =
                  lora_a_base0[k + local_k + static_cast<int64_t>(rank) * input_hidden_size];
            } else {
              b_smem0[stage][idx] =
                  rank < lora_rank0
                      ? lora_a_base0[k + local_k + static_cast<int64_t>(rank) * input_hidden_size]
                      : __float2bfloat16(0.0f);
            }
          }
          if (use1 && !shared_lora_a) {
            if (full_rank1) {
              b_smem1[stage][idx] =
                  lora_a_base1[k + local_k + static_cast<int64_t>(rank) * input_hidden_size];
            } else {
              b_smem1[stage][idx] =
                  rank < lora_rank1
                      ? lora_a_base1[k + local_k + static_cast<int64_t>(rank) * input_hidden_size]
                      : __float2bfloat16(0.0f);
            }
          }
        }
        for (int idx = lane; idx < 16 * 16; idx += 32) {
          int const local_row = idx >> 4;
          int const local_col = idx & 15;
          if (local_row < valid_rows) {
            int64_t input_row = row + local_row;
            if constexpr (UseOriginalInput) {
              input_row =
                  static_cast<int64_t>(input_row_to_unpermuted_row[input_row]) % original_num_rows;
            }
            a_smem[warp_id][idx] = input[input_row * input_hidden_size + k + local_col];
          } else {
            a_smem[warp_id][idx] = __float2bfloat16(0.0f);
          }
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag, a_smem[warp_id], 16);
        if (use0) {
          wmma::load_matrix_sync(b_frag0, b_smem0[stage], 16);
          wmma::mma_sync(c_frag0, a_frag, b_frag0, c_frag0);
        }
        if (use1 && !shared_lora_a) {
          wmma::load_matrix_sync(b_frag1, b_smem1[stage], 16);
          wmma::mma_sync(c_frag1, a_frag, b_frag1, c_frag1);
        }
      }
    }

    if (use0) {
      wmma::store_matrix_sync(c_smem0[warp_id], c_frag0, 16, wmma::mem_row_major);
    }
    if (use1 && !shared_lora_a) {
      wmma::store_matrix_sync(c_smem1[warp_id], c_frag1, 16, wmma::mem_row_major);
    }
    __syncwarp();

    for (int idx = lane; idx < valid_rows * 16; idx += 32) {
      int const local_row = idx / 16;
      int const rank = idx - local_row * 16;
      size_t const out_offset = (static_cast<size_t>(row + local_row) * 16) + rank;
      if (use0) {
        low_rank_workspace0[out_offset] =
            (full_rank0 || rank < lora_rank0)
                ? __float2bfloat16(c_smem0[warp_id][local_row * 16 + rank])
                : __float2bfloat16(0.0f);
      }
      if (use1) {
        low_rank_workspace1[out_offset] =
            (full_rank1 || rank < lora_rank1)
                ? __float2bfloat16((shared_lora_a ? c_smem0 : c_smem1)[warp_id]
                                                               [local_row * 16 + rank])
                : __float2bfloat16(0.0f);
      }
    }
  }
#endif
}

inline void loraLowRank16Bf16TensorCoreGroupedDual(
    __nv_bfloat16 const* input, int64_t input_hidden_size, int64_t expanded_num_rows,
    int64_t num_tokens, int64_t lora_a_expert_stride, int num_lora_adapters,
    int num_experts_per_node, int32_t const* adapter_lora_ranks0,
    void const* const* adapter_lora_weight_ptrs0, int32_t const* adapter_lora_ranks1,
    void const* const* adapter_lora_weight_ptrs1, int32_t const* adapter_expert_row_counts,
    int32_t const* adapter_expert_row_starts, __nv_bfloat16* low_rank_workspace0,
    __nv_bfloat16* low_rank_workspace1, cudaStream_t stream,
    int const* input_row_to_unpermuted_row = nullptr, int64_t original_num_rows = 0) {
  constexpr int threads = 32;
  constexpr int rows_per_block = 16;
  int64_t const avg_rows_per_adapter = std::max<int64_t>(
      1, (expanded_num_rows + num_experts_per_node * (num_lora_adapters + 1) - 1) /
             (num_experts_per_node * (num_lora_adapters + 1)));
  int max_row_blocks_per_bucket = static_cast<int>(std::min<int64_t>(
      64, std::max<int64_t>(1, (avg_rows_per_adapter + rows_per_block - 1) / rows_per_block)));
  if (input_hidden_size == 6144 && expanded_num_rows >= 262144 && num_experts_per_node == 256 &&
      num_lora_adapters <= 4) {
    max_row_blocks_per_bucket = 96;
  }
  if (char const* raw =
          std::getenv("FLASHINFER_CUTLASS_MOE_LORA_DUAL_LOW_RANK_ROW_BLOCKS")) {
    int const requested = std::atoi(raw);
    if (requested > 0) {
      max_row_blocks_per_bucket = std::min(128, requested);
    }
  }
  dim3 const grid(static_cast<unsigned int>(max_row_blocks_per_bucket),
                  static_cast<unsigned int>(num_lora_adapters),
                  static_cast<unsigned int>(num_experts_per_node));
  bool const force_shared_lora_a = false;
  if (input_row_to_unpermuted_row != nullptr && original_num_rows > 0) {
    loraLowRank16Bf16TensorCoreGroupedDualKernel<true><<<grid, threads, 0, stream>>>(
        input, input_hidden_size, expanded_num_rows, num_tokens, lora_a_expert_stride,
        num_lora_adapters, num_experts_per_node, adapter_lora_ranks0, adapter_lora_weight_ptrs0,
        adapter_lora_ranks1, adapter_lora_weight_ptrs1, adapter_expert_row_counts,
        adapter_expert_row_starts, low_rank_workspace0, low_rank_workspace1,
        input_row_to_unpermuted_row, original_num_rows, force_shared_lora_a);
  } else {
    loraLowRank16Bf16TensorCoreGroupedDualKernel<false><<<grid, threads, 0, stream>>>(
        input, input_hidden_size, expanded_num_rows, num_tokens, lora_a_expert_stride,
        num_lora_adapters, num_experts_per_node, adapter_lora_ranks0, adapter_lora_weight_ptrs0,
        adapter_lora_ranks1, adapter_lora_weight_ptrs1, adapter_expert_row_counts,
        adapter_expert_row_starts, low_rank_workspace0, low_rank_workspace1, nullptr, 0,
        force_shared_lora_a);
  }
}

template <bool AddToOutput, int kWarpsPerBlock, int kNTilesPerWarp>
__global__ void loraOutput16Bf16TensorCoreGroupedKernel(
    __nv_bfloat16* output, __nv_bfloat16 const* low_rank_workspace, int64_t out_hidden_size,
    int64_t output_stride, int num_lora_adapters, int num_experts_per_node,
    int32_t const* adapter_lora_ranks, void const* const* adapter_lora_weight_ptrs,
    int32_t const* adapter_expert_row_counts, int32_t const* adapter_expert_row_starts,
    int n_tile_start, int n_tile_end, int row_split_groups, int row_split_rows) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  int const warp_id = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  int const row_group_count = max(1, row_split_groups);
  int const linear_tile_group = static_cast<int>(blockIdx.x);
  int const n_tile_group = linear_tile_group / row_group_count;
  int const row_group = linear_tile_group - n_tile_group * row_group_count;
  int const n_tile_base =
      n_tile_start + (n_tile_group * kWarpsPerBlock + warp_id) * kNTilesPerWarp;
  int const adapter = static_cast<int>(blockIdx.y);
  int const expert = static_cast<int>(blockIdx.z);
  int const out_tiles = static_cast<int>(out_hidden_size >> 4);
  int const bucket = expert * num_lora_adapters + adapter;
  int32_t const row_count = adapter_expert_row_counts[bucket];
  int32_t const row_start = adapter_expert_row_starts[bucket];
  if (row_count <= 0 || row_start < 0) {
    return;
  }

  int32_t const raw_rank = adapter_lora_ranks[adapter];
  int const lora_rank = raw_rank <= 0 ? 0 : (raw_rank < 16 ? raw_rank : 16);
  bool const full_rank = lora_rank == 16;
  auto const* lora_b_base =
      reinterpret_cast<__nv_bfloat16 const*>(adapter_lora_weight_ptrs[adapter * 2 + 1]);
  bool const has_lora = lora_rank > 0 && lora_b_base != nullptr;
  if (has_lora) {
    lora_b_base += static_cast<size_t>(expert) * static_cast<size_t>(out_hidden_size) * 16;
  } else if constexpr (AddToOutput) {
    return;
  }

  __shared__ __nv_bfloat16 a_smem[16 * 16];
  __shared__ float c_smem[kWarpsPerBlock][16 * 16];

  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>
      b_frags[kNTilesPerWarp];
  bool valid_n_tiles[kNTilesPerWarp];
  bool has_valid_lora_tiles[kNTilesPerWarp];
  bool do_mma = false;
#pragma unroll
  for (int tile_offset = 0; tile_offset < kNTilesPerWarp; ++tile_offset) {
    int const n_tile = n_tile_base + tile_offset;
    bool const valid_n_tile = n_tile < out_tiles && n_tile < n_tile_end;
    bool const has_valid_lora_tile = has_lora && valid_n_tile;
    valid_n_tiles[tile_offset] = valid_n_tile;
    has_valid_lora_tiles[tile_offset] = has_valid_lora_tile;
    do_mma = do_mma || has_valid_lora_tile;
    if (has_valid_lora_tile) {
      wmma::load_matrix_sync(b_frags[tile_offset],
                             lora_b_base + static_cast<size_t>(n_tile) * 16 * 16, 16);
    }
  }
  int const row_span = row_split_rows > 0 ? row_split_rows : row_count;
  int const first_row = row_group * row_span;
  int const row_stride = row_group_count * row_span;
  for (int row_block = first_row; row_block < row_count; row_block += row_stride) {
    int const row_block_end = min(row_count, row_block + row_span);
    for (int tile_row = row_block; tile_row < row_block_end; tile_row += 16) {
      int const valid_rows = min(16, row_count - tile_row);
      int64_t const row = static_cast<int64_t>(row_start) + tile_row;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
      if (do_mma) {
        bool const direct_full_rank_tile = full_rank && valid_rows == 16;
        if (direct_full_rank_tile) {
          wmma::load_matrix_sync(a_frag, low_rank_workspace + row * 16, 16);
        } else {
          for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
            int const local_row = idx >> 4;
            int const rank = idx & 15;
            bool const valid_row = local_row < valid_rows;
            if (has_lora && full_rank) {
              a_smem[idx] = valid_row ? low_rank_workspace[(row + local_row) * 16 + rank]
                                      : __float2bfloat16(0.0f);
            } else {
              a_smem[idx] = (valid_row && rank < lora_rank)
                                ? low_rank_workspace[(row + local_row) * 16 + rank]
                                : __float2bfloat16(0.0f);
            }
          }
          __syncthreads();
          wmma::load_matrix_sync(a_frag, a_smem, 16);
        }
      }

#pragma unroll
      for (int tile_offset = 0; tile_offset < kNTilesPerWarp; ++tile_offset) {
        int const n_tile = n_tile_base + tile_offset;
        bool const valid_n_tile = valid_n_tiles[tile_offset];
        bool const has_valid_lora_tile = has_valid_lora_tiles[tile_offset];
        if (has_valid_lora_tile) {
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
          wmma::fill_fragment(c_frag, 0.0f);
          wmma::mma_sync(c_frag, a_frag, b_frags[tile_offset], c_frag);
          wmma::store_matrix_sync(c_smem[warp_id], c_frag, 16, wmma::mem_row_major);
        }
        __syncwarp();
        if (!valid_n_tile) {
          continue;
        }
        for (int idx = lane; idx < valid_rows * 8; idx += 32) {
          int const local_row = idx >> 3;
          int const local_pair = idx & 7;
          int const c_offset = local_row * 16 + local_pair * 2;
          auto* out_pairs = reinterpret_cast<__nv_bfloat162*>(
              output + (row + local_row) * output_stride + static_cast<int64_t>(n_tile) * 16);
          if (has_valid_lora_tile) {
            float value0 = c_smem[warp_id][c_offset];
            float value1 = c_smem[warp_id][c_offset + 1];
            if constexpr (AddToOutput) {
              float2 prev = __bfloat1622float2(out_pairs[local_pair]);
              value0 += prev.x;
              value1 += prev.y;
            }
            out_pairs[local_pair] = __floats2bfloat162_rn(value0, value1);
          } else {
            out_pairs[local_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
          }
        }
        __syncwarp();
      }
    }
  }
#endif
}

template <bool AddToOutput, int kWarpsPerBlock, int kNTilesPerWarp>
inline void launchLoraOutput16Bf16TensorCoreGrouped(
    __nv_bfloat16* output, __nv_bfloat16 const* low_rank_workspace, int64_t out_hidden_size,
    int64_t output_stride, int num_lora_adapters, int num_experts_per_node,
    int32_t const* adapter_lora_ranks, void const* const* adapter_lora_weight_ptrs,
    int32_t const* adapter_expert_row_counts, int32_t const* adapter_expert_row_starts,
    cudaStream_t stream, int output_split_count = 1, int output_split_index = 0,
    int row_split_groups = 1, int row_split_rows = 0) {
  constexpr int threads = 32 * kWarpsPerBlock;
  int const out_tiles = static_cast<int>(out_hidden_size >> 4);
  int const split_count = std::max(1, output_split_count);
  int const split_index = std::max(0, std::min(output_split_index, split_count - 1));
  int const n_tile_start =
      static_cast<int>((static_cast<int64_t>(out_tiles) * split_index) / split_count);
  int const n_tile_end =
      static_cast<int>((static_cast<int64_t>(out_tiles) * (split_index + 1)) / split_count);
  if (n_tile_end <= n_tile_start) {
    return;
  }
  int const n_tile_groups =
      static_cast<int>(((n_tile_end - n_tile_start) + kWarpsPerBlock * kNTilesPerWarp - 1) /
                       (kWarpsPerBlock * kNTilesPerWarp));
  int const safe_row_split_groups = std::max(1, std::min(row_split_groups, 64));
  int const safe_row_split_rows = std::max(0, row_split_rows);
  dim3 const grid(static_cast<unsigned int>(n_tile_groups * safe_row_split_groups),
                  static_cast<unsigned int>(num_lora_adapters),
                  static_cast<unsigned int>(num_experts_per_node));
  loraOutput16Bf16TensorCoreGroupedKernel<AddToOutput, kWarpsPerBlock, kNTilesPerWarp>
      <<<grid, threads, 0, stream>>>(output, low_rank_workspace, out_hidden_size, output_stride,
                                     num_lora_adapters, num_experts_per_node, adapter_lora_ranks,
                                     adapter_lora_weight_ptrs, adapter_expert_row_counts,
                                     adapter_expert_row_starts, n_tile_start, n_tile_end,
                                     safe_row_split_groups, safe_row_split_rows);
}

template <bool AddToOutput>
inline void loraOutput16Bf16TensorCoreGrouped(
    __nv_bfloat16* output, __nv_bfloat16 const* low_rank_workspace, int64_t out_hidden_size,
    int64_t output_stride, int64_t expanded_num_rows, int num_lora_adapters,
    int num_experts_per_node, int32_t const* adapter_lora_ranks,
    void const* const* adapter_lora_weight_ptrs, int32_t const* adapter_expert_row_counts,
    int32_t const* adapter_expert_row_starts, cudaStream_t stream, int output_split_count = 1,
    int output_split_index = 0) {
  if (out_hidden_size == 6144 && output_stride == 6144 && expanded_num_rows >= 262144 &&
      num_experts_per_node == 256) {
    int row_split_groups = 1;
    int row_split_rows = 0;
    if (char const* raw = std::getenv("FLASHINFER_CUTLASS_MOE_LORA_OUTPUT_ROW_SPLIT_GROUPS")) {
      row_split_groups = std::max(1, std::atoi(raw));
    }
    if (char const* raw = std::getenv("FLASHINFER_CUTLASS_MOE_LORA_OUTPUT_ROW_SPLIT_ROWS")) {
      row_split_rows = std::max(0, std::atoi(raw));
    }
    if (row_split_groups > 1 && row_split_rows == 0) {
      row_split_rows = 128;
    }
    char const* raw_tile_shape = std::getenv("FLASHINFER_CUTLASS_MOE_LORA_OUTPUT_TILE");
    if (raw_tile_shape == nullptr) {
      raw_tile_shape = "8x2";
    }
    if (std::strcmp(raw_tile_shape, "2x4") == 0) {
      launchLoraOutput16Bf16TensorCoreGrouped<AddToOutput, 2, 4>(
          output, low_rank_workspace, out_hidden_size, output_stride, num_lora_adapters,
          num_experts_per_node, adapter_lora_ranks, adapter_lora_weight_ptrs,
          adapter_expert_row_counts, adapter_expert_row_starts, stream, output_split_count,
          output_split_index, row_split_groups, row_split_rows);
    } else if (std::strcmp(raw_tile_shape, "4x4") == 0) {
      launchLoraOutput16Bf16TensorCoreGrouped<AddToOutput, 4, 4>(
          output, low_rank_workspace, out_hidden_size, output_stride, num_lora_adapters,
          num_experts_per_node, adapter_lora_ranks, adapter_lora_weight_ptrs,
          adapter_expert_row_counts, adapter_expert_row_starts, stream, output_split_count,
          output_split_index, row_split_groups, row_split_rows);
    } else if (std::strcmp(raw_tile_shape, "4x6") == 0) {
      launchLoraOutput16Bf16TensorCoreGrouped<AddToOutput, 4, 6>(
          output, low_rank_workspace, out_hidden_size, output_stride, num_lora_adapters,
          num_experts_per_node, adapter_lora_ranks, adapter_lora_weight_ptrs,
          adapter_expert_row_counts, adapter_expert_row_starts, stream, output_split_count,
          output_split_index, row_split_groups, row_split_rows);
    } else if (std::strcmp(raw_tile_shape, "4x8") == 0) {
      launchLoraOutput16Bf16TensorCoreGrouped<AddToOutput, 4, 8>(
          output, low_rank_workspace, out_hidden_size, output_stride, num_lora_adapters,
          num_experts_per_node, adapter_lora_ranks, adapter_lora_weight_ptrs,
          adapter_expert_row_counts, adapter_expert_row_starts, stream, output_split_count,
          output_split_index, row_split_groups, row_split_rows);
    } else if (std::strcmp(raw_tile_shape, "8x2") == 0) {
      launchLoraOutput16Bf16TensorCoreGrouped<AddToOutput, 8, 2>(
          output, low_rank_workspace, out_hidden_size, output_stride, num_lora_adapters,
          num_experts_per_node, adapter_lora_ranks, adapter_lora_weight_ptrs,
          adapter_expert_row_counts, adapter_expert_row_starts, stream, output_split_count,
          output_split_index, row_split_groups, row_split_rows);
    } else if (std::strcmp(raw_tile_shape, "8x4") == 0) {
      launchLoraOutput16Bf16TensorCoreGrouped<AddToOutput, 8, 4>(
          output, low_rank_workspace, out_hidden_size, output_stride, num_lora_adapters,
          num_experts_per_node, adapter_lora_ranks, adapter_lora_weight_ptrs,
          adapter_expert_row_counts, adapter_expert_row_starts, stream, output_split_count,
          output_split_index, row_split_groups, row_split_rows);
    } else {
      launchLoraOutput16Bf16TensorCoreGrouped<AddToOutput, 6, 4>(
          output, low_rank_workspace, out_hidden_size, output_stride, num_lora_adapters,
          num_experts_per_node, adapter_lora_ranks, adapter_lora_weight_ptrs,
          adapter_expert_row_counts, adapter_expert_row_starts, stream, output_split_count,
          output_split_index, row_split_groups, row_split_rows);
    }
    return;
  }
  launchLoraOutput16Bf16TensorCoreGrouped<AddToOutput, 4, 4>(
      output, low_rank_workspace, out_hidden_size, output_stride, num_lora_adapters,
      num_experts_per_node, adapter_lora_ranks, adapter_lora_weight_ptrs, adapter_expert_row_counts,
      adapter_expert_row_starts, stream, output_split_count, output_split_index);
}

template <int kWarpsPerBlock, int kNTilesPerWarp>
__global__ void loraOutput16Bf16TensorCoreGroupedDualKernel(
    __nv_bfloat16* output0, __nv_bfloat16* output1, __nv_bfloat16 const* low_rank_workspace0,
    __nv_bfloat16 const* low_rank_workspace1, int64_t out_hidden_size, int64_t output_stride,
    int num_lora_adapters, int num_experts_per_node, int32_t const* adapter_lora_ranks0,
    void const* const* adapter_lora_weight_ptrs0, int32_t const* adapter_lora_ranks1,
    void const* const* adapter_lora_weight_ptrs1, int32_t const* adapter_expert_row_counts,
    int32_t const* adapter_expert_row_starts, int32_t const* no_adapter_expert_row_counts,
    int32_t const* no_adapter_expert_row_starts, bool zero_no_adapter_rows,
    int row_split_groups, int row_split_rows) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
  int const warp_id = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  int const row_group_count = max(1, row_split_groups);
  int const linear_tile_group = static_cast<int>(blockIdx.x);
  int const n_tile_group = linear_tile_group / row_group_count;
  int const row_group = linear_tile_group - n_tile_group * row_group_count;
  int const n_tile_base =
      (n_tile_group * kWarpsPerBlock + warp_id) * kNTilesPerWarp;
  int const adapter = static_cast<int>(blockIdx.y);
  int const expert = static_cast<int>(blockIdx.z);
  int const out_tiles = static_cast<int>(out_hidden_size >> 4);
  int const bucket = expert * num_lora_adapters + adapter;
  int32_t const row_count = adapter_expert_row_counts[bucket];
  int32_t const row_start = adapter_expert_row_starts[bucket];

  int32_t const raw_rank0 = adapter_lora_ranks0[adapter];
  int32_t const raw_rank1 = adapter_lora_ranks1[adapter];
  int const lora_rank0 = raw_rank0 <= 0 ? 0 : (raw_rank0 < 16 ? raw_rank0 : 16);
  int const lora_rank1 = raw_rank1 <= 0 ? 0 : (raw_rank1 < 16 ? raw_rank1 : 16);
  bool const full_rank0 = lora_rank0 == 16;
  bool const full_rank1 = lora_rank1 == 16;
  auto const* lora_b_base0 =
      reinterpret_cast<__nv_bfloat16 const*>(adapter_lora_weight_ptrs0[adapter * 2 + 1]);
  auto const* lora_b_base1 =
      reinterpret_cast<__nv_bfloat16 const*>(adapter_lora_weight_ptrs1[adapter * 2 + 1]);
  bool const has_lora0 = lora_rank0 > 0 && lora_b_base0 != nullptr;
  bool const has_lora1 = lora_rank1 > 0 && lora_b_base1 != nullptr;
  if (has_lora0) {
    lora_b_base0 += static_cast<size_t>(expert) * static_cast<size_t>(out_hidden_size) * 16;
  }
  if (has_lora1) {
    lora_b_base1 += static_cast<size_t>(expert) * static_cast<size_t>(out_hidden_size) * 16;
  }

  __shared__ __nv_bfloat16 a_smem0[16 * 16];
  __shared__ __nv_bfloat16 a_smem1[16 * 16];
  __shared__ float c_smem[kWarpsPerBlock][kNTilesPerWarp][16 * 16];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>
      b_frags0[kNTilesPerWarp];
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>
      b_frags1[kNTilesPerWarp];
  bool valid_n_tiles[kNTilesPerWarp];
  bool has_valid_lora0_tiles[kNTilesPerWarp];
  bool has_valid_lora1_tiles[kNTilesPerWarp];
  bool do_mma0 = false;
  bool do_mma1 = false;
#pragma unroll
  for (int tile_offset = 0; tile_offset < kNTilesPerWarp; ++tile_offset) {
    int const n_tile = n_tile_base + tile_offset;
    bool const valid_n_tile = n_tile < out_tiles;
    bool const has_valid_lora0 = has_lora0 && valid_n_tile;
    bool const has_valid_lora1 = has_lora1 && valid_n_tile;
    valid_n_tiles[tile_offset] = valid_n_tile;
    has_valid_lora0_tiles[tile_offset] = has_valid_lora0;
    has_valid_lora1_tiles[tile_offset] = has_valid_lora1;
    do_mma0 = do_mma0 || has_valid_lora0;
    do_mma1 = do_mma1 || has_valid_lora1;
    if (has_valid_lora0) {
      wmma::load_matrix_sync(b_frags0[tile_offset],
                             lora_b_base0 + static_cast<size_t>(n_tile) * 16 * 16, 16);
    }
    if (has_valid_lora1) {
      wmma::load_matrix_sync(b_frags1[tile_offset],
                             lora_b_base1 + static_cast<size_t>(n_tile) * 16 * 16, 16);
    }
  }

  if (zero_no_adapter_rows && adapter == 0 && no_adapter_expert_row_counts != nullptr &&
      no_adapter_expert_row_starts != nullptr) {
    int32_t const no_adapter_row_count = no_adapter_expert_row_counts[expert];
    int32_t const no_adapter_row_start = no_adapter_expert_row_starts[expert];
    if (no_adapter_row_count > 0 && no_adapter_row_start >= 0) {
      int const no_adapter_row_span =
          row_split_rows > 0 ? row_split_rows : no_adapter_row_count;
      int const no_adapter_first_row = row_group * no_adapter_row_span;
      int const no_adapter_row_stride = row_group_count * no_adapter_row_span;
      for (int row_block = no_adapter_first_row; row_block < no_adapter_row_count;
           row_block += no_adapter_row_stride) {
        int const row_block_end = min(no_adapter_row_count, row_block + no_adapter_row_span);
        for (int tile_row = row_block; tile_row < row_block_end; tile_row += 16) {
          int const valid_rows = min(16, row_block_end - tile_row);
          int64_t const row = static_cast<int64_t>(no_adapter_row_start) + tile_row;
#pragma unroll
          for (int tile_offset = 0; tile_offset < kNTilesPerWarp; ++tile_offset) {
            int const n_tile = n_tile_base + tile_offset;
            if (!valid_n_tiles[tile_offset]) {
              continue;
            }
            for (int idx = lane; idx < valid_rows * 8; idx += 32) {
              int const local_row = idx >> 3;
              int const local_pair = idx & 7;
              auto* out0_pairs = reinterpret_cast<__nv_bfloat162*>(
                  output0 + (row + local_row) * output_stride + static_cast<int64_t>(n_tile) * 16);
              auto* out1_pairs = reinterpret_cast<__nv_bfloat162*>(
                  output1 + (row + local_row) * output_stride + static_cast<int64_t>(n_tile) * 16);
              out0_pairs[local_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
              out1_pairs[local_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
            }
          }
        }
      }
    }
  }

  if (row_count <= 0 || row_start < 0) {
    return;
  }

  int const row_span = row_split_rows > 0 ? row_split_rows : row_count;
  int const first_row = row_group * row_span;
  int const row_stride = row_group_count * row_span;
  for (int row_block = first_row; row_block < row_count; row_block += row_stride) {
    int const row_block_end = min(row_count, row_block + row_span);
    for (int tile_row = row_block; tile_row < row_block_end; tile_row += 16) {
      int const valid_rows = min(16, row_block_end - tile_row);
      int64_t const row = static_cast<int64_t>(row_start) + tile_row;
      for (int idx = threadIdx.x; idx < 16 * 16; idx += blockDim.x) {
        int const local_row = idx >> 4;
        int const rank = idx & 15;
        bool const valid_row = local_row < valid_rows;
        a_smem0[idx] = (valid_row && has_lora0 && (full_rank0 || rank < lora_rank0))
                           ? low_rank_workspace0[(row + local_row) * 16 + rank]
                           : __float2bfloat16(0.0f);
        a_smem1[idx] = (valid_row && has_lora1 && (full_rank1 || rank < lora_rank1))
                           ? low_rank_workspace1[(row + local_row) * 16 + rank]
                           : __float2bfloat16(0.0f);
      }
      __syncthreads();

      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag0;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag1;
      if (do_mma0) {
        wmma::load_matrix_sync(a_frag0, a_smem0, 16);
      }
      if (do_mma1) {
        wmma::load_matrix_sync(a_frag1, a_smem1, 16);
      }
      __syncthreads();

      if (do_mma0) {
#pragma unroll
        for (int tile_offset = 0; tile_offset < kNTilesPerWarp; ++tile_offset) {
          if (!has_valid_lora0_tiles[tile_offset]) {
            continue;
          }
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
          wmma::fill_fragment(c_frag, 0.0f);
          wmma::mma_sync(c_frag, a_frag0, b_frags0[tile_offset], c_frag);
          wmma::store_matrix_sync(c_smem[warp_id][tile_offset], c_frag, 16,
                                  wmma::mem_row_major);
        }
      }
      __syncwarp();

#pragma unroll
      for (int tile_offset = 0; tile_offset < kNTilesPerWarp; ++tile_offset) {
        int const n_tile = n_tile_base + tile_offset;
        bool const valid_n_tile = valid_n_tiles[tile_offset];
        if (!valid_n_tile) {
          continue;
        }
        bool const has_valid_lora0 = has_valid_lora0_tiles[tile_offset];
        for (int idx = lane; idx < valid_rows * 8; idx += 32) {
          int const local_row = idx >> 3;
          int const local_pair = idx & 7;
          int const c_offset = local_row * 16 + local_pair * 2;
          auto* out0_pairs = reinterpret_cast<__nv_bfloat162*>(
              output0 + (row + local_row) * output_stride + static_cast<int64_t>(n_tile) * 16);
          if (has_valid_lora0) {
            out0_pairs[local_pair] = __floats2bfloat162_rn(
                c_smem[warp_id][tile_offset][c_offset],
                c_smem[warp_id][tile_offset][c_offset + 1]);
          } else {
            out0_pairs[local_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
          }
        }
      }

      if (do_mma1) {
#pragma unroll
        for (int tile_offset = 0; tile_offset < kNTilesPerWarp; ++tile_offset) {
          if (!has_valid_lora1_tiles[tile_offset]) {
            continue;
          }
          wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
          wmma::fill_fragment(c_frag, 0.0f);
          wmma::mma_sync(c_frag, a_frag1, b_frags1[tile_offset], c_frag);
          wmma::store_matrix_sync(c_smem[warp_id][tile_offset], c_frag, 16,
                                  wmma::mem_row_major);
        }
      }
      __syncwarp();

#pragma unroll
      for (int tile_offset = 0; tile_offset < kNTilesPerWarp; ++tile_offset) {
        int const n_tile = n_tile_base + tile_offset;
        bool const valid_n_tile = valid_n_tiles[tile_offset];
        if (!valid_n_tile) {
          continue;
        }
        bool const has_valid_lora1 = has_valid_lora1_tiles[tile_offset];
        for (int idx = lane; idx < valid_rows * 8; idx += 32) {
          int const local_row = idx >> 3;
          int const local_pair = idx & 7;
          int const c_offset = local_row * 16 + local_pair * 2;
          auto* out1_pairs = reinterpret_cast<__nv_bfloat162*>(
              output1 + (row + local_row) * output_stride + static_cast<int64_t>(n_tile) * 16);
          if (has_valid_lora1) {
            out1_pairs[local_pair] = __floats2bfloat162_rn(
                c_smem[warp_id][tile_offset][c_offset],
                c_smem[warp_id][tile_offset][c_offset + 1]);
          } else {
            out1_pairs[local_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
          }
        }
      }
    }
  }
#endif
}

template <int kWarpsPerBlock, int kNTilesPerWarp>
inline void launchLoraOutput16Bf16TensorCoreGroupedDual(
    __nv_bfloat16* output0, __nv_bfloat16* output1, __nv_bfloat16 const* low_rank_workspace0,
    __nv_bfloat16 const* low_rank_workspace1, int64_t out_hidden_size, int64_t output_stride,
    int num_lora_adapters, int num_experts_per_node, int32_t const* adapter_lora_ranks0,
    void const* const* adapter_lora_weight_ptrs0, int32_t const* adapter_lora_ranks1,
    void const* const* adapter_lora_weight_ptrs1, int32_t const* adapter_expert_row_counts,
    int32_t const* adapter_expert_row_starts, int32_t const* no_adapter_expert_row_counts,
    int32_t const* no_adapter_expert_row_starts, cudaStream_t stream, bool zero_no_adapter_rows,
    int row_split_groups, int row_split_rows) {
  constexpr int threads = 32 * kWarpsPerBlock;
  int const n_tile_groups =
      static_cast<int>(((out_hidden_size >> 4) + kWarpsPerBlock * kNTilesPerWarp - 1) /
                       (kWarpsPerBlock * kNTilesPerWarp));
  dim3 const grid(static_cast<unsigned int>(n_tile_groups * row_split_groups),
                  static_cast<unsigned int>(num_lora_adapters),
                  static_cast<unsigned int>(num_experts_per_node));
  loraOutput16Bf16TensorCoreGroupedDualKernel<kWarpsPerBlock, kNTilesPerWarp>
      <<<grid, threads, 0, stream>>>(
          output0, output1, low_rank_workspace0, low_rank_workspace1, out_hidden_size,
          output_stride, num_lora_adapters, num_experts_per_node, adapter_lora_ranks0,
          adapter_lora_weight_ptrs0, adapter_lora_ranks1, adapter_lora_weight_ptrs1,
          adapter_expert_row_counts, adapter_expert_row_starts, no_adapter_expert_row_counts,
          no_adapter_expert_row_starts, zero_no_adapter_rows, row_split_groups, row_split_rows);
}

inline void loraOutput16Bf16TensorCoreGroupedDual(
    __nv_bfloat16* output0, __nv_bfloat16* output1, __nv_bfloat16 const* low_rank_workspace0,
    __nv_bfloat16 const* low_rank_workspace1, int64_t out_hidden_size, int64_t output_stride,
    int64_t expanded_num_rows, int num_lora_adapters, int num_experts_per_node,
    int32_t const* adapter_lora_ranks0, void const* const* adapter_lora_weight_ptrs0,
    int32_t const* adapter_lora_ranks1, void const* const* adapter_lora_weight_ptrs1,
    int32_t const* adapter_expert_row_counts, int32_t const* adapter_expert_row_starts,
    int32_t const* no_adapter_expert_row_counts, int32_t const* no_adapter_expert_row_starts,
    cudaStream_t stream, bool zero_no_adapter_rows = true) {
  bool const use_glm_32k_default_split =
      out_hidden_size == 2048 && output_stride == 4096 && expanded_num_rows >= 262144 &&
      num_experts_per_node == 256;
  int row_split_groups = use_glm_32k_default_split ? 4 : 1;
  int row_split_rows = use_glm_32k_default_split ? 128 : 0;
  if (out_hidden_size == 2048 && output_stride == 4096 && num_experts_per_node == 256) {
    if (char const* env = std::getenv("FLASHINFER_CUTLASS_MOE_LORA_DUAL_OUTPUT_ROW_SPLIT_GROUPS")) {
      row_split_groups = std::atoi(env);
    }
    if (char const* env = std::getenv("FLASHINFER_CUTLASS_MOE_LORA_DUAL_OUTPUT_ROW_SPLIT_ROWS")) {
      row_split_rows = std::atoi(env);
    }
    if (row_split_groups > 1 && row_split_rows <= 0) {
      row_split_rows = 128;
    }
  }
  row_split_groups = std::max(1, std::min(row_split_groups, 64));
  row_split_rows = std::max(0, row_split_rows);

  char const* raw_tile_shape = std::getenv("FLASHINFER_CUTLASS_MOE_LORA_DUAL_OUTPUT_TILE");
  if (raw_tile_shape != nullptr) {
    if (std::strcmp(raw_tile_shape, "8x2") == 0) {
      launchLoraOutput16Bf16TensorCoreGroupedDual<8, 2>(
          output0, output1, low_rank_workspace0, low_rank_workspace1, out_hidden_size,
          output_stride, num_lora_adapters, num_experts_per_node, adapter_lora_ranks0,
          adapter_lora_weight_ptrs0, adapter_lora_ranks1, adapter_lora_weight_ptrs1,
          adapter_expert_row_counts, adapter_expert_row_starts, no_adapter_expert_row_counts,
          no_adapter_expert_row_starts, stream, zero_no_adapter_rows, row_split_groups,
          row_split_rows);
      return;
    }
    if (std::strcmp(raw_tile_shape, "8x4") == 0) {
      launchLoraOutput16Bf16TensorCoreGroupedDual<8, 4>(
          output0, output1, low_rank_workspace0, low_rank_workspace1, out_hidden_size,
          output_stride, num_lora_adapters, num_experts_per_node, adapter_lora_ranks0,
          adapter_lora_weight_ptrs0, adapter_lora_ranks1, adapter_lora_weight_ptrs1,
          adapter_expert_row_counts, adapter_expert_row_starts, no_adapter_expert_row_counts,
          no_adapter_expert_row_starts, stream, zero_no_adapter_rows, row_split_groups,
          row_split_rows);
      return;
    }
    if (std::strcmp(raw_tile_shape, "4x4") == 0) {
      launchLoraOutput16Bf16TensorCoreGroupedDual<4, 4>(
          output0, output1, low_rank_workspace0, low_rank_workspace1, out_hidden_size,
          output_stride, num_lora_adapters, num_experts_per_node, adapter_lora_ranks0,
          adapter_lora_weight_ptrs0, adapter_lora_ranks1, adapter_lora_weight_ptrs1,
          adapter_expert_row_counts, adapter_expert_row_starts, no_adapter_expert_row_counts,
          no_adapter_expert_row_starts, stream, zero_no_adapter_rows, row_split_groups,
          row_split_rows);
      return;
    }
  }
  launchLoraOutput16Bf16TensorCoreGroupedDual<4, 8>(
      output0, output1, low_rank_workspace0, low_rank_workspace1, out_hidden_size, output_stride,
      num_lora_adapters, num_experts_per_node, adapter_lora_ranks0, adapter_lora_weight_ptrs0,
      adapter_lora_ranks1, adapter_lora_weight_ptrs1, adapter_expert_row_counts,
      adapter_expert_row_starts, no_adapter_expert_row_counts, no_adapter_expert_row_starts,
      stream, zero_no_adapter_rows, row_split_groups, row_split_rows);
}

__global__ void zeroNoAdapterGatedLoraOutputBf16Kernel(__nv_bfloat16* output, int64_t inter_size,
                                                       int64_t output_stride,
                                                       int32_t const* row_counts,
                                                       int32_t const* row_starts) {
  int64_t const out_pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t const out_hidden_pairs = inter_size >> 1;
  int const expert = static_cast<int>(blockIdx.y);
  if (out_pair >= out_hidden_pairs) {
    return;
  }
  int const row_count = row_counts[expert];
  int const row_start = row_starts[expert];
  if (row_count <= 0 || row_start < 0) {
    return;
  }
  for (int offset = 0; offset < row_count; ++offset) {
    int64_t const row = static_cast<int64_t>(row_start) + offset;
    auto* gated_output_pairs = reinterpret_cast<__nv_bfloat162*>(output + row * output_stride);
    auto* fc1_output_pairs =
        reinterpret_cast<__nv_bfloat162*>(output + row * output_stride + inter_size);
    gated_output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
    fc1_output_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
  }
}

inline void zeroNoAdapterGatedLoraOutputBf16(__nv_bfloat16* output, int64_t inter_size,
                                             int64_t output_stride, int num_experts_per_node,
                                             int32_t const* row_counts, int32_t const* row_starts,
                                             cudaStream_t stream) {
  constexpr int threads = 256;
  dim3 const grid(static_cast<unsigned int>(((inter_size >> 1) + threads - 1) / threads),
                  static_cast<unsigned int>(num_experts_per_node));
  zeroNoAdapterGatedLoraOutputBf16Kernel<<<grid, threads, 0, stream>>>(
      output, inter_size, output_stride, row_counts, row_starts);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Enable>::gemm2(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType, IsMXFPX>& gemm_runner,
    DeepSeekBlockScaleGemmRunner* fp8_blockscale_gemm_runner, T const* const input,
    void* const gemm_output, OutputType* const final_output,
    int64_t const* const expert_first_token_offset,
    TmaWarpSpecializedGroupedGemmInput const tma_ws_input_template,
    WeightType const* const fc2_expert_weights, ScaleBiasType const* const fc2_expert_biases,
    ScaleBiasType const* const fc2_int_scales, float const* const fc2_fp8_dequant,
    TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat, QuantParams quant_params,
    float const* const unpermuted_final_scales, float const* const permuted_final_scales,
    int const* const unpermuted_row_to_permuted_row, int const* permuted_row_to_unpermuted_row,
    int const* const token_selected_experts, int64_t const* const num_valid_tokens_ptr,
    int64_t const num_rows, int64_t const expanded_num_rows, int64_t const hidden_size,
    int64_t const unpadded_hidden_size, int64_t const inter_size, int const num_experts_per_node,
    int64_t const k, float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora,
    int32_t const* fc2_lora_ranks, cudaStream_t stream, MOEParallelismConfig parallelism_config,
    bool const enable_alltoall, cutlass_extensions::CutlassGemmConfig config, bool min_latency_mode,
    int* num_active_experts_per, int* active_expert_global_ids, bool defer_finalize,
    bool enable_pdl) {
  int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

  bool const using_tma_ws_gemm2 = gemm_runner.isTmaWarpSpecialized(config);

  if (min_latency_mode) {
    TLLM_CHECK_WITH_INFO(using_tma_ws_gemm2,
                         "Only TMA warp specialized GEMM is supported in min latency mode.");
    TLLM_CHECK(use_fp4);
  }

  if (fp8_blockscale_gemm_runner) {
    Self::BlockScaleFC2(*fp8_blockscale_gemm_runner, input, gemm_output, final_output,
                        expert_first_token_offset, fc2_expert_weights, fc2_expert_biases,
                        unpermuted_final_scales, unpermuted_row_to_permuted_row,
                        permuted_row_to_unpermuted_row, token_selected_experts,
                        num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size,
                        unpadded_hidden_size, inter_size, num_experts_per_node, k,
                        parallelism_config, enable_alltoall, quant_params, enable_pdl, stream);
    return;
  }

  TmaWarpSpecializedGroupedGemmInput tma_ws_input{};
  if (using_tma_ws_gemm2) {
    tma_ws_input = tma_ws_input_template;
    if (tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE) {
      // TODO For some reason this has to be done here, it should not overlap with anything else,
      // but doing it in setupTmaWarpSpecializedInputs gives a different result. Ideally, we want
      // this to run on a second stream and overlap with everything else
      //
      // This also means it is included in the timing for the profiler, which is probably more
      // representative until we can overlap it
      check_cuda_error(cudaMemsetAsync(
          final_output, 0x0, sizeof(OutputType) * num_rows * unpadded_hidden_size, stream));
    }
  } else if (use_fp8) {
    alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                   fc2_fp8_dequant, stream);
  }
  if (use_w4afp8) {
    alpha_scale_ptr_array = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node,
                                                   quant_params.groupwise.fc2.alpha, stream);
  }

  bool fuse_lora_bias = use_lora && !(use_fp8 || using_tma_ws_gemm2);
  // Note: expanded_num_rows, to check this value, it's greater than num_rows * num_experts_per_node
  auto universal_input = GroupedGemmInput<T, WeightType, OutputType, OutputType>{
      input,
      total_tokens_including_expert,
      fc2_expert_weights,
      quant_params.groupwise.group_size > 0
          ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc2.weight_scales)
          : fc2_int_scales,
      quant_params.groupwise.group_size > 0
          ? static_cast<ScaleBiasType const*>(quant_params.groupwise.fc2.weight_zeros)
          : nullptr,
      fuse_lora_bias ? static_cast<ScaleBiasType*>(fc2_lora) : nullptr,
      static_cast<OutputType*>(gemm_output),
      alpha_scale_ptr_array,
      /*occupancy*/ nullptr,
      ActivationType::Identity,
      expanded_num_rows,
      /*N*/ hidden_size,
      /*K*/ inter_size,
      num_experts_per_node,
      quant_params.groupwise.group_size,
      /*bias_is_broadcast*/ false,
      /*use_fused_moe*/ false,
      stream,
      config};
  gemm_runner.moeGemmBiasAct(universal_input, tma_ws_input);
  sync_check_cuda_error(stream);

  if (min_latency_mode) return;

  if (use_lora && !fuse_lora_bias) {
    bool used_rank_aware_add = false;
    if constexpr (std::is_same_v<UnfusedGemmOutputType, __nv_bfloat16> &&
                  std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
      used_rank_aware_add =
          fc2_lora_ranks != nullptr && expanded_num_rows >= 32768 && (hidden_size % 2 == 0);
      if (used_rank_aware_add) {
        addLoraBiasRankAwareBf16x2ByRow(static_cast<__nv_bfloat16*>(gemm_output),
                                        static_cast<__nv_bfloat16 const*>(fc2_lora), fc2_lora_ranks,
                                        expert_first_token_offset, num_experts_per_node,
                                        expanded_num_rows, hidden_size, stream);
      }
    }

    if (!used_rank_aware_add) {
      auto loraBiasApplyFunc =
          doActivation<UnfusedGemmOutputType, UnfusedGemmOutputType, ScaleBiasType>;
      loraBiasApplyFunc(static_cast<UnfusedGemmOutputType*>(gemm_output),
                        static_cast<UnfusedGemmOutputType const*>(gemm_output), nullptr,
                        static_cast<ScaleBiasType const*>(fc2_lora), false,
                        expert_first_token_offset, num_experts_per_node, hidden_size,
                        expanded_num_rows, ActivationParams(ActivationType::Identity), {}, false,
                        nullptr, enable_pdl, stream, nullptr);
    }
    sync_check_cuda_error(stream);
  }

  bool has_different_output_type_ampere = (use_w4afp8 || use_fp8) && !using_tma_ws_gemm2;
  bool using_fused_finalize =
      tma_ws_input.fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
  bool has_different_output_type_tma_ws = !using_fused_finalize && using_tma_ws_gemm2;

  if (defer_finalize) {
    return;
  }

  if (has_different_output_type_ampere || has_different_output_type_tma_ws) {
    finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
        static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
        unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
        token_selected_experts, expert_first_token_offset, num_rows, hidden_size,
        unpadded_hidden_size, k, num_experts_per_node, parallelism_config, enable_alltoall,
        enable_pdl, stream);
  } else if (!using_tma_ws_gemm2) {
    finalizeMoeRoutingKernelLauncher<OutputType, T>(
        static_cast<T const*>(gemm_output), final_output, fc2_expert_biases,
        unpermuted_final_scales, unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row,
        token_selected_experts, expert_first_token_offset, num_rows, hidden_size,
        unpadded_hidden_size, k, num_experts_per_node, parallelism_config, enable_alltoall,
        enable_pdl, stream);
  }
  sync_check_cuda_error(stream);
}

inline size_t loraMetadataAlign16(size_t value) { return ((value + 15) / 16) * 16; }

inline char* advanceLoraMetadataWorkspace(char*& ptr, size_t size) {
  char* out = ptr;
  ptr += loraMetadataAlign16(size);
  return out;
}

__device__ __forceinline__ void appendLoraAdapterExpertRow(int32_t row, int expert_idx,
                                                           int64_t lora_idx, int num_lora_adapters,
                                                           int row_capacity,
                                                           int32_t* adapter_expert_row_counts,
                                                           int32_t* adapter_expert_row_indices) {
  if (adapter_expert_row_counts == nullptr || adapter_expert_row_indices == nullptr ||
      row_capacity <= 0) {
    return;
  }
  int const bucket = expert_idx * num_lora_adapters + static_cast<int>(lora_idx);
  int const offset = atomicAdd(adapter_expert_row_counts + bucket, 1);
  if (offset < row_capacity) {
    adapter_expert_row_indices[static_cast<int64_t>(bucket) * row_capacity + offset] = row;
  }
}

__device__ __forceinline__ void updateLoraAdapterExpertSpan(int32_t row, int expert_idx,
                                                            int64_t lora_idx, int num_lora_adapters,
                                                            int32_t* adapter_expert_row_counts,
                                                            int32_t* adapter_expert_row_starts) {
  if (adapter_expert_row_counts == nullptr || adapter_expert_row_starts == nullptr) {
    return;
  }
  int const bucket = expert_idx * num_lora_adapters + static_cast<int>(lora_idx);
  atomicAdd(adapter_expert_row_counts + bucket, 1);
  atomicMin(adapter_expert_row_starts + bucket, row);
}

__device__ __forceinline__ void appendNoLoraAdapterExpertRow(
    int32_t row, int expert_idx, int row_capacity, int32_t* no_adapter_expert_row_counts,
    int32_t* no_adapter_expert_row_indices) {
  if (no_adapter_expert_row_counts == nullptr || no_adapter_expert_row_indices == nullptr ||
      row_capacity <= 0) {
    return;
  }
  int const offset = atomicAdd(no_adapter_expert_row_counts + expert_idx, 1);
  if (offset < row_capacity) {
    no_adapter_expert_row_indices[static_cast<int64_t>(expert_idx) * row_capacity + offset] = row;
  }
}

__device__ __forceinline__ void updateNoLoraAdapterExpertSpan(
    int32_t row, int expert_idx, int32_t* no_adapter_expert_row_counts,
    int32_t* no_adapter_expert_row_starts) {
  if (no_adapter_expert_row_counts == nullptr || no_adapter_expert_row_starts == nullptr) {
    return;
  }
  atomicAdd(no_adapter_expert_row_counts + expert_idx, 1);
  atomicMin(no_adapter_expert_row_starts + expert_idx, row);
}

template <typename ScaleBiasType, typename TokenIndexT>
__global__ void prepareLoraMetadataKernel(
    int64_t expanded_num_rows, int64_t num_rows, int64_t a_expert_stride, int64_t b_expert_stride,
    int num_lora_adapters, int max_low_rank, int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row, TokenIndexT const* token_lora_indices,
    int32_t const* adapter_lora_ranks, void const* const* adapter_lora_weight_ptrs,
    int32_t* permuted_lora_ranks, void const** permuted_lora_weight_ptrs,
    int32_t* permuted_lora_indices, int32_t* adapter_expert_row_counts,
    int32_t* adapter_expert_row_indices, int adapter_expert_row_capacity,
    int32_t* adapter_expert_row_starts, bool rows_grouped_by_adapter) {
  int const expert_idx = blockIdx.x;
  int64_t const begin = expert_first_token_offset[expert_idx];
  int64_t const end = expert_first_token_offset[expert_idx + 1];

  for (int64_t row = begin + threadIdx.x; row < end; row += blockDim.x) {
    if (row >= expanded_num_rows) {
      continue;
    }
    int const source_index = permuted_row_to_unpermuted_row[row] % num_rows;
    int64_t const lora_idx = static_cast<int64_t>(token_lora_indices[source_index]);
    if (permuted_lora_indices != nullptr) {
      permuted_lora_indices[row] =
          (lora_idx < 0 || lora_idx >= num_lora_adapters) ? -1 : static_cast<int32_t>(lora_idx);
    }
    if (lora_idx < 0 || lora_idx >= num_lora_adapters) {
      permuted_lora_ranks[row] = 0;
      continue;
    }

    int32_t lora_rank = adapter_lora_ranks[lora_idx];
    if (lora_rank <= 0) {
      permuted_lora_ranks[row] = 0;
      continue;
    }
    if (lora_rank > max_low_rank) {
      lora_rank = max_low_rank;
    }

    auto const* lora_a =
        reinterpret_cast<ScaleBiasType const*>(adapter_lora_weight_ptrs[lora_idx * 2]);
    auto const* lora_b =
        reinterpret_cast<ScaleBiasType const*>(adapter_lora_weight_ptrs[lora_idx * 2 + 1]);
    if (lora_a == nullptr || lora_b == nullptr) {
      permuted_lora_ranks[row] = 0;
      continue;
    }

    permuted_lora_ranks[row] = lora_rank;
    permuted_lora_weight_ptrs[row * 2] =
        lora_a + static_cast<size_t>(expert_idx) * static_cast<size_t>(a_expert_stride);
    permuted_lora_weight_ptrs[row * 2 + 1] =
        lora_b + static_cast<size_t>(expert_idx) * static_cast<size_t>(b_expert_stride);
    if (rows_grouped_by_adapter) {
      updateLoraAdapterExpertSpan(static_cast<int32_t>(row), expert_idx, lora_idx,
                                  num_lora_adapters, adapter_expert_row_counts,
                                  adapter_expert_row_starts);
    } else {
      appendLoraAdapterExpertRow(static_cast<int32_t>(row), expert_idx, lora_idx, num_lora_adapters,
                                 adapter_expert_row_capacity, adapter_expert_row_counts,
                                 adapter_expert_row_indices);
    }
  }
}

__device__ __forceinline__ void clearLoraMetadataEntry(int64_t row, int32_t* ranks,
                                                       void const** ptrs) {
  ranks[row] = 0;
  ptrs[row * 2] = nullptr;
  ptrs[row * 2 + 1] = nullptr;
}

template <typename ScaleBiasType>
__device__ __forceinline__ void fillLoraMetadataEntry(int64_t row, int expert_idx, int64_t lora_idx,
                                                      int64_t a_expert_stride,
                                                      int64_t b_expert_stride, int max_low_rank,
                                                      int32_t const* adapter_lora_ranks,
                                                      void const* const* adapter_lora_weight_ptrs,
                                                      int32_t* permuted_lora_ranks,
                                                      void const** permuted_lora_weight_ptrs) {
  int32_t lora_rank = adapter_lora_ranks[lora_idx];
  if (lora_rank <= 0) {
    clearLoraMetadataEntry(row, permuted_lora_ranks, permuted_lora_weight_ptrs);
    return;
  }
  if (lora_rank > max_low_rank) {
    lora_rank = max_low_rank;
  }

  auto const* lora_a =
      reinterpret_cast<ScaleBiasType const*>(adapter_lora_weight_ptrs[lora_idx * 2]);
  auto const* lora_b =
      reinterpret_cast<ScaleBiasType const*>(adapter_lora_weight_ptrs[lora_idx * 2 + 1]);
  if (lora_a == nullptr || lora_b == nullptr) {
    clearLoraMetadataEntry(row, permuted_lora_ranks, permuted_lora_weight_ptrs);
    return;
  }

  permuted_lora_ranks[row] = lora_rank;
  permuted_lora_weight_ptrs[row * 2] =
      lora_a + static_cast<size_t>(expert_idx) * static_cast<size_t>(a_expert_stride);
  permuted_lora_weight_ptrs[row * 2 + 1] =
      lora_b + static_cast<size_t>(expert_idx) * static_cast<size_t>(b_expert_stride);
}

template <typename ScaleBiasType, typename TokenIndexT>
__global__ void prepareGatedLoraMetadataKernel(
    int64_t expanded_num_rows, int64_t num_rows, int64_t fc1_a_expert_stride,
    int64_t fc1_b_expert_stride, int64_t fc2_a_expert_stride, int64_t fc2_b_expert_stride,
    int num_lora_adapters, int max_low_rank, int64_t const* expert_first_token_offset,
    int const* permuted_row_to_unpermuted_row, TokenIndexT const* token_lora_indices,
    int32_t const* fc1_lora_ranks, void const* const* fc1_lora_weight_ptrs,
    int32_t const* fc2_lora_ranks, void const* const* fc2_lora_weight_ptrs,
    int32_t const* gated_lora_ranks, void const* const* gated_lora_weight_ptrs,
    int32_t* permuted_fc1_lora_ranks, void const** permuted_fc1_lora_weight_ptrs,
    int32_t* permuted_fc2_lora_ranks, void const** permuted_fc2_lora_weight_ptrs,
    int32_t* permuted_gated_lora_ranks, void const** permuted_gated_lora_weight_ptrs,
    int32_t* permuted_lora_indices, int32_t* adapter_expert_row_counts,
    int32_t* adapter_expert_row_indices, int adapter_expert_row_capacity,
    int32_t* adapter_expert_row_starts, bool rows_grouped_by_adapter,
    int32_t* no_adapter_expert_row_counts, int32_t* no_adapter_expert_row_indices,
    int no_adapter_expert_row_capacity, int32_t* no_adapter_expert_row_starts,
    int32_t* fc1_gated_lora_has_missing_rows) {
  int const expert_idx = blockIdx.x;
  int64_t const begin = expert_first_token_offset[expert_idx];
  int64_t const end = expert_first_token_offset[expert_idx + 1];

  for (int64_t row = begin + threadIdx.x; row < end; row += blockDim.x) {
    if (row >= expanded_num_rows) {
      continue;
    }

    int const source_index = permuted_row_to_unpermuted_row[row] % num_rows;
    int64_t const lora_idx = static_cast<int64_t>(token_lora_indices[source_index]);
    if (permuted_lora_indices != nullptr) {
      permuted_lora_indices[row] =
          (lora_idx < 0 || lora_idx >= num_lora_adapters) ? -1 : static_cast<int32_t>(lora_idx);
    }
    if (lora_idx < 0 || lora_idx >= num_lora_adapters) {
      clearLoraMetadataEntry(row, permuted_fc1_lora_ranks, permuted_fc1_lora_weight_ptrs);
      clearLoraMetadataEntry(row, permuted_fc2_lora_ranks, permuted_fc2_lora_weight_ptrs);
      clearLoraMetadataEntry(row, permuted_gated_lora_ranks, permuted_gated_lora_weight_ptrs);
      if (rows_grouped_by_adapter) {
        updateNoLoraAdapterExpertSpan(static_cast<int32_t>(row), expert_idx,
                                      no_adapter_expert_row_counts, no_adapter_expert_row_starts);
      } else {
        appendNoLoraAdapterExpertRow(static_cast<int32_t>(row), expert_idx,
                                     no_adapter_expert_row_capacity, no_adapter_expert_row_counts,
                                     no_adapter_expert_row_indices);
      }
      continue;
    }

    fillLoraMetadataEntry<ScaleBiasType>(row, expert_idx, lora_idx, fc1_a_expert_stride,
                                         fc1_b_expert_stride, max_low_rank, fc1_lora_ranks,
                                         fc1_lora_weight_ptrs, permuted_fc1_lora_ranks,
                                         permuted_fc1_lora_weight_ptrs);
    fillLoraMetadataEntry<ScaleBiasType>(row, expert_idx, lora_idx, fc2_a_expert_stride,
                                         fc2_b_expert_stride, max_low_rank, fc2_lora_ranks,
                                         fc2_lora_weight_ptrs, permuted_fc2_lora_ranks,
                                         permuted_fc2_lora_weight_ptrs);
    fillLoraMetadataEntry<ScaleBiasType>(row, expert_idx, lora_idx, fc1_a_expert_stride,
                                         fc1_b_expert_stride, max_low_rank, gated_lora_ranks,
                                         gated_lora_weight_ptrs, permuted_gated_lora_ranks,
                                         permuted_gated_lora_weight_ptrs);
    if (fc1_gated_lora_has_missing_rows != nullptr &&
        (permuted_gated_lora_weight_ptrs[row * 2 + 1] == nullptr ||
         permuted_fc1_lora_weight_ptrs[row * 2 + 1] == nullptr)) {
      atomicExch(fc1_gated_lora_has_missing_rows, 1);
    }
    if (permuted_fc1_lora_ranks[row] > 0 || permuted_fc2_lora_ranks[row] > 0 ||
        permuted_gated_lora_ranks[row] > 0) {
      if (rows_grouped_by_adapter) {
        updateLoraAdapterExpertSpan(static_cast<int32_t>(row), expert_idx, lora_idx,
                                    num_lora_adapters, adapter_expert_row_counts,
                                    adapter_expert_row_starts);
      } else {
        appendLoraAdapterExpertRow(static_cast<int32_t>(row), expert_idx, lora_idx,
                                   num_lora_adapters, adapter_expert_row_capacity,
                                   adapter_expert_row_counts, adapter_expert_row_indices);
      }
    }
  }
}

template <typename ScaleBiasType>
void prepareLoraMetadata(int64_t expanded_num_rows, int64_t num_rows, int64_t a_expert_stride,
                         int64_t b_expert_stride, int num_lora_adapters, int max_low_rank,
                         int num_experts_per_node, int64_t const* expert_first_token_offset,
                         int const* permuted_row_to_unpermuted_row, void const* token_lora_indices,
                         bool token_lora_indices_is_int64, int32_t const* adapter_lora_ranks,
                         void const* const* adapter_lora_weight_ptrs, int32_t* permuted_lora_ranks,
                         void const** permuted_lora_weight_ptrs, int32_t* permuted_lora_indices,
                         int32_t* adapter_expert_row_counts, int32_t* adapter_expert_row_indices,
                         int adapter_expert_row_capacity, int32_t* adapter_expert_row_starts,
                         bool rows_grouped_by_adapter, cudaStream_t stream) {
  TLLM_CUDA_CHECK(
      cudaMemsetAsync(permuted_lora_ranks, 0, expanded_num_rows * sizeof(int32_t), stream));
  constexpr int threads = 256;
  if (token_lora_indices_is_int64) {
    prepareLoraMetadataKernel<ScaleBiasType, int64_t><<<num_experts_per_node, threads, 0, stream>>>(
        expanded_num_rows, num_rows, a_expert_stride, b_expert_stride, num_lora_adapters,
        max_low_rank, expert_first_token_offset, permuted_row_to_unpermuted_row,
        static_cast<int64_t const*>(token_lora_indices), adapter_lora_ranks,
        adapter_lora_weight_ptrs, permuted_lora_ranks, permuted_lora_weight_ptrs,
        permuted_lora_indices, adapter_expert_row_counts, adapter_expert_row_indices,
        adapter_expert_row_capacity, adapter_expert_row_starts, rows_grouped_by_adapter);
  } else {
    prepareLoraMetadataKernel<ScaleBiasType, int32_t><<<num_experts_per_node, threads, 0, stream>>>(
        expanded_num_rows, num_rows, a_expert_stride, b_expert_stride, num_lora_adapters,
        max_low_rank, expert_first_token_offset, permuted_row_to_unpermuted_row,
        static_cast<int32_t const*>(token_lora_indices), adapter_lora_ranks,
        adapter_lora_weight_ptrs, permuted_lora_ranks, permuted_lora_weight_ptrs,
        permuted_lora_indices, adapter_expert_row_counts, adapter_expert_row_indices,
        adapter_expert_row_capacity, adapter_expert_row_starts, rows_grouped_by_adapter);
  }
}

template <typename ScaleBiasType>
void prepareGatedLoraMetadata(
    int64_t expanded_num_rows, int64_t num_rows, int64_t fc1_a_expert_stride,
    int64_t fc1_b_expert_stride, int64_t fc2_a_expert_stride, int64_t fc2_b_expert_stride,
    int num_lora_adapters, int max_low_rank, int num_experts_per_node,
    int64_t const* expert_first_token_offset, int const* permuted_row_to_unpermuted_row,
    void const* token_lora_indices, bool token_lora_indices_is_int64, int32_t const* fc1_lora_ranks,
    void const* const* fc1_lora_weight_ptrs, int32_t const* fc2_lora_ranks,
    void const* const* fc2_lora_weight_ptrs, int32_t const* gated_lora_ranks,
    void const* const* gated_lora_weight_ptrs, int32_t* permuted_fc1_lora_ranks,
    void const** permuted_fc1_lora_weight_ptrs, int32_t* permuted_fc2_lora_ranks,
    void const** permuted_fc2_lora_weight_ptrs, int32_t* permuted_gated_lora_ranks,
    void const** permuted_gated_lora_weight_ptrs, int32_t* permuted_lora_indices,
    int32_t* adapter_expert_row_counts, int32_t* adapter_expert_row_indices,
    int adapter_expert_row_capacity, int32_t* adapter_expert_row_starts,
    bool rows_grouped_by_adapter, int32_t* no_adapter_expert_row_counts,
    int32_t* no_adapter_expert_row_indices, int no_adapter_expert_row_capacity,
    int32_t* no_adapter_expert_row_starts, int32_t* fc1_gated_lora_has_missing_rows,
    cudaStream_t stream) {
  constexpr int threads = 256;
  if (token_lora_indices_is_int64) {
    prepareGatedLoraMetadataKernel<ScaleBiasType, int64_t>
        <<<num_experts_per_node, threads, 0, stream>>>(
            expanded_num_rows, num_rows, fc1_a_expert_stride, fc1_b_expert_stride,
            fc2_a_expert_stride, fc2_b_expert_stride, num_lora_adapters, max_low_rank,
            expert_first_token_offset, permuted_row_to_unpermuted_row,
            static_cast<int64_t const*>(token_lora_indices), fc1_lora_ranks, fc1_lora_weight_ptrs,
            fc2_lora_ranks, fc2_lora_weight_ptrs, gated_lora_ranks, gated_lora_weight_ptrs,
            permuted_fc1_lora_ranks, permuted_fc1_lora_weight_ptrs, permuted_fc2_lora_ranks,
            permuted_fc2_lora_weight_ptrs, permuted_gated_lora_ranks,
            permuted_gated_lora_weight_ptrs, permuted_lora_indices, adapter_expert_row_counts,
            adapter_expert_row_indices, adapter_expert_row_capacity, adapter_expert_row_starts,
            rows_grouped_by_adapter, no_adapter_expert_row_counts, no_adapter_expert_row_indices,
            no_adapter_expert_row_capacity, no_adapter_expert_row_starts,
            fc1_gated_lora_has_missing_rows);
  } else {
    prepareGatedLoraMetadataKernel<ScaleBiasType, int32_t>
        <<<num_experts_per_node, threads, 0, stream>>>(
            expanded_num_rows, num_rows, fc1_a_expert_stride, fc1_b_expert_stride,
            fc2_a_expert_stride, fc2_b_expert_stride, num_lora_adapters, max_low_rank,
            expert_first_token_offset, permuted_row_to_unpermuted_row,
            static_cast<int32_t const*>(token_lora_indices), fc1_lora_ranks, fc1_lora_weight_ptrs,
            fc2_lora_ranks, fc2_lora_weight_ptrs, gated_lora_ranks, gated_lora_weight_ptrs,
            permuted_fc1_lora_ranks, permuted_fc1_lora_weight_ptrs, permuted_fc2_lora_ranks,
            permuted_fc2_lora_weight_ptrs, permuted_gated_lora_ranks,
            permuted_gated_lora_weight_ptrs, permuted_lora_indices, adapter_expert_row_counts,
            adapter_expert_row_indices, adapter_expert_row_capacity, adapter_expert_row_starts,
            rows_grouped_by_adapter, no_adapter_expert_row_counts, no_adapter_expert_row_indices,
            no_adapter_expert_row_capacity, no_adapter_expert_row_starts,
            fc1_gated_lora_has_missing_rows);
  }
}

__global__ void prepareGroupedGatedLoraFastMetadataKernel(
    int64_t expanded_num_rows, int num_experts_per_node, int num_lora_adapters, int max_low_rank,
    int const* adapter_bucket_counts, int const* adapter_bucket_offsets,
    int32_t const* fc2_lora_ranks, void const* const* fc2_lora_weight_ptrs,
    int32_t const* fc1_lora_ranks, void const* const* fc1_lora_weight_ptrs,
    int32_t const* gated_lora_ranks, void const* const* gated_lora_weight_ptrs,
    int32_t* permuted_fc2_lora_ranks, int32_t* permuted_fc1_gated_lora_bias_ranks,
    int32_t* adapter_expert_row_counts, int32_t* adapter_expert_row_starts,
    int32_t* no_adapter_expert_row_counts, int32_t* no_adapter_expert_row_starts,
    int32_t* fc1_gated_lora_has_missing_rows) {
  int const expert = static_cast<int>(blockIdx.x);
  int const bucket = static_cast<int>(blockIdx.y);
  int const adapter_bucket_count = num_lora_adapters + 1;
  if (expert >= num_experts_per_node || bucket >= adapter_bucket_count) {
    return;
  }

  int const src = expert * adapter_bucket_count + bucket;
  int const row_count = adapter_bucket_counts[src];
  int const row_start = adapter_bucket_offsets[src];

  int rank = 0;
  int fc1_gated_bias_rank = 0;
  if (bucket > 0) {
    int const adapter = bucket - 1;
    int32_t const raw_rank = fc2_lora_ranks[adapter];
    auto const* lora_a = fc2_lora_weight_ptrs[adapter * 2];
    auto const* lora_b = fc2_lora_weight_ptrs[adapter * 2 + 1];
    rank = (raw_rank > 0 && lora_a != nullptr && lora_b != nullptr)
               ? (raw_rank < max_low_rank ? raw_rank : max_low_rank)
               : 0;
    bool const has_fc1_lora = fc1_lora_ranks != nullptr && fc1_lora_weight_ptrs != nullptr &&
                              fc1_lora_ranks[adapter] > 0 &&
                              fc1_lora_weight_ptrs[adapter * 2] != nullptr &&
                              fc1_lora_weight_ptrs[adapter * 2 + 1] != nullptr;
    bool const has_gated_lora = gated_lora_ranks != nullptr && gated_lora_weight_ptrs != nullptr &&
                                gated_lora_ranks[adapter] > 0 &&
                                gated_lora_weight_ptrs[adapter * 2] != nullptr &&
                                gated_lora_weight_ptrs[adapter * 2 + 1] != nullptr;
    fc1_gated_bias_rank = (has_fc1_lora || has_gated_lora) ? 1 : 0;
    if (threadIdx.x == 0 && adapter_expert_row_counts != nullptr &&
        adapter_expert_row_starts != nullptr) {
      int const dst = expert * num_lora_adapters + adapter;
      adapter_expert_row_counts[dst] = row_count;
      adapter_expert_row_starts[dst] = row_count > 0 ? row_start : 0x7f7f7f7f;
    }
  } else if (threadIdx.x == 0 && no_adapter_expert_row_counts != nullptr &&
             no_adapter_expert_row_starts != nullptr) {
    no_adapter_expert_row_counts[expert] = row_count;
    no_adapter_expert_row_starts[expert] = row_count > 0 ? row_start : 0x7f7f7f7f;
  }

  if ((permuted_fc2_lora_ranks != nullptr || permuted_fc1_gated_lora_bias_ranks != nullptr) &&
      row_count > 0 && row_start >= 0) {
    int64_t const begin = static_cast<int64_t>(row_start);
    int64_t const end = min(begin + static_cast<int64_t>(row_count), expanded_num_rows);
    for (int64_t row = begin + threadIdx.x; row < end; row += blockDim.x) {
      if (permuted_fc2_lora_ranks != nullptr) {
        permuted_fc2_lora_ranks[row] = rank;
      }
      if (permuted_fc1_gated_lora_bias_ranks != nullptr) {
        permuted_fc1_gated_lora_bias_ranks[row] = fc1_gated_bias_rank;
      }
    }
  }

  if (threadIdx.x == 0 && fc1_gated_lora_has_missing_rows != nullptr) {
    *fc1_gated_lora_has_missing_rows = 0;
  }
}

void prepareGroupedGatedLoraFastMetadata(
    int64_t expanded_num_rows, int num_experts_per_node, int num_lora_adapters, int max_low_rank,
    int const* adapter_bucket_counts, int const* adapter_bucket_offsets,
    int32_t const* fc2_lora_ranks, void const* const* fc2_lora_weight_ptrs,
    int32_t const* fc1_lora_ranks, void const* const* fc1_lora_weight_ptrs,
    int32_t const* gated_lora_ranks, void const* const* gated_lora_weight_ptrs,
    int32_t* permuted_fc2_lora_ranks, int32_t* permuted_fc1_gated_lora_bias_ranks,
    int32_t* adapter_expert_row_counts, int32_t* adapter_expert_row_starts,
    int32_t* no_adapter_expert_row_counts, int32_t* no_adapter_expert_row_starts,
    int32_t* fc1_gated_lora_has_missing_rows, cudaStream_t stream) {
  constexpr int threads = 256;
  dim3 const grid(static_cast<unsigned int>(num_experts_per_node),
                  static_cast<unsigned int>(num_lora_adapters + 1));
  prepareGroupedGatedLoraFastMetadataKernel<<<grid, threads, 0, stream>>>(
      expanded_num_rows, num_experts_per_node, num_lora_adapters, max_low_rank,
      adapter_bucket_counts, adapter_bucket_offsets, fc2_lora_ranks, fc2_lora_weight_ptrs,
      fc1_lora_ranks, fc1_lora_weight_ptrs, gated_lora_ranks, gated_lora_weight_ptrs,
      permuted_fc2_lora_ranks, permuted_fc1_gated_lora_bias_ranks, adapter_expert_row_counts,
      adapter_expert_row_starts, no_adapter_expert_row_counts, no_adapter_expert_row_starts,
      fc1_gated_lora_has_missing_rows);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
bool CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Enable>::setupLoraWorkspace(int64_t expanded_num_rows, int64_t num_rows,
                                                    int64_t inter_size, int64_t hidden_size,
                                                    int start_expert, bool is_gated_activation,
                                                    int num_experts_per_node, bool needs_num_valid,
                                                    LoraParams& lora_params,
                                                    bool use_grouped_lora_metadata_fast_path,
                                                    cudaStream_t stream) {
  TLLM_CHECK_WITH_INFO(!needs_num_valid && start_expert == 0,
                       "LoRA metadata setup does not support expert parallelism yet");
  TLLM_CHECK_WITH_INFO(lora_params.metadata_workspace != nullptr,
                       "LoRA metadata workspace must not be null");
  TLLM_CHECK_WITH_INFO(lora_params.token_lora_indices != nullptr,
                       "LoRA token mapping must not be null");
  TLLM_CHECK_WITH_INFO(lora_params.max_low_rank > 0, "LoRA max rank must be positive");
  TLLM_CHECK_WITH_INFO(lora_params.num_lora_adapters > 0, "LoRA adapter count must be positive");

  char* metadata_ptr = static_cast<char*>(lora_params.metadata_workspace);
  auto allocateRanks = [&](int64_t entries) {
    return reinterpret_cast<int32_t*>(
        advanceLoraMetadataWorkspace(metadata_ptr, entries * sizeof(int32_t)));
  };
  auto allocatePtrs = [&](int64_t entries) {
    return reinterpret_cast<void const**>(
        advanceLoraMetadataWorkspace(metadata_ptr, entries * 2 * sizeof(void const*)));
  };
  int64_t const adapter_expert_buckets =
      static_cast<int64_t>(lora_params.num_lora_adapters) * num_experts_per_node;
  bool const rows_grouped_by_adapter = lora_params.rows_grouped_by_adapter;
  bool use_fast_grouped_metadata = false;
  if constexpr (std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
    use_fast_grouped_metadata = use_grouped_lora_metadata_fast_path && is_gated_activation &&
                                rows_grouped_by_adapter && lora_params.max_low_rank == 16 &&
                                expanded_num_rows >= 1024 && hidden_size % 16 == 0 &&
                                inter_size % 16 == 0;
  }

  if (use_fast_grouped_metadata) {
    TLLM_CHECK(blocked_expert_counts_ != nullptr);
    TLLM_CHECK(blocked_expert_counts_cumsum_ != nullptr);
  }

  lora_params.adapter_expert_row_counts = reinterpret_cast<int32_t*>(
      advanceLoraMetadataWorkspace(metadata_ptr, adapter_expert_buckets * sizeof(int32_t)));
  lora_params.adapter_expert_row_starts = reinterpret_cast<int32_t*>(
      advanceLoraMetadataWorkspace(metadata_ptr, adapter_expert_buckets * sizeof(int32_t)));
  if (use_fast_grouped_metadata) {
    lora_params.adapter_expert_row_capacity = 0;
    lora_params.adapter_expert_row_indices = nullptr;
  } else {
    lora_params.adapter_expert_row_capacity = static_cast<int>(num_rows);
    lora_params.adapter_expert_row_indices =
        reinterpret_cast<int32_t*>(advanceLoraMetadataWorkspace(
            metadata_ptr, adapter_expert_buckets * num_rows * sizeof(int32_t)));
    TLLM_CUDA_CHECK(cudaMemsetAsync(lora_params.adapter_expert_row_counts, 0,
                                    adapter_expert_buckets * sizeof(int32_t), stream));
    TLLM_CUDA_CHECK(cudaMemsetAsync(lora_params.adapter_expert_row_starts, 0x7f,
                                    adapter_expert_buckets * sizeof(int32_t), stream));
  }

  bool const needs_fc1_no_adapter_row_list =
      is_gated_activation && lora_params.max_low_rank == 16 && expanded_num_rows >= 1024 &&
      (inter_size % 2 == 0);
  if (needs_fc1_no_adapter_row_list) {
    lora_params.no_adapter_expert_row_counts =
        reinterpret_cast<int32_t*>(advanceLoraMetadataWorkspace(
            metadata_ptr, static_cast<int64_t>(num_experts_per_node) * sizeof(int32_t)));
    lora_params.no_adapter_expert_row_starts =
        reinterpret_cast<int32_t*>(advanceLoraMetadataWorkspace(
            metadata_ptr, static_cast<int64_t>(num_experts_per_node) * sizeof(int32_t)));
    if (use_fast_grouped_metadata) {
      lora_params.no_adapter_expert_row_capacity = 0;
      lora_params.no_adapter_expert_row_indices = nullptr;
    } else {
      lora_params.no_adapter_expert_row_capacity = static_cast<int>(num_rows);
      lora_params.no_adapter_expert_row_indices = reinterpret_cast<int32_t*>(
          advanceLoraMetadataWorkspace(metadata_ptr, static_cast<int64_t>(num_experts_per_node) *
                                                         num_rows * sizeof(int32_t)));
    }
    lora_params.fc1_gated_lora_has_missing_rows =
        reinterpret_cast<int32_t*>(advanceLoraMetadataWorkspace(metadata_ptr, sizeof(int32_t)));
    if (!use_fast_grouped_metadata) {
      TLLM_CUDA_CHECK(cudaMemsetAsync(lora_params.no_adapter_expert_row_counts, 0,
                                      static_cast<int64_t>(num_experts_per_node) * sizeof(int32_t),
                                      stream));
      TLLM_CUDA_CHECK(cudaMemsetAsync(lora_params.no_adapter_expert_row_starts, 0x7f,
                                      static_cast<int64_t>(num_experts_per_node) * sizeof(int32_t),
                                      stream));
      TLLM_CUDA_CHECK(
          cudaMemsetAsync(lora_params.fc1_gated_lora_has_missing_rows, 0, sizeof(int32_t), stream));
    }
  } else {
    lora_params.no_adapter_expert_row_counts = nullptr;
    lora_params.no_adapter_expert_row_indices = nullptr;
    lora_params.no_adapter_expert_row_capacity = 0;
    lora_params.no_adapter_expert_row_starts = nullptr;
    lora_params.fc1_gated_lora_has_missing_rows = nullptr;
  }

  lora_params.permuted_fc1_gated_lora_bias_ranks =
      use_fast_grouped_metadata ? allocateRanks(expanded_num_rows) : nullptr;

  if (is_gated_activation && !use_fast_grouped_metadata) {
    auto* permuted_fc1_gated_lora_ranks = allocateRanks(2 * expanded_num_rows);
    auto* permuted_fc1_gated_lora_weight_ptrs = allocatePtrs(2 * expanded_num_rows);
    lora_params.permuted_gated_lora_ranks = permuted_fc1_gated_lora_ranks;
    lora_params.permuted_gated_lora_weight_ptrs = permuted_fc1_gated_lora_weight_ptrs;
    lora_params.permuted_fc1_lora_ranks = permuted_fc1_gated_lora_ranks + expanded_num_rows;
    lora_params.permuted_fc1_lora_weight_ptrs =
        permuted_fc1_gated_lora_weight_ptrs + expanded_num_rows * 2;
  } else if (!is_gated_activation) {
    lora_params.permuted_fc1_lora_ranks = allocateRanks(expanded_num_rows);
    lora_params.permuted_fc1_lora_weight_ptrs = allocatePtrs(expanded_num_rows);
  } else {
    lora_params.permuted_gated_lora_ranks = nullptr;
    lora_params.permuted_gated_lora_weight_ptrs = nullptr;
    lora_params.permuted_fc1_lora_ranks = nullptr;
    lora_params.permuted_fc1_lora_weight_ptrs = nullptr;
  }
  lora_params.permuted_fc2_lora_ranks = allocateRanks(expanded_num_rows);
  lora_params.permuted_fc2_lora_weight_ptrs =
      use_fast_grouped_metadata ? nullptr : allocatePtrs(expanded_num_rows);

  int64_t const fc1_a_expert_stride = hidden_size * lora_params.max_low_rank;
  int64_t const fc1_b_expert_stride = inter_size * lora_params.max_low_rank;
  int64_t const fc2_a_expert_stride = inter_size * lora_params.max_low_rank;
  int64_t const fc2_b_expert_stride = hidden_size * lora_params.max_low_rank;

  if (use_fast_grouped_metadata) {
    lora_params.permuted_lora_indices = nullptr;
    prepareGroupedGatedLoraFastMetadata(
        expanded_num_rows, num_experts_per_node, lora_params.num_lora_adapters,
        lora_params.max_low_rank, blocked_expert_counts_, blocked_expert_counts_cumsum_,
        lora_params.fc2_lora_ranks, lora_params.fc2_lora_weight_ptrs, lora_params.fc1_lora_ranks,
        lora_params.fc1_lora_weight_ptrs, lora_params.gated_lora_ranks,
        lora_params.gated_lora_weight_ptrs, lora_params.permuted_fc2_lora_ranks,
        lora_params.permuted_fc1_gated_lora_bias_ranks, lora_params.adapter_expert_row_counts,
        lora_params.adapter_expert_row_starts, lora_params.no_adapter_expert_row_counts,
        lora_params.no_adapter_expert_row_starts, lora_params.fc1_gated_lora_has_missing_rows,
        stream);
    return false;
  }

  lora_params.permuted_lora_indices = reinterpret_cast<int32_t*>(
      advanceLoraMetadataWorkspace(metadata_ptr, expanded_num_rows * sizeof(int32_t)));

  if (is_gated_activation) {
    prepareGatedLoraMetadata<ScaleBiasType>(
        expanded_num_rows, num_rows, fc1_a_expert_stride, fc1_b_expert_stride, fc2_a_expert_stride,
        fc2_b_expert_stride, lora_params.num_lora_adapters, lora_params.max_low_rank,
        num_experts_per_node, expert_first_token_offset_, permuted_row_to_unpermuted_row_,
        lora_params.token_lora_indices, lora_params.token_lora_indices_is_int64,
        lora_params.fc1_lora_ranks, lora_params.fc1_lora_weight_ptrs, lora_params.fc2_lora_ranks,
        lora_params.fc2_lora_weight_ptrs, lora_params.gated_lora_ranks,
        lora_params.gated_lora_weight_ptrs, lora_params.permuted_fc1_lora_ranks,
        lora_params.permuted_fc1_lora_weight_ptrs, lora_params.permuted_fc2_lora_ranks,
        lora_params.permuted_fc2_lora_weight_ptrs, lora_params.permuted_gated_lora_ranks,
        lora_params.permuted_gated_lora_weight_ptrs, lora_params.permuted_lora_indices,
        lora_params.adapter_expert_row_counts, lora_params.adapter_expert_row_indices,
        lora_params.adapter_expert_row_capacity, lora_params.adapter_expert_row_starts,
        lora_params.rows_grouped_by_adapter, lora_params.no_adapter_expert_row_counts,
        lora_params.no_adapter_expert_row_indices, lora_params.no_adapter_expert_row_capacity,
        lora_params.no_adapter_expert_row_starts, lora_params.fc1_gated_lora_has_missing_rows,
        stream);
  } else {
    prepareLoraMetadata<ScaleBiasType>(
        expanded_num_rows, num_rows, fc1_a_expert_stride, fc1_b_expert_stride,
        lora_params.num_lora_adapters, lora_params.max_low_rank, num_experts_per_node,
        expert_first_token_offset_, permuted_row_to_unpermuted_row_, lora_params.token_lora_indices,
        lora_params.token_lora_indices_is_int64, lora_params.fc1_lora_ranks,
        lora_params.fc1_lora_weight_ptrs, lora_params.permuted_fc1_lora_ranks,
        lora_params.permuted_fc1_lora_weight_ptrs, lora_params.permuted_lora_indices, nullptr,
        nullptr, 0, nullptr, false, stream);
    prepareLoraMetadata<ScaleBiasType>(
        expanded_num_rows, num_rows, fc2_a_expert_stride, fc2_b_expert_stride,
        lora_params.num_lora_adapters, lora_params.max_low_rank, num_experts_per_node,
        expert_first_token_offset_, permuted_row_to_unpermuted_row_, lora_params.token_lora_indices,
        lora_params.token_lora_indices_is_int64, lora_params.fc2_lora_ranks,
        lora_params.fc2_lora_weight_ptrs, lora_params.permuted_fc2_lora_ranks,
        lora_params.permuted_fc2_lora_weight_ptrs, nullptr, lora_params.adapter_expert_row_counts,
        lora_params.adapter_expert_row_indices, lora_params.adapter_expert_row_capacity,
        lora_params.adapter_expert_row_starts, lora_params.rows_grouped_by_adapter, stream);
  }

  return false;
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
auto CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Enable>::loraFC1(int64_t expanded_num_rows, int64_t inter_size,
                                         int64_t hidden_size, int num_experts_per_node,
                                         int start_expert, int64_t const* num_valid_tokens_ptr,
                                         bool is_gated_activation,
                                         ScaleBiasType const* fc1_expert_biases,
                                         LoraParams& lora_params, float const* input_fp8_dequant,
                                         InputType const* original_input_activations,
                                         int64_t num_rows, cudaStream_t stream)
    -> ScaleBiasType const* {
  TLLM_CHECK_WITH_INFO(!act_fp4, "LoRA does not support FP4 activations");
  (void)start_expert;
  auto fc1_lora_impl = lora_params.fc1_lora_impl;
  int num_reqs = lora_params.num_reqs;

  ScaleBiasType *lora_gated_out = nullptr, *lora_fc1_result = nullptr;

  if (is_gated_activation) {
    lora_gated_out = lora_fc1_result_;
    lora_fc1_result = lora_fc1_result_ + expanded_num_rows * inter_size;
  } else {
    lora_fc1_result = lora_fc1_result_;
  }

  ScaleBiasType* input{};
  if constexpr (use_fp8) {
    TLLM_CHECK(lora_input_);
    bool const scale_is_dequant = true;
    dequantFP8<ScaleBiasType, T>(lora_input_, permuted_data_, num_valid_tokens_ptr, hidden_size,
                                 expanded_num_rows, input_fp8_dequant, scale_is_dequant, stream);
    sync_check_cuda_error(stream);
    input = lora_input_;
  } else if constexpr (!act_fp4) {
    TLLM_CHECK(!lora_input_);
    input = reinterpret_cast<ScaleBiasType*>(permuted_data_);
  }

  void* lora_workspace = lora_params.workspace;
  int64_t num_valid_tokens = expanded_num_rows;
  int64_t num_reqs_lora =
      std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));
  bool const use_gated_direct_layout = is_gated_activation && fc1_expert_biases == nullptr;

  if (use_gated_direct_layout) {
    TLLM_CHECK(lora_params.fc1_gated_lora_impl != nullptr);

    bool used_grouped_fc1_gated_output = false;
    if constexpr (std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
      used_grouped_fc1_gated_output =
          lora_params.max_low_rank == 16 && expanded_num_rows >= 1024 && (inter_size % 2 == 0);
      if (used_grouped_fc1_gated_output) {
        bool const used_tc_grouped_low_rank = lora_params.rows_grouped_by_adapter &&
                                              hidden_size % 16 == 0 &&
                                              lora_params.adapter_expert_row_counts != nullptr &&
                                              lora_params.adapter_expert_row_starts != nullptr;
        bool const used_tc_grouped_output = lora_params.rows_grouped_by_adapter &&
                                            (inter_size % 16 == 0) &&
                                            lora_params.adapter_expert_row_counts != nullptr &&
                                            lora_params.adapter_expert_row_starts != nullptr;
        if (used_tc_grouped_low_rank) {
          auto* low_rank_workspace = reinterpret_cast<__nv_bfloat16*>(lora_workspace);
          auto const* low_rank_input = reinterpret_cast<__nv_bfloat16 const*>(input);
          loraLowRank16Bf16TensorCoreGroupedDual(
              low_rank_input, hidden_size, expanded_num_rows, num_valid_tokens, hidden_size * 16,
              lora_params.num_lora_adapters, num_experts_per_node, lora_params.gated_lora_ranks,
              lora_params.gated_lora_weight_ptrs, lora_params.fc1_lora_ranks,
              lora_params.fc1_lora_weight_ptrs, lora_params.adapter_expert_row_counts,
              lora_params.adapter_expert_row_starts, low_rank_workspace,
              low_rank_workspace + expanded_num_rows * 16, stream);
        } else {
          TLLM_CHECK(lora_params.permuted_gated_lora_ranks != nullptr);
          TLLM_CHECK(lora_params.permuted_gated_lora_weight_ptrs != nullptr);
          ::tensorrt_llm::kernels::Lora_run_device_low_rank(
              lora_params.fc1_gated_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
              lora_params.permuted_gated_lora_ranks, lora_params.permuted_gated_lora_weight_ptrs, 0,
              lora_workspace, stream);
        }
        auto const* low_rank_workspace = reinterpret_cast<__nv_bfloat16 const*>(lora_workspace);
        if (used_tc_grouped_output) {
          loraOutput16Bf16TensorCoreGroupedDual(
              static_cast<__nv_bfloat16*>(lora_add_bias_),
              static_cast<__nv_bfloat16*>(lora_add_bias_ + inter_size), low_rank_workspace,
              low_rank_workspace + expanded_num_rows * 16, inter_size, inter_size * 2,
              expanded_num_rows,
              lora_params.num_lora_adapters, num_experts_per_node, lora_params.gated_lora_ranks,
              lora_params.gated_lora_weight_ptrs, lora_params.fc1_lora_ranks,
              lora_params.fc1_lora_weight_ptrs, lora_params.adapter_expert_row_counts,
              lora_params.adapter_expert_row_starts, lora_params.no_adapter_expert_row_counts,
              lora_params.no_adapter_expert_row_starts, stream,
              lora_params.permuted_fc1_gated_lora_bias_ranks == nullptr);
        } else {
          TLLM_CHECK(lora_params.permuted_gated_lora_weight_ptrs != nullptr);
          TLLM_CHECK(lora_params.permuted_fc1_lora_weight_ptrs != nullptr);
          fc1GatedLoraRank16Bf16GroupedOutput(
              static_cast<__nv_bfloat16*>(lora_add_bias_), low_rank_workspace,
              expert_first_token_offset_, expanded_num_rows, inter_size, num_experts_per_node,
              lora_params.num_lora_adapters, lora_params.gated_lora_ranks,
              lora_params.gated_lora_weight_ptrs, lora_params.fc1_lora_ranks,
              lora_params.fc1_lora_weight_ptrs, lora_params.permuted_gated_lora_weight_ptrs,
              lora_params.permuted_fc1_lora_weight_ptrs, lora_params.permuted_lora_indices,
              lora_params.adapter_expert_row_counts, lora_params.adapter_expert_row_indices,
              lora_params.adapter_expert_row_capacity, lora_params.no_adapter_expert_row_counts,
              lora_params.no_adapter_expert_row_indices, lora_params.no_adapter_expert_row_capacity,
              lora_params.adapter_expert_row_starts, lora_params.no_adapter_expert_row_starts,
              lora_params.rows_grouped_by_adapter, lora_params.fc1_gated_lora_has_missing_rows,
              stream);
        }
      }
    }

    if (!used_grouped_fc1_gated_output) {
      TLLM_CHECK(lora_params.permuted_gated_lora_ranks != nullptr);
      TLLM_CHECK(lora_params.permuted_gated_lora_weight_ptrs != nullptr);
      void* tmp_lora_outputs[] = {static_cast<void*>(lora_add_bias_),
                                  static_cast<void*>(lora_add_bias_ + inter_size)};
      int64_t const lora_output_strides[] = {2 * inter_size, 2 * inter_size};
      ::tensorrt_llm::kernels::Lora_run_device_strided_outputs(
          lora_params.fc1_gated_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
          lora_params.permuted_gated_lora_ranks, lora_params.permuted_gated_lora_weight_ptrs, 0,
          tmp_lora_outputs, lora_output_strides, lora_workspace, stream);
    }
  } else {
    void* tmp_lora_fc_result = static_cast<void*>(lora_fc1_result);
    ::tensorrt_llm::kernels::Lora_run_device(fc1_lora_impl.get(), num_valid_tokens, num_reqs_lora,
                                             input, lora_params.permuted_fc1_lora_ranks,
                                             lora_params.permuted_fc1_lora_weight_ptrs, 0,
                                             &tmp_lora_fc_result, lora_workspace, stream);
    if (is_gated_activation) {
      TLLM_CHECK(lora_params.permuted_gated_lora_ranks != nullptr);
      TLLM_CHECK(lora_params.permuted_gated_lora_weight_ptrs != nullptr);
      void* tmp_lora_gated_out = static_cast<void*>(lora_gated_out);
      ::tensorrt_llm::kernels::Lora_run_device(fc1_lora_impl.get(), num_valid_tokens, num_reqs_lora,
                                               input, lora_params.permuted_gated_lora_ranks,
                                               lora_params.permuted_gated_lora_weight_ptrs, 0,
                                               &tmp_lora_gated_out, lora_workspace, stream);
    }
  }

  // add bias and reorder
  if (fc1_expert_biases != nullptr) {
    loraAddBias(lora_add_bias_, lora_fc1_result_, fc1_expert_biases, num_valid_tokens_ptr,
                inter_size, permuted_token_selected_experts_, expanded_num_rows,
                is_gated_activation, stream);
    return lora_add_bias_;
  } else if (is_gated_activation) {
    if (!use_gated_direct_layout) {
      loraReorder(lora_add_bias_, lora_fc1_result_, num_valid_tokens_ptr, inter_size,
                  expanded_num_rows, stream);
    }
    return lora_add_bias_;
  } else {
    return lora_fc1_result_;
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Enable>::loraFC2(int64_t inter_size, int64_t hidden_size,
                                         int num_experts_per_node, int start_expert,
                                         int64_t const* num_valid_tokens_ptr, int64_t num_tokens,
                                         LoraParams& lora_params, float const* fc2_fp8_quant,
                                         cudaStream_t stream) {
  (void)start_expert;
  auto fc2_lora_impl = lora_params.fc2_lora_impl;
  int num_reqs = lora_params.num_reqs;

  ScaleBiasType* input{};
  if constexpr (use_fp8) {
    TLLM_CHECK(lora_input_);
    bool const scale_is_dequant = false;
    dequantFP8(lora_input_, fc1_result_, num_valid_tokens_ptr, inter_size, num_tokens,
               fc2_fp8_quant, scale_is_dequant, stream);
    sync_check_cuda_error(stream);
    input = lora_input_;
  } else if constexpr (!act_fp4) {
    TLLM_CHECK(!lora_input_);
    input = reinterpret_cast<ScaleBiasType*>(fc1_result_);
  }

  void* lora_workspace = lora_params.workspace;
  int64_t num_valid_tokens = num_tokens;
  void* tmp_lora_fc_result = static_cast<void*>(lora_fc2_result_);
  int64_t num_reqs_lora =
      std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));

  ::tensorrt_llm::kernels::Lora_run_device(fc2_lora_impl.get(), num_valid_tokens, num_reqs_lora,
                                           input, lora_params.permuted_fc2_lora_ranks,
                                           lora_params.permuted_fc2_lora_weight_ptrs, 0,
                                           &tmp_lora_fc_result, lora_workspace, stream);
  sync_check_cuda_error(stream);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Enable>::loraFC2LowRank(int64_t inter_size, int64_t hidden_size,
                                                int num_experts_per_node, int start_expert,
                                                int64_t const* num_valid_tokens_ptr,
                                                int64_t num_tokens, LoraParams& lora_params,
                                                float const* fc2_fp8_quant, cudaStream_t stream) {
  (void)hidden_size;
  (void)start_expert;
  auto fc2_lora_impl = lora_params.fc2_lora_impl;
  int num_reqs = lora_params.num_reqs;

  ScaleBiasType* input{};
  if constexpr (use_fp8) {
    TLLM_CHECK(lora_input_);
    bool const scale_is_dequant = false;
    dequantFP8(lora_input_, fc1_result_, num_valid_tokens_ptr, inter_size, num_tokens,
               fc2_fp8_quant, scale_is_dequant, stream);
    sync_check_cuda_error(stream);
    input = lora_input_;
  } else if constexpr (!act_fp4) {
    TLLM_CHECK(!lora_input_);
    input = reinterpret_cast<ScaleBiasType*>(fc1_result_);
  }

  int64_t num_valid_tokens = num_tokens;
  int64_t num_reqs_lora =
      std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));
  if constexpr (std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
    bool const used_tc_grouped_low_rank =
        lora_params.max_low_rank == 16 && lora_params.rows_grouped_by_adapter &&
        inter_size % 16 == 0 && lora_params.adapter_expert_row_counts != nullptr &&
        lora_params.adapter_expert_row_starts != nullptr;
    if (used_tc_grouped_low_rank) {
      loraLowRank16Bf16TensorCoreGrouped(
          reinterpret_cast<__nv_bfloat16 const*>(input), inter_size, num_tokens, num_valid_tokens,
          inter_size * 16, lora_params.num_lora_adapters, num_experts_per_node, 1,
          lora_params.fc2_lora_ranks, lora_params.fc2_lora_weight_ptrs,
          lora_params.adapter_expert_row_counts, lora_params.adapter_expert_row_starts,
          reinterpret_cast<__nv_bfloat16*>(lora_params.workspace), stream);
    } else {
      ::tensorrt_llm::kernels::Lora_run_device_low_rank(
          fc2_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
          lora_params.permuted_fc2_lora_ranks, lora_params.permuted_fc2_lora_weight_ptrs, 0,
          lora_params.workspace, stream);
    }
  } else {
    ::tensorrt_llm::kernels::Lora_run_device_low_rank(
        fc2_lora_impl.get(), num_valid_tokens, num_reqs_lora, input,
        lora_params.permuted_fc2_lora_ranks, lora_params.permuted_fc2_lora_weight_ptrs, 0,
        lora_params.workspace, stream);
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Enable>::loraFC2OutputAdd(int64_t hidden_size, int num_experts_per_node,
                                                  int64_t num_tokens, LoraParams& lora_params,
                                                  void* output, cudaStream_t stream) {
  if constexpr (std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
    TLLM_CHECK(lora_params.max_low_rank == 16);
    TLLM_CHECK((hidden_size % 2) == 0);
    auto const* low_rank_workspace = reinterpret_cast<__nv_bfloat16 const*>(lora_params.workspace);
    bool const used_tc_grouped_output = lora_params.rows_grouped_by_adapter &&
                                        (hidden_size % 16 == 0) &&
                                        lora_params.adapter_expert_row_counts != nullptr &&
                                        lora_params.adapter_expert_row_starts != nullptr;
    if (used_tc_grouped_output) {
      loraOutput16Bf16TensorCoreGrouped<true>(
          static_cast<__nv_bfloat16*>(output), low_rank_workspace, hidden_size, hidden_size,
          num_tokens, lora_params.num_lora_adapters, num_experts_per_node,
          lora_params.fc2_lora_ranks, lora_params.fc2_lora_weight_ptrs,
          lora_params.adapter_expert_row_counts, lora_params.adapter_expert_row_starts, stream);
    } else {
      fc2LoraRank16Bf16GroupedOutputAdd(
          static_cast<__nv_bfloat16*>(output), low_rank_workspace, expert_first_token_offset_,
          hidden_size, num_experts_per_node, lora_params.num_lora_adapters, lora_params.fc2_lora_ranks,
          lora_params.fc2_lora_weight_ptrs, lora_params.permuted_lora_indices,
          lora_params.adapter_expert_row_counts, lora_params.adapter_expert_row_indices,
          lora_params.adapter_expert_row_capacity, lora_params.adapter_expert_row_starts,
          lora_params.rows_grouped_by_adapter, stream);
    }
  } else {
    (void)hidden_size;
    (void)num_experts_per_node;
    (void)num_tokens;
    (void)lora_params;
    (void)output;
    (void)stream;
    TLLM_THROW("Split FC2 LoRA output-add is only implemented for BF16.");
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX,
                        Enable>::loraFC2Output(int64_t hidden_size, int num_experts_per_node,
                                               int64_t num_tokens, LoraParams& lora_params,
                                               void* output, cudaStream_t stream,
                                               int output_split_count, int output_split_index) {
  if constexpr (std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
    TLLM_CHECK(lora_params.max_low_rank == 16);
    TLLM_CHECK((hidden_size % 2) == 0);
    auto const* low_rank_workspace = reinterpret_cast<__nv_bfloat16 const*>(lora_params.workspace);
    bool const used_tc_grouped_output = lora_params.rows_grouped_by_adapter &&
                                        (hidden_size % 16 == 0) &&
                                        lora_params.adapter_expert_row_counts != nullptr &&
                                        lora_params.adapter_expert_row_starts != nullptr;
    if (used_tc_grouped_output) {
      loraOutput16Bf16TensorCoreGrouped<false>(
          static_cast<__nv_bfloat16*>(output), low_rank_workspace, hidden_size, hidden_size,
          num_tokens, lora_params.num_lora_adapters, num_experts_per_node,
          lora_params.fc2_lora_ranks, lora_params.fc2_lora_weight_ptrs,
          lora_params.adapter_expert_row_counts, lora_params.adapter_expert_row_starts, stream,
          output_split_count, output_split_index);
    } else {
      fc2LoraRank16Bf16GroupedOutput(
          static_cast<__nv_bfloat16*>(output), low_rank_workspace, expert_first_token_offset_,
          hidden_size, num_experts_per_node, lora_params.num_lora_adapters, lora_params.fc2_lora_ranks,
          lora_params.fc2_lora_weight_ptrs, lora_params.permuted_lora_indices,
          lora_params.adapter_expert_row_counts, lora_params.adapter_expert_row_indices,
          lora_params.adapter_expert_row_capacity, lora_params.adapter_expert_row_starts,
          lora_params.rows_grouped_by_adapter, stream);
    }
  } else {
    (void)hidden_size;
    (void)num_experts_per_node;
    (void)num_tokens;
    (void)lora_params;
    (void)output;
    (void)stream;
    TLLM_THROW("Split FC2 LoRA output materialization is only implemented for BF16.");
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Enable>::
    runMoe(void const* input_activations_void, void const* input_sf_void,
           bool const swizzled_input_sf, int const* token_selected_experts,
           float const* token_final_scales, void const* fc1_expert_weights_void,
           void const* fc1_expert_biases_void, ActivationParams fc1_activation_type,
           void const* fc2_expert_weights_void, void const* fc2_expert_biases_void,
           QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size,
           int64_t const unpadded_hidden_size, int64_t const inter_size, int const full_num_experts,
           int const experts_per_token, char* workspace_ptr, void* final_output_void,
           int* unpermuted_row_to_permuted_row, MOEParallelismConfig parallelism_config,
           bool const enable_alltoall, bool use_lora, LoraParams& lora_params,
           bool use_deepseek_fp8_block_scale, bool use_mxfp8_act_scaling, bool min_latency_mode,
           MoeMinLatencyParams& min_latency_params, bool enable_pdl, cudaStream_t stream) {
  static constexpr bool int_scales_required = std::is_same<WeightType, uint8_t>::value ||
                                              std::is_same<WeightType, cutlass::uint4b_t>::value ||
                                              use_wfp4a16;
  static constexpr bool fp8_scales_required = std::is_same<WeightType, __nv_fp8_e4m3>::value ||
                                              std::is_same<WeightType, __nv_fp8_e5m2>::value;

  auto const* input_activations = static_cast<InputType const*>(input_activations_void);
  auto const* input_sf =
      input_sf_void
          ? reinterpret_cast<TmaWarpSpecializedGroupedGemmInput::ElementSF const*>(input_sf_void)
          : nullptr;
  auto const* fc1_expert_weights = static_cast<WeightType const*>(fc1_expert_weights_void);
  auto const* fc1_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc1_expert_biases_void);
  auto const* fc2_expert_weights = static_cast<WeightType const*>(fc2_expert_weights_void);
  auto const* fc1_int_scales =
      reinterpret_cast<ScaleBiasType const*>(quant_params.wo.fc1_weight_scales);
  auto const* fc2_int_scales =
      reinterpret_cast<ScaleBiasType const*>(quant_params.wo.fc2_weight_scales);

  auto const* fc1_fp8_dequant = quant_params.fp8.dequant_fc1;
  auto const* fc2_fp8_quant = quant_params.fp8.quant_fc2;
  auto const* fc2_fp8_dequant = quant_params.fp8.dequant_fc2;
  // WMXFP8AMXFP8 keeps dequant scales in mxfp8_mxfp8; remap these shared FP8 pointers so
  // downstream GEMM/TMA setup can reuse the common FP8 code path.
  if (fp8_scales_required && use_mxfp8_act_scaling) {
    TLLM_CHECK_WITH_INFO(quant_params.mxfp8_mxfp8.fc1.weight_block_scale != nullptr,
                         "WMXFP8AMXFP8 requires FC1 weight_block_scale to be non-null");
    fc1_fp8_dequant = quant_params.mxfp8_mxfp8.fc1.global_scale;
    fc2_fp8_dequant = quant_params.mxfp8_mxfp8.fc2.global_scale;
  }
  auto const* input_fp8_dequant = quant_params.fp8.dequant_input;

  auto const* fc2_wfp4afp8_quant_scale = quant_params.fp8_mxfp4.fc2.act_global_scale;

  auto const* fc2_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc2_expert_biases_void);
  auto* final_output = static_cast<OutputType*>(final_output_void);
  float const* token_topk_unpermuted_scales = token_final_scales;
  // Note: getDeepSeekBlockScaleGemmRunner will do a sanity check on our template parameters.
  auto* blockscale_gemm_runner =
      use_deepseek_fp8_block_scale ? getDeepSeekBlockScaleGemmRunner() : nullptr;

  TLLM_CHECK(input_activations);
  TLLM_CHECK(token_selected_experts);
  TLLM_CHECK(fc1_expert_weights);
  TLLM_CHECK(fc2_expert_weights);
  TLLM_CHECK(workspace_ptr);
  // TLLM_CHECK(token_topk_unpermuted_scales);
  TLLM_CHECK(unpermuted_row_to_permuted_row);
  TLLM_CHECK(full_num_experts % parallelism_config.ep_size == 0);
  TLLM_CHECK(full_num_experts % parallelism_config.cluster_size == 0);

  if (quant_params.mxfp8_mxfp4.fc1.weight_block_scale ||
      quant_params.mxfp8_mxfp8.fc1.weight_block_scale) {
    TLLM_CHECK_WITH_INFO(
        hidden_size % (64 * 8 / sizeof_bits<WeightType>::value) == 0,
        "Hidden size %d does not meet minimum alignment requirements for MXFP8 MOE GEMM %d",
        (int)hidden_size, (int)(64 * 8 / sizeof_bits<WeightType>::value));
    TLLM_CHECK_WITH_INFO(
        inter_size % (64 * 8 / sizeof_bits<WeightType>::value) == 0,
        "Inter size %d does not meet minimum alignment requirements for MXFP8 MOE GEMM %d",
        (int)inter_size, (int)(64 * 8 / sizeof_bits<WeightType>::value));
  } else {
    // For NoSmem epilogue schedule, we need to align the output of the GEMM to 256 bits, for gated
    // activation this is automatic if the usual alignment requirement is met
    if (gemm1_config_->epilogue_schedule == cutlass_extensions::EpilogueScheduleType::NO_SMEM &&
        !isGatedActivation(fc1_activation_type)) {
      TLLM_CHECK_WITH_INFO(
          inter_size % (256 / sizeof_bits<WeightType>::value) == 0,
          "Inter size %d does not meet minimum alignment requirements for MOE GEMM %d",
          (int)inter_size, (int)(256 / sizeof_bits<WeightType>::value));
    }

    if (gemm2_config_->epilogue_schedule == cutlass_extensions::EpilogueScheduleType::NO_SMEM) {
      TLLM_CHECK_WITH_INFO(
          gemm2_config_->epilogue_fusion_type !=
              cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE,
          "Got NoSmem epilogue schedule, which is not supported for finalize fusion");
      TLLM_CHECK_WITH_INFO(
          hidden_size % (256 / sizeof_bits<WeightType>::value) == 0,
          "Hidden size %d does not meet minimum alignment requirements for MOE GEMM %d",
          (int)hidden_size, (int)(256 / sizeof_bits<WeightType>::value));
    }

    // Require at least 128 bits of alignment for MOE GEMM
    TLLM_CHECK_WITH_INFO(
        hidden_size % (128 / sizeof_bits<WeightType>::value) == 0,
        "Hidden size %d does not meet minimum alignment requirements for MOE GEMM %d",
        (int)hidden_size, (int)(128 / sizeof_bits<WeightType>::value));
    TLLM_CHECK_WITH_INFO(
        inter_size % (128 / sizeof_bits<WeightType>::value) == 0,
        "Inter size %d does not meet minimum alignment requirements for MOE GEMM %d",
        (int)inter_size, (int)(128 / sizeof_bits<WeightType>::value));
  }

  // These values must fit into an int for building the source maps
  TLLM_CHECK_WITH_INFO(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
  TLLM_CHECK_WITH_INFO(num_rows * full_num_experts <= std::numeric_limits<int>::max(),
                       "Number of rows * num_experts is too large");
  TLLM_CHECK_WITH_INFO(experts_per_token * full_num_experts <= std::numeric_limits<int>::max(),
                       "experts_per_token * num_experts is too large");

  TLLM_CHECK_WITH_INFO(gemm1_config_, "MOE GEMM1 Config is not set");
  TLLM_CHECK_WITH_INFO(gemm2_config_, "MOE GEMM2 Config is not set");

  TLLM_CHECK_WITH_INFO(!use_lora || !act_fp4, "MOE does not support LoRA with FP4 model");

  if (int_scales_required) {
    if (!(quant_params.groupwise.fc1.weight_scales && quant_params.groupwise.fc2.weight_scales)) {
      TLLM_CHECK_WITH_INFO(fc1_int_scales != nullptr,
                           "Weight scales expected but scale for first matmul is a null pointer");
      TLLM_CHECK_WITH_INFO(fc2_int_scales != nullptr,
                           "Weight scales expected but scale for second matmul is a null pointer");
    }
    TLLM_CHECK_WITH_INFO(
        fc1_fp8_dequant == nullptr && fc2_fp8_quant == nullptr && fc2_fp8_dequant == nullptr,
        "FP8 scales are provided for integer quantization");
  } else if (fp8_scales_required && use_mxfp8_act_scaling) {
    TLLM_CHECK_WITH_INFO(
        fc1_fp8_dequant != nullptr,
        "WMXFP8AMXFP8 scales expected but dequant scale for FC1 is a null pointer");
    TLLM_CHECK_WITH_INFO(
        fc2_fp8_dequant != nullptr,
        "WMXFP8AMXFP8 scales expected but dequant scale for FC2 is a null pointer");
    TLLM_CHECK_WITH_INFO(fc1_int_scales == nullptr && fc2_int_scales == nullptr,
                         "Integer scales are provided for WMXFP8AMXFP8 quantization");
  } else if (fp8_scales_required && !use_deepseek_fp8_block_scale) {
    TLLM_CHECK_WITH_INFO(fc1_fp8_dequant != nullptr,
                         "FP8 scales expected but dequant scale for FC1 is a null pointer");
    TLLM_CHECK_WITH_INFO(fc2_fp8_quant != nullptr,
                         "FP8 scales expected but quant scale for FC2 is a null pointer");
    TLLM_CHECK_WITH_INFO(fc2_fp8_dequant != nullptr,
                         "FP8 scales expected but quant scale for FC2 is a null pointer");

    TLLM_CHECK_WITH_INFO(fc1_int_scales == nullptr && fc2_int_scales == nullptr,
                         "Integer scales are provided for FP8 quantization");
  } else if (use_lora && use_fp8) {
    TLLM_CHECK_WITH_INFO(input_fp8_dequant != nullptr,
                         "FP8 scales expected but quant scale for input is a null pointer");
  } else {
    TLLM_CHECK_WITH_INFO(fc1_int_scales == nullptr,
                         "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC1");
    TLLM_CHECK_WITH_INFO(fc2_int_scales == nullptr,
                         "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC2");
    TLLM_CHECK_WITH_INFO(
        fc1_fp8_dequant == nullptr,
        "Scales are ignored for fp32/fp16/bf16 but received dequant scale for FC1");
    TLLM_CHECK_WITH_INFO(fc2_fp8_quant == nullptr,
                         "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
    TLLM_CHECK_WITH_INFO(fc2_fp8_dequant == nullptr,
                         "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
  }

  bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales &&
                 !use_wfp4a16;
  int const num_experts_per_node = full_num_experts / parallelism_config.ep_size;

  configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts_per_node,
                  experts_per_token, fc1_activation_type, parallelism_config, use_lora,
                  use_deepseek_fp8_block_scale, use_mxfp8_act_scaling, min_latency_mode, use_awq);

  int start_expert = num_experts_per_node * parallelism_config.ep_rank;
  int end_expert = start_expert + num_experts_per_node;

  bool const needs_num_valid = parallelism_config.ep_size > 1;
  int64_t const* num_valid_tokens_ptr =
      needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;

  auto expanded_num_rows = num_rows * experts_per_token;

  bool split_fc2_lora_output_add = false;
  if constexpr (std::is_same_v<UnfusedGemmOutputType, __nv_bfloat16> &&
                std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
    split_fc2_lora_output_add = use_lora && expanded_num_rows >= 1024 &&
                                lora_params.max_low_rank == 16 && (hidden_size % 2 == 0);
  }
  bool fuse_fc2_lora_finalize = false;
  if constexpr (std::is_same_v<OutputType, __nv_bfloat16> &&
                std::is_same_v<UnfusedGemmOutputType, __nv_bfloat16> &&
                std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
    fuse_fc2_lora_finalize =
        split_fc2_lora_output_add && num_rows >= 32768 && hidden_size == unpadded_hidden_size &&
        parallelism_config.tp_size == 1 && parallelism_config.tp_rank == 0 &&
        parallelism_config.ep_size == 1 && parallelism_config.ep_rank == 0 && !enable_alltoall &&
        !use_deepseek_fp8_block_scale && !use_awq && !use_w4_groupwise;
  }
  bool const overlap_fc2_lora_with_gemm2 = split_fc2_lora_output_add;
  if (min_latency_mode) {
    TLLM_CHECK(use_lora == false);
    TLLM_CHECK(use_awq == false);
    TLLM_CHECK(use_fp4 == true);

    buildMinLatencyActiveExpertMaps(
        min_latency_params.num_active_experts_per_node, min_latency_params.experts_to_token_score,
        min_latency_params.active_expert_global_ids, expert_first_token_offset_,
        token_selected_experts, token_final_scales, num_rows, experts_per_token, start_expert,
        end_expert, num_experts_per_node, parallelism_config.cluster_rank,
        parallelism_config.cluster_size, full_num_experts, enable_pdl, stream);
    sync_check_cuda_error(stream);

    auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(
        num_rows, expanded_num_rows, fc1_activation_type, hidden_size, unpadded_hidden_size,
        inter_size, num_experts_per_node, input_activations_void, input_sf, final_output,
        fc1_expert_weights, fc2_expert_weights, quant_params, fc1_expert_biases, fc2_expert_biases,
        min_latency_mode, min_latency_params, use_lora, start_expert, parallelism_config, false,
        enable_pdl, stream);

    // todo: input_activations_void should be nvfp4, waiting for yuxian's mr ready
    Self::gemm1(moe_gemm_runner_, blockscale_gemm_runner,
                reinterpret_cast<T const*>(input_activations_void), fc1_result_, glu_inter_result_,
                expert_first_token_offset_, gemm1_tma_ws_input, fc1_expert_weights,
                fc1_expert_biases, num_valid_tokens_ptr, fc1_int_scales, fc1_fp8_dequant,
                use_wfp4afp8 ? fc2_wfp4afp8_quant_scale : fc2_fp8_quant,
                input_sf /*input fp4 scale or expanded fp4 scale*/, fc2_fp4_act_scale_,
                quant_params, num_rows, expanded_num_rows, hidden_size, inter_size,
                num_experts_per_node, fc1_activation_type, alpha_scale_ptr_array_fc1_, !use_lora,
                stream, *gemm1_config_, true, min_latency_params.num_active_experts_per_node,
                min_latency_params.active_expert_global_ids, enable_pdl, nullptr);
    sync_check_cuda_error(stream);

    auto gemm2_input =
        applyPrequantScale(smoothed_act_, fc1_result_, quant_params.groupwise.fc2.act_scales,
                           num_valid_tokens_ptr, expanded_num_rows, inter_size, use_awq, stream);
    Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input, final_output, nullptr,
                expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights,
                fc2_expert_biases, fc2_int_scales, fc2_fp8_dequant, fc2_fp4_act_scale_,
                quant_params, token_topk_unpermuted_scales, permuted_token_final_scales_,
                unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row_,
                token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows,
                hidden_size, unpadded_hidden_size, inter_size, num_experts_per_node,
                experts_per_token, alpha_scale_ptr_array_fc2_, use_lora, lora_fc2_result_, nullptr,
                stream, parallelism_config, enable_alltoall, *gemm2_config_, true,
                min_latency_params.num_active_experts_per_node,
                min_latency_params.active_expert_global_ids, false, enable_pdl);
    sync_check_cuda_error(stream);
  } else {
    bool fused_prologue_result = false;
    if (!use_w4_groupwise) {
      // WAR: fusedBuildExpertMapsSortFirstToken kernel will lead to illegal memory access for
      // W4AFP8
      fused_prologue_result = fusedBuildExpertMapsSortFirstToken(
          token_selected_experts, permuted_row_to_unpermuted_row_, unpermuted_row_to_permuted_row,
          expert_first_token_offset_, num_rows, num_experts_per_node, experts_per_token,
          start_expert, end_expert, enable_pdl, stream);
    }

    if (!fused_prologue_result) {
      TLLM_LOG_TRACE("Falling back to unfused prologue");
      threeStepBuildExpertMapsSortFirstToken(
          token_selected_experts, permuted_token_selected_experts_, permuted_row_to_unpermuted_row_,
          unpermuted_row_to_permuted_row, expert_first_token_offset_, blocked_expert_counts_,
          blocked_expert_counts_cumsum_, blocked_row_to_unpermuted_row_, num_rows,
          num_experts_per_node, experts_per_token, start_expert, enable_pdl, stream);
    }

    if (use_lora && lora_params.token_lora_indices != nullptr && num_rows >= 32768) {
      int64_t const num_tokens_per_block = computeNumTokensPerBlock(num_rows, num_experts_per_node);
      int64_t const num_blocks_per_seq =
          tensorrt_llm::common::ceilDiv(num_rows, num_tokens_per_block);
      int64_t const routing_bucket_capacity = num_experts_per_node * num_blocks_per_seq;
      lora_params.rows_grouped_by_adapter = regroupExpertRowsByLoraAdapter(
          permuted_token_selected_experts_, permuted_row_to_unpermuted_row_,
          unpermuted_row_to_permuted_row, expert_first_token_offset_,
          lora_params.token_lora_indices, lora_params.token_lora_indices_is_int64,
          blocked_expert_counts_, blocked_expert_counts_cumsum_, blocked_row_to_unpermuted_row_,
          num_rows, expanded_num_rows, num_experts_per_node, lora_params.num_lora_adapters,
          routing_bucket_capacity, false, stream);
      if (lora_params.rows_grouped_by_adapter) {
        permuted_row_to_unpermuted_row_ = blocked_row_to_unpermuted_row_;
      }
    }

    sync_check_cuda_error(stream);

    std::function<void()> deferred_fc1_lora_work;
    bool is_gated_activation = isGatedActivation(fc1_activation_type);
    cudaEvent_t fc1_lora_ready_event = nullptr;
    bool overlap_fc1_lora_with_gemm1 = false;
    if constexpr (std::is_same_v<T, __nv_bfloat16> && std::is_same_v<InputType, __nv_bfloat16> &&
                  std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
		      overlap_fc1_lora_with_gemm1 =
		          use_lora && is_gated_activation && fc1_expert_biases == nullptr &&
	          lora_params.max_low_rank == 16 && expanded_num_rows >= 1024 &&
	          lora_params.rows_grouped_by_adapter && !use_deepseek_fp8_block_scale && !use_awq &&
	          !use_w4_groupwise && gemm1_config_->swap_ab &&
	          moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_);
	    }

    // Only NVFP4xNVFP4 supports FC1 per-expert act scale
    bool use_per_expert_act_scale = use_fp4 ? quant_params.fp4.fc1.use_per_expert_act_scale : false;
    T* gemm1_input_expand =
        use_w4afp8 ? reinterpret_cast<T*>(smoothed_act_) : reinterpret_cast<T*>(permuted_data_);
    expandInputRowsKernelLauncher(
        input_activations, gemm1_input_expand, token_topk_unpermuted_scales,
        permuted_token_final_scales_, permuted_row_to_unpermuted_row_,
        unpermuted_row_to_permuted_row, token_selected_experts, num_rows, hidden_size,
        experts_per_token, num_experts_per_node, start_expert, quant_params,
        use_per_expert_act_scale, expert_first_token_offset_, fc1_fp4_act_scale_, input_sf,
        swizzled_input_sf,
        (use_w4afp8 && !use_fp8_input) ? quant_params.groupwise.fc1.act_scales : nullptr,
        enable_pdl, stream);
    auto const* gemm1_input = gemm1_input_expand;

    sync_check_cuda_error(stream);

    if (use_lora) {
      bool const use_grouped_lora_metadata_fast_path =
          lora_params.rows_grouped_by_adapter && is_gated_activation &&
          fc1_expert_biases == nullptr && split_fc2_lora_output_add;
      bool all_token_without_lora =
          setupLoraWorkspace(expanded_num_rows, num_rows, inter_size, hidden_size, start_expert,
                             is_gated_activation, num_experts_per_node, needs_num_valid,
                             lora_params, use_grouped_lora_metadata_fast_path, stream);

      if (!all_token_without_lora) {
        if (overlap_fc1_lora_with_gemm1) {
          auto const* fc1_lora_base_biases = fc1_expert_biases;
          fc1_expert_biases = lora_add_bias_;
          check_cuda_error(cudaEventRecord(lora_start_event_, stream));
          deferred_fc1_lora_work = [&, fc1_lora_base_biases]() {
            check_cuda_error(cudaStreamWaitEvent(lora_stream_, lora_start_event_, 0));
            (void)loraFC1(expanded_num_rows, inter_size, hidden_size, num_experts_per_node,
                          start_expert, num_valid_tokens_ptr, is_gated_activation,
                          fc1_lora_base_biases, lora_params, input_fp8_dequant, input_activations,
                          num_rows, lora_stream_);
            check_cuda_error(cudaEventRecord(lora_ready_event_, lora_stream_));
          };
          fc1_lora_ready_event = lora_ready_event_;
        } else {
          fc1_expert_biases =
              loraFC1(expanded_num_rows, inter_size, hidden_size, num_experts_per_node,
                      start_expert, num_valid_tokens_ptr, is_gated_activation, fc1_expert_biases,
                      lora_params, input_fp8_dequant, input_activations, num_rows, stream);
          sync_check_cuda_error(stream);
        }
      } else {
        use_lora = false;
        split_fc2_lora_output_add = false;
        fuse_fc2_lora_finalize = false;
        fc1_lora_ready_event = nullptr;
      }
    }

    auto [gemm1_tma_ws_input, gemm2_tma_ws_input] = setupTmaWarpSpecializedInputs(
        num_rows, expanded_num_rows, fc1_activation_type, hidden_size, unpadded_hidden_size,
        inter_size, num_experts_per_node, input_activations_void, input_sf, final_output,
        fc1_expert_weights, fc2_expert_weights, quant_params, fc1_expert_biases, fc2_expert_biases,
        min_latency_mode, min_latency_params, use_lora, start_expert, parallelism_config,
        fuse_fc2_lora_finalize, enable_pdl, stream);

    if constexpr (!use_w4afp8) {
      gemm1_input =
          applyPrequantScale(smoothed_act_, permuted_data_, quant_params.groupwise.fc1.act_scales,
                             num_valid_tokens_ptr, expanded_num_rows, hidden_size, use_awq, stream);
    }
    sync_check_cuda_error(stream);
    Self::gemm1(moe_gemm_runner_, blockscale_gemm_runner, gemm1_input, fc1_result_,
                glu_inter_result_, expert_first_token_offset_, gemm1_tma_ws_input,
                fc1_expert_weights, fc1_expert_biases, num_valid_tokens_ptr, fc1_int_scales,
                fc1_fp8_dequant, use_wfp4afp8 ? fc2_wfp4afp8_quant_scale : fc2_fp8_quant,
                fc1_fp4_act_scale_, fc2_fp4_act_scale_, quant_params, num_rows, expanded_num_rows,
                hidden_size, inter_size, num_experts_per_node, fc1_activation_type,
                alpha_scale_ptr_array_fc1_, !use_lora, stream, *gemm1_config_, false, nullptr,
                nullptr, enable_pdl, fc1_lora_ready_event,
                lora_params.permuted_fc1_gated_lora_bias_ranks,
                deferred_fc1_lora_work ? &deferred_fc1_lora_work : nullptr);
    sync_check_cuda_error(stream);

    auto gemm2_input =
        applyPrequantScale(smoothed_act_, fc1_result_, quant_params.groupwise.fc2.act_scales,
                           num_valid_tokens_ptr, expanded_num_rows, inter_size, use_awq, stream);
    sync_check_cuda_error(stream);

    auto launch_split_fc2_lora_low_rank = [&](cudaStream_t fc2_lora_stream) {
      loraFC2LowRank(inter_size, hidden_size, num_experts_per_node, start_expert,
                     num_valid_tokens_ptr, expanded_num_rows, lora_params, fc2_fp8_quant,
                     fc2_lora_stream);
    };
    auto launch_split_fc2_lora_output = [&](cudaStream_t fc2_lora_stream) {
      if (fuse_fc2_lora_finalize) {
        loraFC2Output(hidden_size, num_experts_per_node, expanded_num_rows, lora_params,
                      lora_fc2_result_, fc2_lora_stream);
      }
    };
    auto launch_split_fc2_lora = [&](cudaStream_t fc2_lora_stream) {
      launch_split_fc2_lora_low_rank(fc2_lora_stream);
      launch_split_fc2_lora_output(fc2_lora_stream);
    };

    bool queued_overlapped_fc2_lora = false;
    if (use_lora) {
      if (split_fc2_lora_output_add) {
        if (overlap_fc2_lora_with_gemm2) {
          check_cuda_error(cudaEventRecord(lora_start_event_, stream));
          check_cuda_error(cudaStreamWaitEvent(lora_stream_, lora_start_event_, 0));
          launch_split_fc2_lora(lora_stream_);
          check_cuda_error(cudaEventRecord(lora_fc2_ready_event_, lora_stream_));
          queued_overlapped_fc2_lora = true;
        } else {
          launch_split_fc2_lora(stream);
        }
      } else {
        loraFC2(inter_size, hidden_size, num_experts_per_node, start_expert, num_valid_tokens_ptr,
                expanded_num_rows, lora_params, fc2_fp8_quant, stream);
      }
      sync_check_cuda_error(stream);
    }

    Self::gemm2(moe_gemm_runner_, blockscale_gemm_runner, gemm2_input, fc2_result_, final_output,
                expert_first_token_offset_, gemm2_tma_ws_input, fc2_expert_weights,
                fc2_expert_biases, fc2_int_scales, fc2_fp8_dequant, fc2_fp4_act_scale_,
                quant_params, token_topk_unpermuted_scales, permuted_token_final_scales_,
                unpermuted_row_to_permuted_row, permuted_row_to_unpermuted_row_,
                token_selected_experts, num_valid_tokens_ptr, num_rows, expanded_num_rows,
                hidden_size, unpadded_hidden_size, inter_size, num_experts_per_node,
                experts_per_token, alpha_scale_ptr_array_fc2_,
                split_fc2_lora_output_add ? false : use_lora, lora_fc2_result_,
                lora_params.permuted_fc2_lora_ranks, stream, parallelism_config, enable_alltoall,
                *gemm2_config_, false, nullptr, nullptr, split_fc2_lora_output_add, enable_pdl);
    sync_check_cuda_error(stream);

    if (split_fc2_lora_output_add) {
      bool waited_for_fc2_lora = false;
      if (overlap_fc2_lora_with_gemm2) {
        if (!queued_overlapped_fc2_lora) {
          check_cuda_error(cudaStreamWaitEvent(lora_stream_, lora_start_event_, 0));
          launch_split_fc2_lora(lora_stream_);
          check_cuda_error(cudaEventRecord(lora_fc2_ready_event_, lora_stream_));
        }
      }
      if (fuse_fc2_lora_finalize) {
        bool const used_base_finalize_fusion =
            gemm2_tma_ws_input.fusion ==
            TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
        if (used_base_finalize_fusion) {
          if (overlap_fc2_lora_with_gemm2 && !waited_for_fc2_lora) {
            check_cuda_error(cudaStreamWaitEvent(stream, lora_fc2_ready_event_, 0));
            waited_for_fc2_lora = true;
          }
          addFinalizedLoraToOutputKernelLauncher<OutputType, ScaleBiasType>(
              static_cast<ScaleBiasType const*>(lora_fc2_result_),
              lora_params.permuted_fc2_lora_ranks, final_output, token_topk_unpermuted_scales,
              unpermuted_row_to_permuted_row, token_selected_experts, num_rows, hidden_size,
              unpadded_hidden_size, experts_per_token, num_experts_per_node, parallelism_config,
              enable_pdl, stream);
        } else {
          bool used_token_lora_finalize = false;
          if constexpr (std::is_same_v<OutputType, __nv_bfloat16> &&
                        std::is_same_v<UnfusedGemmOutputType, __nv_bfloat16> &&
                        std::is_same_v<ScaleBiasType, __nv_bfloat16>) {
            used_token_lora_finalize = fc2_expert_biases == nullptr &&
                                       lora_params.token_lora_indices != nullptr &&
                                       lora_params.fc2_lora_ranks != nullptr &&
                                       lora_params.fc2_lora_weight_ptrs != nullptr &&
                                       lora_params.num_lora_adapters > 0 && experts_per_token <= 16;
            if (used_token_lora_finalize) {
              if (lora_params.token_lora_indices_is_int64) {
                if (overlap_fc2_lora_with_gemm2 && !waited_for_fc2_lora) {
                  check_cuda_error(cudaStreamWaitEvent(stream, lora_fc2_ready_event_, 0));
                  waited_for_fc2_lora = true;
                }
                finalizeMoeRoutingWithTokenLoraKernelLauncher<OutputType, UnfusedGemmOutputType,
                                                              ScaleBiasType, int64_t>(
                    static_cast<UnfusedGemmOutputType const*>(fc2_result_),
                    static_cast<ScaleBiasType const*>(lora_fc2_result_), final_output,
                    token_topk_unpermuted_scales, unpermuted_row_to_permuted_row,
                    token_selected_experts,
                    static_cast<int64_t const*>(lora_params.token_lora_indices),
                    lora_params.fc2_lora_ranks, lora_params.fc2_lora_weight_ptrs, num_rows,
                    hidden_size, unpadded_hidden_size, experts_per_token, num_experts_per_node,
                    parallelism_config, lora_params.num_lora_adapters, enable_pdl, stream);
              } else {
                if (overlap_fc2_lora_with_gemm2 && !waited_for_fc2_lora) {
                  check_cuda_error(cudaStreamWaitEvent(stream, lora_fc2_ready_event_, 0));
                  waited_for_fc2_lora = true;
                }
                finalizeMoeRoutingWithTokenLoraKernelLauncher<OutputType, UnfusedGemmOutputType,
                                                              ScaleBiasType, int32_t>(
                    static_cast<UnfusedGemmOutputType const*>(fc2_result_),
                    static_cast<ScaleBiasType const*>(lora_fc2_result_), final_output,
                    token_topk_unpermuted_scales, unpermuted_row_to_permuted_row,
                    token_selected_experts,
                    static_cast<int32_t const*>(lora_params.token_lora_indices),
                    lora_params.fc2_lora_ranks, lora_params.fc2_lora_weight_ptrs, num_rows,
                    hidden_size, unpadded_hidden_size, experts_per_token, num_experts_per_node,
                    parallelism_config, lora_params.num_lora_adapters, enable_pdl, stream);
              }
            }
          }
          if (!used_token_lora_finalize) {
            if (overlap_fc2_lora_with_gemm2 && !waited_for_fc2_lora) {
              check_cuda_error(cudaStreamWaitEvent(stream, lora_fc2_ready_event_, 0));
              waited_for_fc2_lora = true;
            }
            finalizeMoeRoutingWithLoraKernelLauncher<OutputType, UnfusedGemmOutputType,
                                                     ScaleBiasType>(
                static_cast<UnfusedGemmOutputType const*>(fc2_result_),
                static_cast<ScaleBiasType const*>(lora_fc2_result_),
                lora_params.permuted_fc2_lora_ranks, final_output, fc2_expert_biases,
                token_topk_unpermuted_scales, unpermuted_row_to_permuted_row,
                token_selected_experts, num_rows, hidden_size, unpadded_hidden_size,
                experts_per_token, num_experts_per_node, parallelism_config, enable_pdl, stream);
          }
        }
      } else {
        loraFC2OutputAdd(hidden_size, num_experts_per_node, expanded_num_rows, lora_params,
                         fc2_result_, stream);
        sync_check_cuda_error(stream);
        finalizeMoeRoutingKernelLauncher<OutputType, UnfusedGemmOutputType>(
            static_cast<UnfusedGemmOutputType const*>(fc2_result_), final_output, fc2_expert_biases,
            token_topk_unpermuted_scales, unpermuted_row_to_permuted_row,
            permuted_row_to_unpermuted_row_, token_selected_experts, expert_first_token_offset_,
            num_rows, hidden_size, unpadded_hidden_size, experts_per_token, num_experts_per_node,
            parallelism_config, enable_alltoall, enable_pdl, stream);
      }
      sync_check_cuda_error(stream);
    }
  }
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Enable>::
    computeStridesTmaWarpSpecialized(
        int64_t const* expert_first_token_offset, TmaWarpSpecializedGroupedGemmInput layout_info1,
        TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens,
        int64_t expanded_num_tokens, int64_t gemm1_n, int64_t gemm1_k, int64_t gemm2_n,
        int64_t gemm2_k, int const num_experts_per_node, T const* gemm1_in, T const* gemm2_in,
        WeightType const* weights1, WeightType const* weights2, float const* fp8_dequant1,
        float const* fp8_dequant2,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat1,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fp4_act_flat2,
        QuantParams quant_params, ScaleBiasType const* bias1, ScaleBiasType const* bias2,
        UnfusedGemmOutputType* gemm1_output, UnfusedGemmOutputType* gemm2_output,
        float const* router_scales, int const* permuted_row_to_unpermuted_row, bool enable_pdl,
        cudaStream_t stream) {
  // Always nullptr
  layout_info1.ptr_c = nullptr;
  layout_info1.stride_c = nullptr;
  layout_info2.ptr_c = nullptr;
  layout_info2.stride_c = nullptr;

  layout_info1.fused_finalize_epilogue.ptr_bias = nullptr;
  if (!bias2) {
    layout_info2.fused_finalize_epilogue.ptr_bias = nullptr;
  }

  auto alpha_scale_flat1 = use_fp4        ? quant_params.fp4.fc1.global_scale
                           : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc1.global_scale
                           : use_fp8      ? fp8_dequant1
                                          : nullptr;
  auto alpha_scale_flat2 = use_fp4        ? quant_params.fp4.fc2.global_scale
                           : use_wfp4afp8 ? quant_params.fp8_mxfp4.fc2.global_scale
                           : use_fp8      ? fp8_dequant2
                                          : nullptr;
  if (!alpha_scale_flat1 && !alpha_scale_flat2) {
    layout_info1.alpha_scale_ptr_array = nullptr;
    layout_info2.alpha_scale_ptr_array = nullptr;
  }

  layout_info1.int4_groupwise_params.enabled = use_w4_groupwise;
  layout_info2.int4_groupwise_params.enabled = use_w4_groupwise;
  layout_info1.int4_groupwise_params.use_wfp4a16 = use_wfp4a16;
  layout_info2.int4_groupwise_params.use_wfp4a16 = use_wfp4a16;

  layout_info1.fpX_block_scaling_type = getScalingType();
  layout_info2.fpX_block_scaling_type = getScalingType();

  // Use a smaller block size to spread work across multiple SMs. Each thread handles one expert,
  // so we only need num_experts_per_node threads total. With 1 warp per block, 128 experts
  // yields 4 blocks across 4 SMs instead of 1 block on 1 SM.
  int const threads = std::min(32, num_experts_per_node);
  int const blocks = (num_experts_per_node + threads - 1) / threads;

  auto* kernel_instance =
      &computeStridesTmaWarpSpecializedKernel<T, WeightType, OutputType, ScaleBiasType>;

  cudaLaunchConfig_t config;
  config.gridDim = blocks;
  config.blockDim = threads;
  config.dynamicSmemBytes = 0;
  config.stream = stream;
  cudaLaunchAttribute attrs[1];
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = enable_pdl;
  config.numAttrs = 1;
  config.attrs = attrs;
  cudaLaunchKernelEx(&config, kernel_instance, expert_first_token_offset, layout_info1,
                     layout_info2, num_tokens, expanded_num_tokens, gemm1_n, gemm1_k, gemm2_n,
                     gemm2_k, num_experts_per_node, gemm1_in, gemm2_in, weights1, weights2,
                     alpha_scale_flat1, alpha_scale_flat2, fp4_act_flat1, fp4_act_flat2,
                     quant_params, bias1, bias2, gemm1_output, gemm2_output, router_scales,
                     permuted_row_to_unpermuted_row);

  return std::make_pair(layout_info1, layout_info2);
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Enable>::
    computeStridesTmaWarpSpecializedLowLatency(
        TmaWarpSpecializedGroupedGemmInput layout_info1,
        TmaWarpSpecializedGroupedGemmInput layout_info2, int64_t num_tokens, int64_t gemm1_n,
        int64_t gemm1_k, int64_t gemm2_n, int64_t gemm2_k, int const num_experts, T const* input1,
        T const* input2, WeightType const* weights1, WeightType const* weights2,
        float const* fp8_dequant1, float const* fp8_dequant2,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc1_fp4_act_flat,
        TmaWarpSpecializedGroupedGemmInput::ElementSF const* fc2_fp4_act_flat,
        QuantParams quant_params, ScaleBiasType const* bias1, ScaleBiasType const* bias2,
        UnfusedGemmOutputType* output1, UnfusedGemmOutputType* output2,
        int const* num_active_experts_per, int const* active_expert_global_ids, int start_expert,
        bool enable_pdl, cudaStream_t stream) {
  TLLM_THROW("Min latency mode is no longer supported");
}

template <class T, class WeightType, class OutputType, class InputType, class BackBoneType,
          bool IsMXFPX, class Enable>
std::pair<TmaWarpSpecializedGroupedGemmInput, TmaWarpSpecializedGroupedGemmInput>
CutlassMoeFCRunner<T, WeightType, OutputType, InputType, BackBoneType, IsMXFPX, Enable>::
    setupTmaWarpSpecializedInputs(int64_t num_rows, int64_t expanded_num_rows,
                                  ActivationParams fc1_activation_type, int64_t hidden_size,
                                  int64_t unpadded_hidden_size, int64_t inter_size,
                                  int64_t num_experts_per_node, void const* input_activations_void,
                                  TmaWarpSpecializedGroupedGemmInput::ElementSF const* input_sf,
                                  void* final_output, WeightType const* fc1_expert_weights,
                                  WeightType const* fc2_expert_weights, QuantParams quant_params,
                                  ScaleBiasType const* fc1_expert_biases,
                                  ScaleBiasType const* fc2_expert_biases, bool min_latency_mode,
                                  MoeMinLatencyParams& min_latency_params, bool use_lora,
                                  int start_expert, MOEParallelismConfig parallelism_config,
                                  bool allow_lora_finalize_fusion, bool enable_pdl,
                                  cudaStream_t stream) {
  auto gemm1_tma_ws_input = tma_ws_grouped_gemm1_input_;
  auto gemm2_tma_ws_input = tma_ws_grouped_gemm2_input_;

  // Set enable_pdl for both GEMM inputs
  gemm1_tma_ws_input.enable_pdl = enable_pdl;
  gemm2_tma_ws_input.enable_pdl = enable_pdl;
  if (!moe_gemm_runner_.isTmaWarpSpecialized(*gemm1_config_) &&
      !moe_gemm_runner_.isTmaWarpSpecialized(*gemm2_config_)) {
    return std::make_pair(gemm1_tma_ws_input, gemm2_tma_ws_input);
  }

  bool use_awq = quant_params.groupwise.fc1.act_scales && quant_params.groupwise.fc2.act_scales &&
                 !use_wfp4a16;

  bool is_gated_activation = isGatedActivation(fc1_activation_type);
  int64_t const fc1_out_size = is_gated_activation ? inter_size * 2 : inter_size;

  bool has_different_gemm_output_type = !std::is_same_v<T, UnfusedGemmOutputType>;
  bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
  auto* gemm1_output = has_intermediate ? glu_inter_result_ : static_cast<void*>(fc1_result_);

  bool use_prequant_scale_kernel = use_awq && !std::is_same_v<T, WeightType>;
  auto gemm2_input = use_prequant_scale_kernel ? smoothed_act_ : fc1_result_;

  if (min_latency_mode) {
    auto gemm1_input = input_activations_void;

    gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
    gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
    gemm1_tma_ws_input.swap_ab = true;
    gemm2_tma_ws_input.swap_ab = true;

    TLLM_CHECK_WITH_INFO(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
    return Self::computeStridesTmaWarpSpecializedLowLatency(
        gemm1_tma_ws_input, gemm2_tma_ws_input, num_rows, fc1_out_size, hidden_size, hidden_size,
        inter_size, num_experts_per_node, reinterpret_cast<T const*>(gemm1_input),
        reinterpret_cast<T const*>(gemm2_input), fc1_expert_weights, fc2_expert_weights,
        quant_params.fp8.dequant_fc1, quant_params.fp8.dequant_fc2, input_sf, fc2_fp4_act_scale_,
        quant_params, nullptr, nullptr, reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
        reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_),
        min_latency_params.num_active_experts_per_node, min_latency_params.active_expert_global_ids,
        start_expert, enable_pdl, stream);
  } else {
    auto gemm1_input = use_prequant_scale_kernel ? smoothed_act_ : permuted_data_;

    gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
    gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

    gemm1_tma_ws_input.swap_ab = gemm1_config_->swap_ab;
    gemm2_tma_ws_input.swap_ab = gemm2_config_->swap_ab;
    TLLM_CHECK_WITH_INFO(
        (gemm1_tma_ws_input.swap_ab && gemm2_tma_ws_input.swap_ab) || !use_w4_groupwise,
        "Hopper w4 mixed input groupwise requires swap_ab");

    bool apply_bias = parallelism_config.tp_rank == 0;
    auto* fc2_bias = apply_bias ? fc2_expert_biases : nullptr;
    bool gemm2_using_finalize_fusion =
        gemm2_config_->epilogue_fusion_type ==
        cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
    bool using_fused_finalize = use_fused_finalize_ && gemm2_using_finalize_fusion &&
                                !use_w4_groupwise && (!use_lora || allow_lora_finalize_fusion);
    TLLM_CHECK_WITH_INFO(
        using_fused_finalize == gemm2_using_finalize_fusion,
        "GEMM2 tactic requests finalize fusion, but the runner is not configured to use it");
    if (using_fused_finalize) {
      assert(min_latency_mode == false);
      bool use_reduction = expanded_num_rows > num_rows;
      gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
      gemm2_tma_ws_input.setFinalizeFusionParams(final_output, unpadded_hidden_size, num_rows,
                                                 use_reduction);
    }

    // fp8_mxfp4 memsets the scaling factors to 1.0f
    if (quant_params.fp8_mxfp4.fc1.weight_block_scale) {
      // We are in FP8 x MXFP4 mode
      TLLM_CHECK(quant_params.fp8_mxfp4.fc2.weight_block_scale);
      TLLM_CHECK(fc1_fp4_act_scale_ != nullptr);
      TLLM_CHECK_WITH_INFO(fc1_fp4_act_scale_ == fc2_fp4_act_scale_,
                           "WFP4AFP8 expects the scaling factors to be aliased for gemm1 & gemm2");

      TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF weight_block_scale_value_int{};
#if defined(FLASHINFER_ENABLE_FP8_E8M0) && CUDART_VERSION >= 12080
      __nv_fp8_e8m0 tmp;
      tmp.__x = __nv_cvt_float_to_e8m0(1.0f, __NV_SATFINITE, cudaRoundPosInf);
      std::memcpy(&weight_block_scale_value_int, &tmp, sizeof(tmp));
#endif

      auto act_sf_rows = std::min(expanded_num_rows, num_rows * num_experts_per_node);
      auto fc1_sf_offset =
          getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size,
                                TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
      auto fc2_sf_offset =
          getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size,
                                TmaWarpSpecializedGroupedGemmInput::FpXBlockScalingType::MXFPX);
      auto max_size = std::max(fc1_sf_offset, fc2_sf_offset) *
                      sizeof(TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF);
      check_cuda_error(
          cudaMemsetAsync(fc1_fp4_act_scale_, weight_block_scale_value_int, max_size, stream));
    }

    TLLM_CHECK_WITH_INFO(gemm1_input != gemm1_output, "Input and output buffers are overlapping");
    return Self::computeStridesTmaWarpSpecialized(
        expert_first_token_offset_, gemm1_tma_ws_input, gemm2_tma_ws_input, num_rows,
        expanded_num_rows, fc1_out_size, hidden_size, hidden_size, inter_size, num_experts_per_node,
        reinterpret_cast<T const*>(gemm1_input), reinterpret_cast<T const*>(gemm2_input),
        fc1_expert_weights, fc2_expert_weights, quant_params.fp8.dequant_fc1,
        quant_params.fp8.dequant_fc2, fc1_fp4_act_scale_, fc2_fp4_act_scale_, quant_params,
        fc1_expert_biases, fc2_bias, reinterpret_cast<UnfusedGemmOutputType*>(gemm1_output),
        reinterpret_cast<UnfusedGemmOutputType*>(fc2_result_), permuted_token_final_scales_,
        permuted_row_to_unpermuted_row_, enable_pdl, stream);
  }
}

// ==================== Helper for getting load balanced routing for profiling
// ==================================

__global__ void prepareFakeRouterBuffers(int* token_selected_experts, int64_t num_tokens, int64_t k,
                                         int64_t num_experts) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t sample = blockIdx.y;
  if (tid >= num_tokens) {
    return;
  }

  // Offset the buffers to the start of the sample
  token_selected_experts += sample * num_tokens * k;

  // This is not perf sensitive we just init the state here every time prepare is called
  // This means the first N tokens will always have the same distribution, regardless of num_tokens
  curandStatePhilox4_32_10_t state;
  curand_init(sample, tid, 0, &state);
  for (int k_idx = 0; k_idx < k; k_idx++) {
    while (true) {
      // curand_uniform includes 1 but not 0, so round up and subtract 1
      int expert = std::ceil(static_cast<float>(num_experts) * curand_uniform(&state)) - 1;

      bool valid = true;
      for (int prev_k = 0; prev_k < k_idx; prev_k++) {
        int prev_expert = token_selected_experts[k * tid + prev_k];
        if (expert == prev_expert) {
          valid = false;
          break;
        }
      }

      if (valid) {
        token_selected_experts[k * tid + k_idx] = expert;
        break;
      }
    }
  }
}

__global__ void populateRandomBufferKernel(void* buffer_void, size_t size) {
  int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= size) {
    return;
  }

  curandStatePhilox4_32_10_t state;
  curand_init(size, tid, 0, &state);

  constexpr int elem_per_thread = 128 / sizeof(uint4);
  auto* buffer = reinterpret_cast<uint4*>(buffer_void);
#pragma unroll
  for (int i = 0; i < elem_per_thread; i++) buffer[tid * elem_per_thread + i] = curand4(&state);
}

template <int BLOCK_SIZE, int NUM_ROUTING_SAMPLES>
__global__ void prepareMinLatencyBuffer(int* num_active_experts_per_node,
                                        int* active_expert_global_ids,
                                        int64_t* expert_first_token_offset, int const num_tokens,
                                        int const num_experts_per_token,
                                        int const num_experts_per_node) {
  int tid = threadIdx.x;
  int bid = blockIdx.x;

  // 0. set offset
  num_active_experts_per_node += bid;
  active_expert_global_ids += bid * num_experts_per_node;
  expert_first_token_offset += bid * (num_experts_per_node + 1);

  // 1. set the num_active_experts_per_node
  int num_active = max(1, (int)(bid * ((float)num_experts_per_node / NUM_ROUTING_SAMPLES)));
  *num_active_experts_per_node = num_active;

  // 2. generate random active experts
  extern __shared__ float s_buf[];
  float* expert_refs = s_buf;
  int* expert_refs_idx = reinterpret_cast<int*>(expert_refs + num_experts_per_node);

  curandState_t local_state;
  curand_init(bid, tid, 0, &local_state);
  for (int i = tid; i < num_experts_per_node; i += BLOCK_SIZE) {
    expert_refs[i] = (float)curand_uniform(&local_state);
    expert_refs_idx[i] = (int)i;
  }
  __syncthreads();

  float thread_key[1];
  int thread_value[1];
  thread_key[0] = std::numeric_limits<float>::max();
  thread_value[0] = num_experts_per_node;

  if (tid < num_experts_per_node) {
    thread_key[0] = expert_refs[tid];
    thread_value[0] = expert_refs_idx[tid];
  }

  using BlockRadixSort = cub::BlockRadixSort<float, BLOCK_SIZE, 1, int>;
  using BlockRadixSortValue = cub::BlockRadixSort<int, BLOCK_SIZE, 1>;

  union TempStorage {
    typename BlockRadixSort::TempStorage key_value;
    typename BlockRadixSortValue::TempStorage value;
  };
  __shared__ union TempStorage temp_storage;

  BlockRadixSort(temp_storage.key_value).Sort(thread_key, thread_value);
  __syncthreads();

  if (tid > num_active) {
    thread_value[0] = std::numeric_limits<int>::max();
  }
  BlockRadixSortValue(temp_storage.value).Sort(thread_value);
  __syncthreads();

  // 3. set the active_expert_global_ids and expert_first_token_offset
  for (int i = tid; i < num_experts_per_node; i += BLOCK_SIZE) {
    if (i < num_active) {
      active_expert_global_ids[i] = thread_value[0];
      expert_first_token_offset[i] = i * num_tokens;
    } else {
      active_expert_global_ids[i] = -1;
      expert_first_token_offset[i] = num_active * num_tokens;
    }
  }
  if (tid == 0) {
    expert_first_token_offset[num_experts_per_node] = num_active * num_tokens;
  }
}

void populateRandomBuffer(void* buffer_void, size_t size, cudaStream_t stream) {
  // Each thread initialises 128 bytes
  TLLM_CHECK_WITH_INFO(size % 128 == 0, "Unexpected size alignment");
  auto threads = size / 128;
  populateRandomBufferKernel<<<ceilDiv(threads, 128), 128, 0, stream>>>(buffer_void, threads);
}

std::map<std::string, std::pair<size_t, size_t>> GemmProfilerBackend::getProfilerWorkspaces(
    int maxM, bool is_tma_ws_input) {
  size_t k = mK;
  size_t num_expanded_tokens = mMinLatencyMode ? maxM * mNumExpertsPerNode : maxM * k;

  TLLM_CHECK(mDType != nvinfer1::DataType::kINT4);
  // nvllm still uses int64 because torch doesn't have fp4 yet.
  bool is_4bit_act = mDType == nvinfer1::DataType::kFP4 || mDType == nvinfer1::DataType::kINT64;
  bool is_4bit_weight = mWType == nvinfer1::DataType::kINT4 || mWType == nvinfer1::DataType::kFP4 ||
                        mWType == nvinfer1::DataType::kINT64;
  TLLM_CHECK_WITH_INFO(!is_4bit_act || is_4bit_weight,
                       "Cannot have 4-bit activation with non-4-bit weight");
  float dtype_bytes = is_4bit_act ? 0.5f
                                  : static_cast<float>((mWType == nvinfer1::DataType::kINT4 ||
                                                        mWType == nvinfer1::DataType::kUINT8)
                                                           ? getDTypeSize(mOType)
                                                           : getDTypeSize(mDType));
  float weight_bytes = is_4bit_weight ? 0.5f : static_cast<float>(getDTypeSize(mWType));
  size_t output_bytes = getDTypeSize(mOType);
  size_t gemm_output_bytes =
      (mOType == nvinfer1::DataType::kFP8)
          ? sizeof(TmaWarpSpecializedGroupedGemmInput::OutputTypeAdaptor_t<__nv_fp8_e4m3>)
          : output_bytes;

  size_t hidden_size = mExpertHiddenSize;
  size_t inter_size = mExpertInterSize;  // Already divided by TP
  size_t num_experts_per_node = mNumExpertsPerNode;

  size_t fc1_out_size = inter_size;
  if (isGatedActivation(mActivationType)) {
    fc1_out_size = inter_size * 2;
  }

  // TODO Needs updated when gather/finalize fusion is integrated
  size_t input_size1 = hidden_size * num_expanded_tokens * dtype_bytes;
  size_t output_size1 = inter_size * num_expanded_tokens * dtype_bytes;

  size_t input_size2 = inter_size * num_expanded_tokens * dtype_bytes;
  size_t output_size2 = hidden_size * output_bytes;

  size_t input_size = mGemmToProfile == GemmToProfile::GEMM_1 ? input_size1 : input_size2;
  size_t output_size = mGemmToProfile == GemmToProfile::GEMM_1 ? output_size1 : output_size2;

  // This may allocate a pointer when not required. That's fine it will be ignored at the cost of
  // some memory
  size_t intermediate_size1 =
      fc1_out_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes
  size_t intermediate_size2 =
      hidden_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes

  size_t intermediate_size =
      mGemmToProfile == GemmToProfile::GEMM_1 ? intermediate_size1 : intermediate_size2;

  size_t weights_1 = hidden_size * fc1_out_size * num_experts_per_node * weight_bytes;
  size_t bias_1 = mBias ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
  if (mUseLora && !is_tma_ws_input) bias_1 = output_size1;
  size_t weights_2 = hidden_size * inter_size * num_experts_per_node * weight_bytes;
  size_t bias_2 = mBias ? hidden_size * num_experts_per_node * dtype_bytes : 0;

  size_t weights_size =
      mNeedWeights ? (mGemmToProfile == GemmToProfile::GEMM_1 ? weights_1 : weights_2) : 0;
  size_t bias_size = mGemmToProfile == GemmToProfile::GEMM_1 ? bias_1 : bias_2;

  // TODO Make quant 2 & 4 bigger for FP8 if we ever change to scaling per expert
  bool is_int_w_quant =
      (mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4 ||
       mWType == nvinfer1::DataType::kUINT8) &&
      mGroupSize <= 0;
  bool is_int_groupwise_w_quant =
      (mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4 ||
       mWType == nvinfer1::DataType::kUINT8) &&
      mGroupSize > 0;
  bool is_fp8_act_quant = mDType == nvinfer1::DataType::kFP8;
  bool is_fp8_w_quant = mWType == nvinfer1::DataType::kFP8;
  // nvllm still uses int64 because torch doesn't have fp4 yet.
  // bool is_fp4_act_quant = mDType == nvinfer1::DataType::kFP4 || mDType ==
  // nvinfer1::DataType::kINT64;
  bool is_fp4_w_quant = mWType == nvinfer1::DataType::kFP4 || mWType == nvinfer1::DataType::kINT64;
  bool is_w4afp8_quant = is_int_groupwise_w_quant && is_fp8_act_quant;
  // bool is_wfp4afp8_quant = is_fp4_w_quant && is_fp8_act_quant;
  bool is_wfp4a16_quant =
      (mDType == nvinfer1::DataType::kHALF || mDType == nvinfer1::DataType::kBF16) &&
      mWType == nvinfer1::DataType::kUINT8;

  // Int sizes
  size_t quant_1_size = is_int_w_quant ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
  size_t quant_2_size = is_int_w_quant ? hidden_size * num_experts_per_node * dtype_bytes : 0;
  if (is_int_w_quant) {
    quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes;
    quant_2_size = hidden_size * num_experts_per_node * dtype_bytes;
  } else if (is_int_groupwise_w_quant || is_wfp4a16_quant) {
    quant_1_size = fc1_out_size * num_experts_per_node * dtype_bytes * hidden_size / mGroupSize;
    quant_2_size = hidden_size * num_experts_per_node * dtype_bytes * inter_size / mGroupSize;
  }

  // FP8 sizes
  quant_1_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : quant_1_size;
  quant_2_size = is_fp8_w_quant ? sizeof(float) : quant_2_size;
  size_t quant_3_size = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : 0;
  size_t quant_4_size = 0;  // Currently ignored by the GEMM
  if (is_int_groupwise_w_quant) {
    quant_3_size = quant_1_size;
    quant_4_size = quant_2_size;
  }

  // FP4 sizes
  quant_1_size = is_fp4_w_quant ? sizeof(float) : quant_1_size;
  quant_2_size = is_fp4_w_quant ? getOffsetWeightSF(num_experts_per_node, inter_size, hidden_size,
                                                    mScalingType) *
                                      sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
                                : quant_2_size;
  quant_3_size = is_fp4_w_quant ? num_experts_per_node * sizeof(float) : quant_3_size;
  quant_4_size = is_fp4_w_quant ? sizeof(float) : quant_4_size;
  size_t quant_5_size = is_fp4_w_quant ? getOffsetWeightSF(num_experts_per_node, hidden_size,
                                                           inter_size, mScalingType) *
                                             sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF)
                                       : 0;
  size_t quant_6_size = is_fp4_w_quant ? num_experts_per_node * sizeof(float) : 0;

  size_t tma_ws_input_workspace_size = 0;
  if (is_tma_ws_input) {
    tma_ws_input_workspace_size =
        TmaWarpSpecializedGroupedGemmInput::workspaceSize(num_experts_per_node, mScalingType) *
        (NUM_ROUTING_SAMPLES * NUM_FUSION_TYPES * NUM_SWAP_AB_TYPES + 1);

    if (is_w4afp8_quant || is_wfp4a16_quant) {
      quant_3_size = 0;
      quant_4_size = 0;
    }
  }

  auto act_sf_rows = mMinLatencyMode ? num_expanded_tokens
                                     : std::min(num_expanded_tokens,
                                                static_cast<size_t>(maxM * num_experts_per_node));
  // getOffsetActivationSF returns zero if scaling_type is NONE
  size_t const fc1_fp4_act_scale_size =
      getOffsetActivationSF(num_experts_per_node, act_sf_rows, hidden_size, mScalingType) *
      sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
  size_t const fc2_fp4_act_scale_size =
      getOffsetActivationSF(num_experts_per_node, act_sf_rows, inter_size, mScalingType) *
      sizeof(TmaWarpSpecializedGroupedGemmInput::ElementSF);
  size_t const fp4_act_scale_flat_size = std::max(fc1_fp4_act_scale_size, fc2_fp4_act_scale_size);

  size_t w4a8_alpha_size =
      (is_w4afp8_quant || is_wfp4a16_quant) ? num_experts_per_node * sizeof(float) : 0;
  size_t alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float**);
  size_t gemm_workspace_size = mInterface->getGemmWorkspaceSize(num_experts_per_node);

  // Routing info
  size_t expert_first_token_offset_size =
      (num_experts_per_node + 1) * sizeof(int64_t) * NUM_ROUTING_SAMPLES;
  size_t map_size = mMinLatencyMode ? 0 : NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t unpermuted_size =
      mMinLatencyMode ? 0 : NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
  size_t permuted_size = mMinLatencyMode ? 0 : num_expanded_tokens * sizeof(int);
  size_t token_topk_unpermuted_scales_size =
      mMinLatencyMode ? 0 : num_expanded_tokens * sizeof(float);

  int64_t const num_tokens_per_block = computeNumTokensPerBlock(maxM, num_experts_per_node);
  int64_t const num_blocks_per_seq = tensorrt_llm::common::ceilDiv(maxM, num_tokens_per_block);
  size_t const blocked_expert_counts_size =
      mMinLatencyMode ? 0 : num_experts_per_node * num_blocks_per_seq * sizeof(int);
  size_t const blocked_expert_counts_cumsum_size = blocked_expert_counts_size;
  size_t const blocked_row_to_unpermuted_row_size =
      mMinLatencyMode ? 0 : num_experts_per_node * maxM * sizeof(int);

  // The follow buffers are used in min_latency_mode
  size_t num_active_experts_per_node_size =
      mMinLatencyMode ? sizeof(int) * NUM_ROUTING_SAMPLES
                      : 0;  // smaller than or equal to num_experts_per_node
  size_t active_expert_global_ids_size =
      mMinLatencyMode ? mNumExpertsPerNode * sizeof(int) * NUM_ROUTING_SAMPLES : 0;

  bool is_swiglu_bias =
      mActivationType == ActivationType::SwigluBias && mGemmToProfile == GemmToProfile::GEMM_1;
  size_t swiglu_alpha_size = is_swiglu_bias ? num_experts_per_node * sizeof(float) : 0;
  size_t swiglu_beta_size = is_swiglu_bias ? num_experts_per_node * sizeof(float) : 0;
  size_t swiglu_limit_size = is_swiglu_bias ? num_experts_per_node * sizeof(float) : 0;

  size_t map_offset = 0;
  std::map<std::string, std::pair<size_t, size_t>> out_map;

#define ADD_NAME(name, size)                              \
  do {                                                    \
    auto aligned_size = alignSize(size, kCudaMemAlign);   \
    out_map[#name] = std::pair{aligned_size, map_offset}; \
    map_offset += aligned_size;                           \
  } while (false)
#define ADD(name) ADD_NAME(name, name##_size)

  ADD(expert_first_token_offset);
  ADD_NAME(unpermuted_row_to_permuted_row, map_size);
  ADD_NAME(permuted_row_to_unpermuted_row, map_size);
  ADD_NAME(token_selected_experts, unpermuted_size);
  ADD_NAME(permuted_token_selected_experts, permuted_size);
  ADD(blocked_expert_counts);
  ADD(blocked_expert_counts_cumsum);
  ADD(blocked_row_to_unpermuted_row);
  ADD(token_topk_unpermuted_scales);
  ADD(num_active_experts_per_node);
  ADD(active_expert_global_ids);
  ADD(input);
  ADD(output);
  ADD(intermediate);
  ADD(weights);
  ADD(bias);
  ADD(quant_1);
  ADD(quant_2);
  ADD(quant_3);
  ADD(quant_4);
  ADD(quant_5);
  ADD(quant_6);
  ADD(tma_ws_input_workspace);
  ADD(w4a8_alpha);
  ADD(alpha_scale_ptr_array);
  ADD(fp4_act_scale_flat);
  ADD(gemm_workspace);
  ADD(swiglu_alpha);
  ADD(swiglu_beta);
  ADD(swiglu_limit);
#undef ADD_NAME
#undef ADD

  return out_map;
}

void GemmProfilerBackend::prepareRouting(int num_tokens, char* workspace_ptr_char, bool enable_pdl,
                                         cudaStream_t stream) {
  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR_BASE(type, name)                                                   \
  auto* name##_base =                                                                 \
      (workspaces.at(#name).first                                                     \
           ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
           : nullptr)
#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)

  GET_WS_PTR_BASE(int64_t*, expert_first_token_offset);
  GET_WS_PTR_BASE(int*, unpermuted_row_to_permuted_row);
  GET_WS_PTR_BASE(int*, permuted_row_to_unpermuted_row);
  GET_WS_PTR_BASE(int*, token_selected_experts);
  GET_WS_PTR(int*, permuted_token_selected_experts);
  GET_WS_PTR(int*, blocked_expert_counts);
  GET_WS_PTR(int*, blocked_expert_counts_cumsum);
  GET_WS_PTR(int*, blocked_row_to_unpermuted_row);
  GET_WS_PTR(int*, num_active_experts_per_node);
  GET_WS_PTR(int*, active_expert_global_ids);

#undef GET_WS_PTR_BASE
#undef GET_WS_PTR

  if (mMinLatencyMode) {
    // expert_first_token_offset for each sample
    TLLM_CHECK_WITH_INFO(mNumExpertsPerNode <= 256,
                         "Min latency mode only supports #experts < 256");
    prepareMinLatencyBuffer<256, NUM_ROUTING_SAMPLES>
        <<<NUM_ROUTING_SAMPLES, 256, mNumExpertsPerNode * (sizeof(float) + sizeof(int)), stream>>>(
            num_active_experts_per_node, active_expert_global_ids, expert_first_token_offset_base,
            num_tokens, mK, mNumExpertsPerNode);
  } else {
    int64_t const num_expanded_tokens = num_tokens * mK;
    int const start_expert_id = mNumExpertsPerNode * mParallelismConfig.ep_rank;

    uint32_t num_threads = 256;
    dim3 grid_dim{(num_tokens + num_threads - 1) / num_threads, NUM_ROUTING_SAMPLES, 1};
    prepareFakeRouterBuffers<<<grid_dim, num_threads, 0, stream>>>(token_selected_experts_base,
                                                                   num_tokens, mK, mNumExperts);
    sync_check_cuda_error(stream);

    for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
      int64_t* expert_first_token_offset =
          expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
      int* unpermuted_row_to_permuted_row =
          unpermuted_row_to_permuted_row_base + i * num_expanded_tokens;
      int* permuted_row_to_unpermuted_row =
          permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;
      int* token_selected_experts = token_selected_experts_base + i * num_expanded_tokens;

      threeStepBuildExpertMapsSortFirstToken(
          token_selected_experts, permuted_token_selected_experts, permuted_row_to_unpermuted_row,
          unpermuted_row_to_permuted_row, expert_first_token_offset, blocked_expert_counts,
          blocked_expert_counts_cumsum, blocked_row_to_unpermuted_row, num_tokens,
          mNumExpertsPerNode, mK, start_expert_id, enable_pdl, stream);
      sync_check_cuda_error(stream);
    }
  }
}

void GemmProfilerBackend::prepareQuantParams(int num_tokens, char* workspace_ptr_char,
                                             cudaStream_t) {
  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);
#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)
  GET_WS_PTR(void const*, quant_1);
  GET_WS_PTR(void const*, quant_2);
  GET_WS_PTR(void const*, quant_3);
  GET_WS_PTR(void const*, quant_4);
  GET_WS_PTR(void const*, quant_5);
  GET_WS_PTR(void const*, quant_6);
  GET_WS_PTR(float const*, w4a8_alpha);
#undef GET_WS_PTR

  if ((mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4 ||
       mWType == nvinfer1::DataType::kUINT8) &&
      mGroupSize < 0) {
    TLLM_CHECK(quant_1 && quant_2);
    mQuantParams = QuantParams::Int(quant_1, quant_2);
  } else if (mWType == nvinfer1::DataType::kINT4 || mWType == nvinfer1::DataType::kUINT8) {
    TLLM_CHECK(quant_1 && quant_2);
    if (mDType == nvinfer1::DataType::kFP8 ||
        (mWType == nvinfer1::DataType::kUINT8 &&
         (mDType == nvinfer1::DataType::kHALF || mDType == nvinfer1::DataType::kBF16))) {
      TLLM_CHECK(w4a8_alpha);
      mQuantParams = QuantParams::GroupWise(mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3,
                                            quant_4, w4a8_alpha, w4a8_alpha);
    } else {
      mQuantParams =
          QuantParams::GroupWise(mGroupSize, quant_1, quant_2, nullptr, nullptr, quant_3, quant_4);
    }
  } else if (mWType == nvinfer1::DataType::kFP8) {
    TLLM_CHECK(quant_1 && quant_2 && quant_3);
    mQuantParams =
        QuantParams::FP8(static_cast<float const*>(quant_1), static_cast<float const*>(quant_2),
                         static_cast<float const*>(quant_3), static_cast<float const*>(quant_4));
  } else if (mDType == nvinfer1::DataType::kFP8 &&
             (mWType == nvinfer1::DataType::kFP4 || mWType == nvinfer1::DataType::kINT64)) {
    TLLM_CHECK(quant_1 && quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
    mQuantParams = QuantParams::FP8MXFP4(
        static_cast<float const*>(quant_1),
        static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_2),
        static_cast<float const*>(quant_3), static_cast<float const*>(quant_4),
        static_cast<TmaWarpSpecializedGroupedGemmInput::MXFPXElementSF const*>(quant_5),
        static_cast<float const*>(quant_6));
  } else if ((mDType == nvinfer1::DataType::kFP4 || mDType == nvinfer1::DataType::kINT64) &&
             (mWType == nvinfer1::DataType::kFP4 || mWType == nvinfer1::DataType::kINT64)) {
    // nvllm still uses int64 because torch doesn't have fp4 yet.
    TLLM_CHECK(quant_1 && quant_2 && quant_3 && quant_4 && quant_5 && quant_6);
    mQuantParams = QuantParams::FP4(
        static_cast<float const*>(quant_1),
        static_cast<TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(quant_2),
        static_cast<float const*>(quant_3), static_cast<float const*>(quant_4),
        static_cast<TmaWarpSpecializedGroupedGemmInput::NVFP4ElementSF const*>(quant_5),
        static_cast<float const*>(quant_6));
  }
}

void GemmProfilerBackend::prepareTmaWsInputs(
    int num_tokens, char* workspace_ptr_char, void const* expert_weights,
    TmaWarpSpecializedGroupedGemmInput::EpilogueFusion fusion, bool swap_ab, bool enable_pdl,
    cudaStream_t stream) {
  if (mSM < 90) {
    return;
  }

  bool use_w4afp8 = (mDType == nvinfer1::DataType::kFP8 && mWType == nvinfer1::DataType::kINT4);
  bool use_wfp4a16 =
      ((mDType == nvinfer1::DataType::kHALF || mDType == nvinfer1::DataType::kBF16) &&
       mWType == nvinfer1::DataType::kUINT8);
  bool use_w4_groupwise = use_w4afp8 || use_wfp4a16;
  bool const use_finalize_fusion =
      fusion == TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
  bool const finalize_fusion_not_supported = !mInterface->use_fused_finalize_ || mMinLatencyMode ||
                                             use_w4_groupwise || mUseLora ||
                                             mGemmToProfile != GemmToProfile::GEMM_2;
  if (use_finalize_fusion && finalize_fusion_not_supported) {
    return;
  }

  if (use_w4_groupwise && !swap_ab) {
    return;
  }

  auto workspaces = getProfilerWorkspaces(num_tokens, mSM >= 90);

#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)

  GET_WS_PTR(int64_t*, expert_first_token_offset);
  int64_t* expert_first_token_offset_base = expert_first_token_offset;
  GET_WS_PTR(int*, permuted_row_to_unpermuted_row);
  int* permuted_row_to_unpermuted_row_base = permuted_row_to_unpermuted_row;
  GET_WS_PTR(void*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(void*, weights);
  TLLM_CHECK(mNeedWeights == (expert_weights == nullptr));
  void const* weights_sel = mNeedWeights ? weights : expert_weights;
  GET_WS_PTR(void*, bias);
  GET_WS_PTR(float*, token_topk_unpermuted_scales);
  GET_WS_PTR(int8_t*, tma_ws_input_workspace);
  GET_WS_PTR(void*, gemm_workspace);
  GET_WS_PTR(float*, alpha_scale_ptr_array);
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);
  GET_WS_PTR(int*, num_active_experts_per_node);
  GET_WS_PTR(int*, active_expert_global_ids);

#undef GET_WS_PTR

  size_t tma_ws_size =
      TmaWarpSpecializedGroupedGemmInput::workspaceSize(mNumExpertsPerNode, mScalingType);

  TmaWarpSpecializedGroupedGemmInput dummy_tma_ws_input;
  dummy_tma_ws_input.configureWorkspace(tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
                                        workspaces.at("gemm_workspace").first, mScalingType);
  dummy_tma_ws_input.enable_pdl = enable_pdl;  // Set enable_pdl for dummy input
  tma_ws_input_workspace += tma_ws_size;

  int workspace_index =
      static_cast<int>(use_finalize_fusion) * (NUM_SWAP_AB_TYPES * NUM_ROUTING_SAMPLES) +
      static_cast<int>(swap_ab) * NUM_ROUTING_SAMPLES;
  tma_ws_input_workspace += workspace_index * tma_ws_size;

  size_t num_expanded_tokens = num_tokens * mK;
  for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++) {
    // Note: Even though we have separate TMA WS inputs for finalize fusion on/off we reuse the same
    // pointers to save space.
    auto& cache_element = mTmaInputCache[use_finalize_fusion][swap_ab][i];
    cache_element.configureWorkspace(tma_ws_input_workspace, mNumExpertsPerNode, gemm_workspace,
                                     workspaces.at("gemm_workspace").first, mScalingType);
    cache_element.enable_pdl = enable_pdl;  // Set enable_pdl for cache element
    tma_ws_input_workspace += tma_ws_size;

    int64_t* expert_first_token_offset =
        expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
    int* permuted_row_to_unpermuted_row =
        permuted_row_to_unpermuted_row_base + i * num_expanded_tokens;

    auto& gemm1_tma_ws_input =
        mGemmToProfile == GemmToProfile::GEMM_1 ? cache_element : dummy_tma_ws_input;
    auto& gemm2_tma_ws_input =
        mGemmToProfile == GemmToProfile::GEMM_2 ? cache_element : dummy_tma_ws_input;
    if (mSM >= 90) {
      auto fc1_output_size =
          isGatedActivation(mActivationType) ? mExpertInterSize * 2 : mExpertInterSize;

      /* GEMM1 */
      gemm1_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;
      gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE;

      gemm1_tma_ws_input.swap_ab = swap_ab;
      gemm2_tma_ws_input.swap_ab = swap_ab;

      if (use_finalize_fusion) {
        assert(!mMinLatencyMode);
        gemm2_tma_ws_input.fusion = TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE;
        gemm2_tma_ws_input.setFinalizeFusionParams(output, mExpertUnpaddedHiddenSize, num_tokens,
                                                   mK > 1);
      }

      if (mMinLatencyMode) {
        std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input) =
            mInterface->computeStridesTmaWarpSpecializedLowLatencyDispatch(
                gemm1_tma_ws_input, gemm2_tma_ws_input, num_tokens, fc1_output_size,
                mExpertHiddenSize, mExpertHiddenSize, mExpertInterSize, mNumExpertsPerNode, input,
                input, weights_sel, weights_sel, mQuantParams.fp8.dequant_fc1,
                mQuantParams.fp8.dequant_fc2, fp4_act_scale_flat, fp4_act_scale_flat, mQuantParams,
                nullptr, nullptr, intermediate, intermediate, num_active_experts_per_node,
                active_expert_global_ids, 0, enable_pdl, stream);
      } else {
        std::tie(gemm1_tma_ws_input, gemm2_tma_ws_input) =
            mInterface->computeStridesTmaWarpSpecializedDispatch(
                expert_first_token_offset, gemm1_tma_ws_input, gemm2_tma_ws_input, num_tokens,
                num_tokens * mK, fc1_output_size, mExpertHiddenSize, mExpertHiddenSize,
                mExpertInterSize, mNumExpertsPerNode, input, input, weights_sel, weights_sel,
                mQuantParams.fp8.dequant_fc1, mQuantParams.fp8.dequant_fc2, fp4_act_scale_flat,
                fp4_act_scale_flat, mQuantParams, nullptr, nullptr, intermediate, intermediate,
                token_topk_unpermuted_scales, permuted_row_to_unpermuted_row, enable_pdl, stream);
      }
      sync_check_cuda_error(stream);
    }
  }
}

void GemmProfilerBackend::prepare(int num_tokens, char* workspace_ptr_char,
                                  void const* expert_weights, bool enable_pdl,
                                  cudaStream_t stream) {
  mSampleIndex = 0;

  auto workspace_size = getWorkspaceSize(num_tokens);
  populateRandomBuffer(workspace_ptr_char, workspace_size, stream);

  prepareRouting(num_tokens, workspace_ptr_char, enable_pdl, stream);
  prepareQuantParams(num_tokens, workspace_ptr_char, stream);
  for (auto fusion : {TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::NONE,
                      TmaWarpSpecializedGroupedGemmInput::EpilogueFusion::FINALIZE}) {
    for (auto swap_ab : {false, true}) {
      prepareTmaWsInputs(num_tokens, workspace_ptr_char, expert_weights, fusion, swap_ab,
                         enable_pdl, stream);
    }
  }
}

size_t GemmProfilerBackend::getWorkspaceSize(int maxM) {
  auto sizes_map = getProfilerWorkspaces(maxM, mSM >= 90);
  std::vector<size_t> sizes(sizes_map.size());
  std::transform(sizes_map.begin(), sizes_map.end(), sizes.begin(),
                 [](auto& v) { return v.second.first; });
  size_t size = calculateTotalWorkspaceSize(sizes.data(), sizes.size());
  TLLM_LOG_TRACE("MOE profiler workspace size: %zu", size);
  return size;
}

void GemmProfilerBackend::runProfiler(int original_num_tokens, Config const& tactic,
                                      char* workspace_ptr_char, void const* expert_weights,
                                      bool enable_pdl, cudaStream_t const& stream) {
  int64_t expanded_num_tokens = original_num_tokens * mK;
  int64_t num_experts_per_node = mNumExpertsPerNode;

  mSampleIndex = (mSampleIndex + 1) % NUM_ROUTING_SAMPLES;

  auto workspaces = getProfilerWorkspaces(original_num_tokens, tactic.is_tma_warp_specialized);

#define GET_WS_PTR_OFFSET(type, name, offset)                                                    \
  auto* name =                                                                                   \
      (workspaces.at(#name).first                                                                \
           ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) + (offset) \
           : nullptr)
#define GET_WS_PTR(type, name)                                                                 \
  auto* name = (workspaces.at(#name).first                                                     \
                    ? reinterpret_cast<type>(workspace_ptr_char + workspaces.at(#name).second) \
                    : nullptr)

  GET_WS_PTR_OFFSET(int64_t const*, expert_first_token_offset,
                    (mSampleIndex * (mNumExpertsPerNode + 1)));
  GET_WS_PTR_OFFSET(int const*, unpermuted_row_to_permuted_row,
                    (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(int const*, permuted_row_to_unpermuted_row,
                    (mSampleIndex * expanded_num_tokens));
  GET_WS_PTR_OFFSET(int const*, token_selected_experts, (mSampleIndex * expanded_num_tokens));

  GET_WS_PTR(float const*, token_topk_unpermuted_scales);
  auto const* token_topk_permuted_scales = token_topk_unpermuted_scales;

  GET_WS_PTR_OFFSET(int*, num_active_experts_per_node, mSampleIndex);
  GET_WS_PTR_OFFSET(int*, active_expert_global_ids, (mSampleIndex * mNumExpertsPerNode));
  GET_WS_PTR(void const*, input);
  GET_WS_PTR(void*, output);
  GET_WS_PTR(void*, intermediate);
  GET_WS_PTR(void const*, weights);
  TLLM_CHECK(mNeedWeights == (expert_weights == nullptr));
  void const* weights_sel = mNeedWeights ? weights : expert_weights;
  GET_WS_PTR(void const*, bias);

  GET_WS_PTR(float const**, alpha_scale_ptr_array);
  GET_WS_PTR(TmaWarpSpecializedGroupedGemmInput::ElementSF*, fp4_act_scale_flat);
  GET_WS_PTR(void*, gemm_workspace);

  GET_WS_PTR(float*, swiglu_alpha);
  GET_WS_PTR(float*, swiglu_beta);
  GET_WS_PTR(float*, swiglu_limit);

#undef GET_WS_PTR_OFFSET
#undef GET_WS_PTR

  TmaWarpSpecializedGroupedGemmInput tma_ws_input_template;
  if (tactic.is_tma_warp_specialized) {
    // Use non-finalize cache when finalize fusion is not supported for the current GEMM
    bool use_w4afp8 = (mDType == nvinfer1::DataType::kFP8 && mWType == nvinfer1::DataType::kINT4);
    bool use_wfp4a16 =
        ((mDType == nvinfer1::DataType::kHALF || mDType == nvinfer1::DataType::kBF16) &&
         mWType == nvinfer1::DataType::kUINT8);
    bool use_w4_groupwise = use_w4afp8 || use_wfp4a16;
    bool finalize_supported_this_gemm = (mGemmToProfile == GemmToProfile::GEMM_2) &&
                                        mInterface->use_fused_finalize_ && !mMinLatencyMode &&
                                        !use_w4_groupwise && !mUseLora;
    bool request_finalize = tactic.epilogue_fusion_type ==
                            cutlass_extensions::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
    TLLM_CHECK_WITH_INFO(
        !request_finalize || finalize_supported_this_gemm,
        "GEMM tactic requests finalize fusion, but profiler is not configured to use it");
    bool use_finalize_index = request_finalize && finalize_supported_this_gemm;

    tma_ws_input_template = mTmaInputCache[use_finalize_index][tactic.swap_ab][mSampleIndex];
    TLLM_CHECK_WITH_INFO(tma_ws_input_template.isValid(),
                         "TMA WS input template is not initialized");
  }

  mInterface->is_profiler = true;
  if (mGemmToProfile == GemmToProfile::GEMM_1) {
    mInterface->gemm1(input,                                             //
                      output,                                            //
                      intermediate,                                      //
                      expert_first_token_offset,                         //
                      tma_ws_input_template,                             //
                      weights_sel,                                       //
                      bias,                                              //
                      expert_first_token_offset + num_experts_per_node,  //
                      mQuantParams.wo.fc1_weight_scales,                 //
                      mQuantParams.fp8.dequant_fc1,                      //
                      mQuantParams.fp8_mxfp4.fc2.act_global_scale
                          ? mQuantParams.fp8_mxfp4.fc2.act_global_scale
                          : mQuantParams.fp8.quant_fc2,  //
                      fp4_act_scale_flat,                //
                      fp4_act_scale_flat,                //
                      mQuantParams,                      //
                      original_num_tokens,               //
                      expanded_num_tokens,               //
                      mExpertHiddenSize,                 //
                      mExpertInterSize,                  //
                      num_experts_per_node,              //
                      ActivationParams(mActivationType, swiglu_alpha, swiglu_beta, swiglu_limit),
                      alpha_scale_ptr_array,                   //
                      !mUseLora,                               //
                      /*use_deepseek_fp8_block_scale=*/false,  //
                      stream,                                  //
                      tactic,                                  //
                      mMinLatencyMode,                         //
                      num_active_experts_per_node,             //
                      active_expert_global_ids,                //
                      enable_pdl);                             //
  } else {
    TLLM_CHECK(mGemmToProfile == GemmToProfile::GEMM_2);
    mInterface->gemm2(input,                                           //
                      intermediate,                                    //
                      output,                                          //
                      expert_first_token_offset,                       //
                      tma_ws_input_template,                           //
                      weights_sel,                                     //
                      bias,                                            //
                      mQuantParams.wo.fc2_weight_scales,               //
                      mQuantParams.fp8.dequant_fc2,                    //
                      fp4_act_scale_flat,                              //
                      mQuantParams,                                    //
                      token_topk_unpermuted_scales,                    //
                      token_topk_permuted_scales,                      //
                      unpermuted_row_to_permuted_row,                  //
                      permuted_row_to_unpermuted_row,                  //
                      token_selected_experts,                          //
                      expert_first_token_offset + mNumExpertsPerNode,  //
                      original_num_tokens,                             //
                      expanded_num_tokens,                             //
                      mExpertHiddenSize,                               //
                      mExpertUnpaddedHiddenSize,                       //
                      mExpertInterSize,                                //
                      num_experts_per_node,                            //
                      mK,                                              //
                      alpha_scale_ptr_array,                           //
                      false,                                           //
                      nullptr,                                         //
                      /*use_deepseek_fp8_block_scale=*/false,          //
                      stream,                                          //
                      mParallelismConfig,                              //
                      mEnableAlltoall,                                 //
                      tactic,                                          //
                      mMinLatencyMode,                                 //
                      num_active_experts_per_node,                     //
                      active_expert_global_ids,                        //
                      enable_pdl);                                     //
  }
  mInterface->is_profiler = false;

  sync_check_cuda_error(stream);
}

}  // namespace tensorrt_llm::kernels::cutlass_kernels
