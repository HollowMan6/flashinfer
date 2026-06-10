/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_GROUP_GEMM_LORA_CUH_
#define FLASHINFER_GROUP_GEMM_LORA_CUH_

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace flashinfer {
namespace group_gemm {
namespace lora {

inline size_t align16(size_t value) { return ((value + 15) / 16) * 16; }

inline size_t get_workspace_size(int64_t num_tokens, int num_lora_modules, int max_low_rank,
                                 size_t dtype_size) {
  size_t const num_rank_entries =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules);
  size_t const rank_size = align16(num_rank_entries * sizeof(int32_t));
  size_t const ptr_size = align16(num_rank_entries * 2 * sizeof(void const*));
  size_t const low_rank_size =
      align16(num_rank_entries * static_cast<size_t>(max_low_rank) * dtype_size);
  return rank_size + ptr_size + low_rank_size;
}

inline size_t get_device_workspace_size(int64_t num_tokens, int num_lora_modules, int max_low_rank,
                                        size_t dtype_size) {
  size_t const num_rank_entries =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules);
  return align16(num_rank_entries * static_cast<size_t>(max_low_rank) * dtype_size);
}

inline char* advance_aligned(char*& ptr, size_t size) {
  char* out = ptr;
  ptr += align16(size);
  return out;
}

template <typename T>
__device__ __forceinline__ float to_float(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float to_float<half>(half value) {
  return __half2float(value);
}

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename T>
__device__ __forceinline__ T from_float(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half from_float<half>(float value) {
  return __float2half(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <typename T>
__global__ void lora_low_rank_kernel(T const* input, int64_t num_tokens, int64_t hidden_size,
                                     int max_low_rank, int num_lora_modules,
                                     int32_t const* lora_ranks, void const* const* lora_weight_ptrs,
                                     int weight_index, T* low_rank_workspace) {
  size_t const total = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules) *
                       static_cast<size_t>(max_low_rank);
  __shared__ float warp_sums[32];

  for (size_t idx = blockIdx.x; idx < total; idx += gridDim.x) {
    int const rank_idx = static_cast<int>(idx % static_cast<size_t>(max_low_rank));
    size_t const token_module = idx / static_cast<size_t>(max_low_rank);
    int64_t const token = static_cast<int64_t>(token_module % static_cast<size_t>(num_tokens));
    int const module = static_cast<int>(token_module / static_cast<size_t>(num_tokens));
    size_t const rank_offset =
        static_cast<size_t>(module) * static_cast<size_t>(num_tokens) + static_cast<size_t>(token);

    int32_t const raw_lora_rank = lora_ranks[rank_offset];
    int const lora_rank =
        raw_lora_rank <= 0 ? 0 : (raw_lora_rank < max_low_rank ? raw_lora_rank : max_low_rank);
    T* low_rank = low_rank_workspace + rank_offset * max_low_rank;
    if (rank_idx >= lora_rank || lora_rank <= 0) {
      continue;
    }

    auto const* lora_a = reinterpret_cast<T const*>(lora_weight_ptrs[rank_offset * 2]);
    if (lora_a == nullptr) {
      if (threadIdx.x == 0) {
        low_rank[rank_idx] = from_float<T>(0.0f);
      }
      continue;
    }

    lora_a += static_cast<size_t>(weight_index) * static_cast<size_t>(hidden_size) *
              static_cast<size_t>(max_low_rank);

    float acc = 0.0f;
    T const* input_row = input + token * hidden_size;
    T const* lora_a_row = lora_a + static_cast<size_t>(rank_idx) * hidden_size;
    for (int64_t hidden_idx = threadIdx.x; hidden_idx < hidden_size; hidden_idx += blockDim.x) {
      acc += to_float(input_row[hidden_idx]) * to_float(lora_a_row[hidden_idx]);
    }

    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
    acc = warp_reduce_sum(acc);
    if (lane == 0) {
      warp_sums[warp] = acc;
    }
    __syncthreads();

    float block_acc = threadIdx.x < (blockDim.x >> 5) ? warp_sums[lane] : 0.0f;
    if (warp == 0) {
      block_acc = warp_reduce_sum(block_acc);
      if (lane == 0) {
        low_rank[rank_idx] = from_float<T>(block_acc);
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void lora_low_rank16_kernel(T const* input, int64_t num_tokens, int64_t hidden_size,
                                       int num_lora_modules, int32_t const* lora_ranks,
                                       void const* const* lora_weight_ptrs, int weight_index,
                                       T* low_rank_workspace) {
  size_t const total =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules);
  __shared__ float warp_sums[16][32];

  for (size_t token_module = blockIdx.x; token_module < total; token_module += gridDim.x) {
    int64_t const token = static_cast<int64_t>(token_module % static_cast<size_t>(num_tokens));
    int const module = static_cast<int>(token_module / static_cast<size_t>(num_tokens));
    size_t const rank_offset =
        static_cast<size_t>(module) * static_cast<size_t>(num_tokens) + static_cast<size_t>(token);

    int32_t const raw_lora_rank = lora_ranks[rank_offset];
    int const lora_rank = raw_lora_rank <= 0 ? 0 : (raw_lora_rank < 16 ? raw_lora_rank : 16);
    T* low_rank = low_rank_workspace + rank_offset * 16;
    if (lora_rank <= 0) {
      continue;
    }

    auto const* lora_a = reinterpret_cast<T const*>(lora_weight_ptrs[rank_offset * 2]);
    if (lora_a == nullptr) {
      if (threadIdx.x < 16) {
        low_rank[threadIdx.x] = from_float<T>(0.0f);
      }
      continue;
    }

    lora_a += static_cast<size_t>(weight_index) * static_cast<size_t>(hidden_size) * 16;

    float acc[16] = {};
    T const* input_row = input + token * hidden_size;
    for (int64_t hidden_idx = threadIdx.x; hidden_idx < hidden_size; hidden_idx += blockDim.x) {
      float const input_value = to_float(input_row[hidden_idx]);
      if (lora_rank == 16) {
#pragma unroll
        for (int rank_idx = 0; rank_idx < 16; ++rank_idx) {
          T const* lora_a_row = lora_a + static_cast<size_t>(rank_idx) * hidden_size;
          acc[rank_idx] += input_value * to_float(lora_a_row[hidden_idx]);
        }
      } else {
#pragma unroll
        for (int rank_idx = 0; rank_idx < 16; ++rank_idx) {
          if (rank_idx < lora_rank) {
            T const* lora_a_row = lora_a + static_cast<size_t>(rank_idx) * hidden_size;
            acc[rank_idx] += input_value * to_float(lora_a_row[hidden_idx]);
          }
        }
      }
    }

    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
#pragma unroll
    for (int rank_idx = 0; rank_idx < 16; ++rank_idx) {
      acc[rank_idx] = warp_reduce_sum(acc[rank_idx]);
      if (lane == 0) {
        warp_sums[rank_idx][warp] = acc[rank_idx];
      }
    }
    __syncthreads();

    if (warp == 0) {
      int const num_warps = blockDim.x >> 5;
#pragma unroll
      for (int rank_idx = 0; rank_idx < 16; ++rank_idx) {
        float block_acc = lane < num_warps ? warp_sums[rank_idx][lane] : 0.0f;
        block_acc = warp_reduce_sum(block_acc);
        if (lane == 0 && rank_idx < lora_rank) {
          low_rank[rank_idx] = from_float<T>(block_acc);
        }
      }
    }
    __syncthreads();
  }
}

__global__ void lora_low_rank16_bf16x2_kernel(
    __nv_bfloat16 const* input, int64_t num_tokens, int64_t hidden_size, int num_lora_modules,
    int32_t const* lora_ranks, void const* const* lora_weight_ptrs, int weight_index,
    __nv_bfloat16* low_rank_workspace) {
  size_t const total =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules);
  int64_t const hidden_pairs = hidden_size >> 1;
  __shared__ float warp_sums[16][32];

  for (size_t token_module = blockIdx.x; token_module < total; token_module += gridDim.x) {
    int64_t const token = static_cast<int64_t>(token_module % static_cast<size_t>(num_tokens));
    int const module = static_cast<int>(token_module / static_cast<size_t>(num_tokens));
    size_t const rank_offset =
        static_cast<size_t>(module) * static_cast<size_t>(num_tokens) + static_cast<size_t>(token);

    int32_t const raw_lora_rank = lora_ranks[rank_offset];
    int const lora_rank = raw_lora_rank <= 0 ? 0 : (raw_lora_rank < 16 ? raw_lora_rank : 16);
    __nv_bfloat16* low_rank = low_rank_workspace + rank_offset * 16;
    if (lora_rank <= 0) {
      continue;
    }

    auto const* lora_a =
        reinterpret_cast<__nv_bfloat16 const*>(lora_weight_ptrs[rank_offset * 2]);
    if (lora_a == nullptr) {
      if (threadIdx.x < 16) {
        low_rank[threadIdx.x] = from_float<__nv_bfloat16>(0.0f);
      }
      continue;
    }

    lora_a += static_cast<size_t>(weight_index) * static_cast<size_t>(hidden_size) * 16;

    float acc[16] = {};
    auto const* input_row =
        reinterpret_cast<__nv_bfloat162 const*>(input + token * hidden_size);
    for (int64_t hidden_pair_idx = threadIdx.x; hidden_pair_idx < hidden_pairs;
         hidden_pair_idx += blockDim.x) {
      float2 const input_value = __bfloat1622float2(input_row[hidden_pair_idx]);
      if (lora_rank == 16) {
#pragma unroll
        for (int rank_idx = 0; rank_idx < 16; ++rank_idx) {
          auto const* lora_a_row = reinterpret_cast<__nv_bfloat162 const*>(
              lora_a + static_cast<size_t>(rank_idx) * hidden_size);
          float2 const lora_value = __bfloat1622float2(lora_a_row[hidden_pair_idx]);
          acc[rank_idx] += input_value.x * lora_value.x + input_value.y * lora_value.y;
        }
      } else {
#pragma unroll
        for (int rank_idx = 0; rank_idx < 16; ++rank_idx) {
          if (rank_idx < lora_rank) {
            auto const* lora_a_row = reinterpret_cast<__nv_bfloat162 const*>(
                lora_a + static_cast<size_t>(rank_idx) * hidden_size);
            float2 const lora_value = __bfloat1622float2(lora_a_row[hidden_pair_idx]);
            acc[rank_idx] += input_value.x * lora_value.x + input_value.y * lora_value.y;
          }
        }
      }
    }

    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x >> 5;
#pragma unroll
    for (int rank_idx = 0; rank_idx < 16; ++rank_idx) {
      acc[rank_idx] = warp_reduce_sum(acc[rank_idx]);
      if (lane == 0) {
        warp_sums[rank_idx][warp] = acc[rank_idx];
      }
    }
    __syncthreads();

    if (warp == 0) {
      int const num_warps = blockDim.x >> 5;
#pragma unroll
      for (int rank_idx = 0; rank_idx < 16; ++rank_idx) {
        float block_acc = lane < num_warps ? warp_sums[rank_idx][lane] : 0.0f;
        block_acc = warp_reduce_sum(block_acc);
        if (lane == 0 && rank_idx < lora_rank) {
          low_rank[rank_idx] = from_float<__nv_bfloat16>(block_acc);
        }
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void lora_output_kernel(T const* low_rank_workspace, int64_t num_tokens,
                                   int64_t out_hidden_size, int max_low_rank, int module,
                                   int32_t const* lora_ranks, void const* const* lora_weight_ptrs,
                                   int weight_index, T* output) {
  size_t const total = static_cast<size_t>(num_tokens) * static_cast<size_t>(out_hidden_size);
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int64_t const out_idx = static_cast<int64_t>(idx % static_cast<size_t>(out_hidden_size));
    int64_t const token = static_cast<int64_t>(idx / static_cast<size_t>(out_hidden_size));
    size_t const rank_offset =
        static_cast<size_t>(module) * static_cast<size_t>(num_tokens) + static_cast<size_t>(token);
    int32_t const raw_lora_rank = lora_ranks[rank_offset];
    int const lora_rank =
        raw_lora_rank <= 0 ? 0 : (raw_lora_rank < max_low_rank ? raw_lora_rank : max_low_rank);
    if (lora_rank <= 0) {
      output[token * out_hidden_size + out_idx] = from_float<T>(0.0f);
      continue;
    }

    auto const* lora_b = reinterpret_cast<T const*>(lora_weight_ptrs[rank_offset * 2 + 1]);
    if (lora_b == nullptr) {
      output[token * out_hidden_size + out_idx] = from_float<T>(0.0f);
      continue;
    }

    lora_b += static_cast<size_t>(weight_index) * static_cast<size_t>(out_hidden_size) *
              static_cast<size_t>(max_low_rank);

    float acc = 0.0f;
    T const* low_rank = low_rank_workspace + rank_offset * max_low_rank;
    T const* lora_b_row = lora_b + static_cast<size_t>(out_idx) * max_low_rank;
    for (int rank_idx = 0; rank_idx < lora_rank; ++rank_idx) {
      acc += to_float(low_rank[rank_idx]) * to_float(lora_b_row[rank_idx]);
    }
    output[token * out_hidden_size + out_idx] = from_float<T>(acc);
  }
}

template <typename T>
__global__ void lora_output_strided_kernel(
    T const* low_rank_workspace, int64_t num_tokens, int64_t out_hidden_size, int64_t output_stride,
    int max_low_rank, int module, int32_t const* lora_ranks,
    void const* const* lora_weight_ptrs, int weight_index, T* output) {
  size_t const total = static_cast<size_t>(num_tokens) * static_cast<size_t>(out_hidden_size);
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int64_t const out_idx = static_cast<int64_t>(idx % static_cast<size_t>(out_hidden_size));
    int64_t const token = static_cast<int64_t>(idx / static_cast<size_t>(out_hidden_size));
    size_t const rank_offset =
        static_cast<size_t>(module) * static_cast<size_t>(num_tokens) + static_cast<size_t>(token);
    T* output_row = output + token * output_stride;
    int32_t const raw_lora_rank = lora_ranks[rank_offset];
    int const lora_rank =
        raw_lora_rank <= 0 ? 0 : (raw_lora_rank < max_low_rank ? raw_lora_rank : max_low_rank);
    if (lora_rank <= 0) {
      output_row[out_idx] = from_float<T>(0.0f);
      continue;
    }

    auto const* lora_b = reinterpret_cast<T const*>(lora_weight_ptrs[rank_offset * 2 + 1]);
    if (lora_b == nullptr) {
      output_row[out_idx] = from_float<T>(0.0f);
      continue;
    }

    lora_b += static_cast<size_t>(weight_index) * static_cast<size_t>(out_hidden_size) *
              static_cast<size_t>(max_low_rank);

    float acc = 0.0f;
    T const* low_rank = low_rank_workspace + rank_offset * max_low_rank;
    T const* lora_b_row = lora_b + static_cast<size_t>(out_idx) * max_low_rank;
    for (int rank_idx = 0; rank_idx < lora_rank; ++rank_idx) {
      acc += to_float(low_rank[rank_idx]) * to_float(lora_b_row[rank_idx]);
    }
    output_row[out_idx] = from_float<T>(acc);
  }
}

__global__ void lora_output_rank16_bf16x2_strided_kernel(
    __nv_bfloat16 const* low_rank_workspace, int64_t num_tokens, int64_t out_hidden_size,
    int64_t output_stride, int module, int32_t const* lora_ranks,
    void const* const* lora_weight_ptrs, int weight_index, __nv_bfloat16* output) {
  int64_t const out_hidden_pairs = out_hidden_size >> 1;
  size_t const total = static_cast<size_t>(num_tokens) * static_cast<size_t>(out_hidden_pairs);
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int64_t const out_pair = static_cast<int64_t>(idx % static_cast<size_t>(out_hidden_pairs));
    int64_t const out_idx = out_pair << 1;
    int64_t const token = static_cast<int64_t>(idx / static_cast<size_t>(out_hidden_pairs));
    size_t const rank_offset =
        static_cast<size_t>(module) * static_cast<size_t>(num_tokens) + static_cast<size_t>(token);
    __nv_bfloat16* output_row = output + token * output_stride;
    auto* output_row_pairs = reinterpret_cast<__nv_bfloat162*>(output_row);

    int32_t const raw_lora_rank = lora_ranks[rank_offset];
    int const lora_rank = raw_lora_rank <= 0 ? 0 : (raw_lora_rank < 16 ? raw_lora_rank : 16);
    if (lora_rank <= 0) {
      output_row_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
      continue;
    }

    auto const* lora_b =
        reinterpret_cast<__nv_bfloat16 const*>(lora_weight_ptrs[rank_offset * 2 + 1]);
    if (lora_b == nullptr) {
      output_row_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
      continue;
    }

    lora_b += static_cast<size_t>(weight_index) * static_cast<size_t>(out_hidden_size) * 16;

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    __nv_bfloat16 const* low_rank = low_rank_workspace + rank_offset * 16;
    __nv_bfloat16 const* lora_b_row0 = lora_b + static_cast<size_t>(out_idx) * 16;
    __nv_bfloat16 const* lora_b_row1 = lora_b_row0 + 16;
    auto const* low_rank_pairs = reinterpret_cast<__nv_bfloat162 const*>(low_rank);
    auto const* lora_b_row0_pairs = reinterpret_cast<__nv_bfloat162 const*>(lora_b_row0);
    auto const* lora_b_row1_pairs = reinterpret_cast<__nv_bfloat162 const*>(lora_b_row1);

    if (lora_rank == 16) {
#pragma unroll
      for (int rank_pair = 0; rank_pair < 8; ++rank_pair) {
        float2 const low_rank_value = __bfloat1622float2(low_rank_pairs[rank_pair]);
        float2 const lora_b0_value = __bfloat1622float2(lora_b_row0_pairs[rank_pair]);
        float2 const lora_b1_value = __bfloat1622float2(lora_b_row1_pairs[rank_pair]);
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
          float2 const lora_b0_value = __bfloat1622float2(lora_b_row0_pairs[rank_pair]);
          float2 const lora_b1_value = __bfloat1622float2(lora_b_row1_pairs[rank_pair]);
          acc0 += low_rank_value.x * lora_b0_value.x;
          acc0 += low_rank_value.y * lora_b0_value.y;
          acc1 += low_rank_value.x * lora_b1_value.x;
          acc1 += low_rank_value.y * lora_b1_value.y;
        }
      }
      if ((lora_rank & 1) != 0) {
        int const rank_idx = rank_pairs << 1;
        float const low_rank_value = __bfloat162float(low_rank[rank_idx]);
        acc0 += low_rank_value * __bfloat162float(lora_b_row0[rank_idx]);
        acc1 += low_rank_value * __bfloat162float(lora_b_row1[rank_idx]);
      }
    }
    output_row_pairs[out_pair] = __floats2bfloat162_rn(acc0, acc1);
  }
}

__global__ void lora_output_rank16_bf16x2_by_row_strided_kernel(
    __nv_bfloat16 const* low_rank_workspace, int64_t num_tokens, int64_t out_hidden_size,
    int64_t output_stride, int module, int32_t const* lora_ranks,
    void const* const* lora_weight_ptrs, int weight_index, __nv_bfloat16* output) {
  int64_t const out_hidden_pairs = out_hidden_size >> 1;
  __shared__ int shared_lora_rank;
  __shared__ __nv_bfloat16 const* shared_lora_b;
  __shared__ __nv_bfloat162 shared_low_rank_pairs[8];

  for (int64_t token = static_cast<int64_t>(blockIdx.x); token < num_tokens;
       token += static_cast<int64_t>(gridDim.x)) {
    size_t const rank_offset =
        static_cast<size_t>(module) * static_cast<size_t>(num_tokens) + static_cast<size_t>(token);
    if (threadIdx.x == 0) {
      int32_t const raw_lora_rank = lora_ranks[rank_offset];
      int lora_rank = raw_lora_rank <= 0 ? 0 : (raw_lora_rank < 16 ? raw_lora_rank : 16);
      auto const* lora_b =
          reinterpret_cast<__nv_bfloat16 const*>(lora_weight_ptrs[rank_offset * 2 + 1]);
      if (lora_rank <= 0 || lora_b == nullptr) {
        lora_rank = 0;
        lora_b = nullptr;
      } else {
        lora_b += static_cast<size_t>(weight_index) * static_cast<size_t>(out_hidden_size) * 16;
      }
      shared_lora_rank = lora_rank;
      shared_lora_b = lora_b;
    }
    __syncthreads();

    if (shared_lora_rank > 0 && threadIdx.x < 8) {
      auto const* low_rank =
          reinterpret_cast<__nv_bfloat162 const*>(low_rank_workspace + rank_offset * 16);
      shared_low_rank_pairs[threadIdx.x] = low_rank[threadIdx.x];
    }
    __syncthreads();

    __nv_bfloat16* output_row = output + token * output_stride;
    auto* output_row_pairs = reinterpret_cast<__nv_bfloat162*>(output_row);
    int const lora_rank = shared_lora_rank;
    auto const* lora_b = shared_lora_b;
    for (int64_t out_pair = threadIdx.x; out_pair < out_hidden_pairs;
         out_pair += static_cast<int64_t>(blockDim.x)) {
      if (lora_rank <= 0) {
        output_row_pairs[out_pair] = __floats2bfloat162_rn(0.0f, 0.0f);
        continue;
      }

      int64_t const out_idx = out_pair << 1;
      float acc0 = 0.0f;
      float acc1 = 0.0f;
      __nv_bfloat16 const* lora_b_row0 = lora_b + static_cast<size_t>(out_idx) * 16;
      __nv_bfloat16 const* lora_b_row1 = lora_b_row0 + 16;
      auto const* lora_b_row0_pairs = reinterpret_cast<__nv_bfloat162 const*>(lora_b_row0);
      auto const* lora_b_row1_pairs = reinterpret_cast<__nv_bfloat162 const*>(lora_b_row1);

      if (lora_rank == 16) {
#pragma unroll
        for (int rank_pair = 0; rank_pair < 8; ++rank_pair) {
          float2 const low_rank_value = __bfloat1622float2(shared_low_rank_pairs[rank_pair]);
          float2 const lora_b0_value = __bfloat1622float2(lora_b_row0_pairs[rank_pair]);
          float2 const lora_b1_value = __bfloat1622float2(lora_b_row1_pairs[rank_pair]);
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
            float2 const low_rank_value = __bfloat1622float2(shared_low_rank_pairs[rank_pair]);
            float2 const lora_b0_value = __bfloat1622float2(lora_b_row0_pairs[rank_pair]);
            float2 const lora_b1_value = __bfloat1622float2(lora_b_row1_pairs[rank_pair]);
            acc0 += low_rank_value.x * lora_b0_value.x;
            acc0 += low_rank_value.y * lora_b0_value.y;
            acc1 += low_rank_value.x * lora_b1_value.x;
            acc1 += low_rank_value.y * lora_b1_value.y;
          }
        }
        if ((lora_rank & 1) != 0) {
          int const rank_idx = rank_pairs << 1;
          float const low_rank_value =
              __bfloat162float(reinterpret_cast<__nv_bfloat16 const*>(shared_low_rank_pairs)[rank_idx]);
          acc0 += low_rank_value * __bfloat162float(lora_b_row0[rank_idx]);
          acc1 += low_rank_value * __bfloat162float(lora_b_row1[rank_idx]);
        }
      }
      output_row_pairs[out_pair] = __floats2bfloat162_rn(acc0, acc1);
    }
    __syncthreads();
  }
}

template <typename T>
void launch_lora_low_rank_kernel(int64_t num_tokens, int num_lora_modules,
                                 int64_t in_hidden_size, int max_low_rank, void const* input,
                                 int32_t const* device_lora_ranks,
                                 void const* const* device_lora_weight_ptrs, int weight_index,
                                 T* low_rank_workspace, cudaStream_t stream) {
  size_t const num_rank_entries =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules);
  int constexpr low_rank_threads = 128;
  int constexpr generic_threads = 256;

  if (max_low_rank == 16) {
    int const low_rank_blocks =
        static_cast<int>(std::min<size_t>(num_rank_entries, 2147483647));
    if constexpr (std::is_same<T, __nv_bfloat16>::value) {
      if ((in_hidden_size & 1) == 0) {
        lora_low_rank16_bf16x2_kernel<<<low_rank_blocks, low_rank_threads, 0, stream>>>(
            static_cast<__nv_bfloat16 const*>(input), num_tokens, in_hidden_size,
            num_lora_modules, device_lora_ranks, device_lora_weight_ptrs, weight_index,
            low_rank_workspace);
        return;
      }
    }
    lora_low_rank16_kernel<T><<<low_rank_blocks, low_rank_threads, 0, stream>>>(
        static_cast<T const*>(input), num_tokens, in_hidden_size, num_lora_modules,
        device_lora_ranks, device_lora_weight_ptrs, weight_index, low_rank_workspace);
  } else {
    size_t const low_rank_total = num_rank_entries * static_cast<size_t>(max_low_rank);
    int const low_rank_blocks =
        static_cast<int>(std::min<size_t>(low_rank_total, 2147483647));
    lora_low_rank_kernel<T><<<low_rank_blocks, generic_threads, 0, stream>>>(
        static_cast<T const*>(input), num_tokens, in_hidden_size, max_low_rank, num_lora_modules,
        device_lora_ranks, device_lora_weight_ptrs, weight_index, low_rank_workspace);
  }
}

template <typename T>
void launch_lora_output(int64_t num_tokens, int64_t out_hidden_size, int max_low_rank, int module,
                        T const* low_rank_workspace, int32_t const* device_lora_ranks,
                        void const* const* device_lora_weight_ptrs, int weight_index, T* output,
                        cudaStream_t stream) {
  int constexpr output_threads = 256;
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    if (max_low_rank == 16 && num_tokens >= 32768 && (out_hidden_size & 1) == 0) {
      int const output_blocks = static_cast<int>(std::min<int64_t>(num_tokens, 65535));
      lora_output_rank16_bf16x2_by_row_strided_kernel<<<output_blocks, output_threads, 0, stream>>>(
          low_rank_workspace, num_tokens, out_hidden_size, out_hidden_size, module,
          device_lora_ranks, device_lora_weight_ptrs, weight_index, output);
      return;
    }
    if (max_low_rank == 16 && (out_hidden_size & 1) == 0) {
      size_t const output_pair_total =
          static_cast<size_t>(num_tokens) * static_cast<size_t>(out_hidden_size >> 1);
      int const output_blocks = static_cast<int>(
          std::min<size_t>((output_pair_total + output_threads - 1) / output_threads, 65535));
      lora_output_rank16_bf16x2_strided_kernel<<<output_blocks, output_threads, 0, stream>>>(
          low_rank_workspace, num_tokens, out_hidden_size, out_hidden_size, module,
          device_lora_ranks, device_lora_weight_ptrs, weight_index, output);
      return;
    }
  }

  size_t const output_total =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(out_hidden_size);
  int const output_blocks = static_cast<int>(
      std::min<size_t>((output_total + output_threads - 1) / output_threads, 65535));
  lora_output_kernel<T><<<output_blocks, output_threads, 0, stream>>>(
      low_rank_workspace, num_tokens, out_hidden_size, max_low_rank, module, device_lora_ranks,
      device_lora_weight_ptrs, weight_index, output);
}

template <typename T>
void launch_lora_output_strided(int64_t num_tokens, int64_t out_hidden_size, int64_t output_stride,
                                int max_low_rank, int module, T const* low_rank_workspace,
                                int32_t const* device_lora_ranks,
                                void const* const* device_lora_weight_ptrs, int weight_index,
                                T* output, cudaStream_t stream) {
  int constexpr output_threads = 256;
  if constexpr (std::is_same<T, __nv_bfloat16>::value) {
    if (max_low_rank == 16 && num_tokens >= 32768 && (out_hidden_size & 1) == 0 &&
        (output_stride & 1) == 0) {
      int const output_blocks = static_cast<int>(std::min<int64_t>(num_tokens, 65535));
      lora_output_rank16_bf16x2_by_row_strided_kernel<<<output_blocks, output_threads, 0, stream>>>(
          low_rank_workspace, num_tokens, out_hidden_size, output_stride, module,
          device_lora_ranks, device_lora_weight_ptrs, weight_index, output);
      return;
    }
    if (max_low_rank == 16 && (out_hidden_size & 1) == 0 && (output_stride & 1) == 0) {
      size_t const output_pair_total =
          static_cast<size_t>(num_tokens) * static_cast<size_t>(out_hidden_size >> 1);
      int const output_blocks = static_cast<int>(
          std::min<size_t>((output_pair_total + output_threads - 1) / output_threads, 65535));
      lora_output_rank16_bf16x2_strided_kernel<<<output_blocks, output_threads, 0, stream>>>(
          low_rank_workspace, num_tokens, out_hidden_size, output_stride, module,
          device_lora_ranks, device_lora_weight_ptrs, weight_index, output);
      return;
    }
  }

  size_t const output_total =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(out_hidden_size);
  int const output_blocks = static_cast<int>(
      std::min<size_t>((output_total + output_threads - 1) / output_threads, 65535));
  lora_output_strided_kernel<T><<<output_blocks, output_threads, 0, stream>>>(
      low_rank_workspace, num_tokens, out_hidden_size, output_stride, max_low_rank, module,
      device_lora_ranks, device_lora_weight_ptrs, weight_index, output);
}

template <typename T>
void run_lora_sgmv(int64_t num_tokens, int num_lora_modules, int64_t in_hidden_size,
                   int const* out_hidden_sizes, int max_low_rank, void const* input,
                   int32_t const* host_lora_ranks, void const* const* host_lora_weight_ptrs,
                   int weight_index, void* const* outputs, void* workspace, cudaStream_t stream) {
  if (num_tokens == 0 || num_lora_modules == 0 || max_low_rank == 0) {
    return;
  }

  size_t const num_rank_entries =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules);
  char* workspace_ptr = static_cast<char*>(workspace);
  auto* device_lora_ranks = reinterpret_cast<int32_t*>(
      advance_aligned(workspace_ptr, num_rank_entries * sizeof(int32_t)));
  auto* device_lora_weight_ptrs = reinterpret_cast<void const**>(
      advance_aligned(workspace_ptr, num_rank_entries * 2 * sizeof(void const*)));
  auto* low_rank_workspace = reinterpret_cast<T*>(
      advance_aligned(workspace_ptr, num_rank_entries * max_low_rank * sizeof(T)));

  cudaMemcpyAsync(device_lora_ranks, host_lora_ranks, num_rank_entries * sizeof(int32_t),
                  cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(device_lora_weight_ptrs, host_lora_weight_ptrs,
                  num_rank_entries * 2 * sizeof(void const*), cudaMemcpyHostToDevice, stream);

  launch_lora_low_rank_kernel<T>(num_tokens, num_lora_modules, in_hidden_size, max_low_rank, input,
                                 device_lora_ranks, device_lora_weight_ptrs, weight_index,
                                 low_rank_workspace, stream);

  for (int module = 0; module < num_lora_modules; ++module) {
    int64_t const out_hidden_size = out_hidden_sizes[module];
    launch_lora_output<T>(num_tokens, out_hidden_size, max_low_rank, module, low_rank_workspace,
                          device_lora_ranks, device_lora_weight_ptrs, weight_index,
                          static_cast<T*>(outputs[module]), stream);
  }
}

template <typename T>
void run_lora_sgmv_device(int64_t num_tokens, int num_lora_modules, int64_t in_hidden_size,
                          int const* out_hidden_sizes, int max_low_rank, void const* input,
                          int32_t const* device_lora_ranks,
                          void const* const* device_lora_weight_ptrs, int weight_index,
                          void* const* outputs, void* workspace, cudaStream_t stream) {
  if (num_tokens == 0 || num_lora_modules == 0 || max_low_rank == 0) {
    return;
  }

  size_t const num_rank_entries =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules);
  char* workspace_ptr = static_cast<char*>(workspace);
  auto* low_rank_workspace = reinterpret_cast<T*>(
      advance_aligned(workspace_ptr, num_rank_entries * max_low_rank * sizeof(T)));

  launch_lora_low_rank_kernel<T>(num_tokens, num_lora_modules, in_hidden_size, max_low_rank, input,
                                 device_lora_ranks, device_lora_weight_ptrs, weight_index,
                                 low_rank_workspace, stream);

  for (int module = 0; module < num_lora_modules; ++module) {
    int64_t const out_hidden_size = out_hidden_sizes[module];
    launch_lora_output<T>(num_tokens, out_hidden_size, max_low_rank, module, low_rank_workspace,
                          device_lora_ranks, device_lora_weight_ptrs, weight_index,
                          static_cast<T*>(outputs[module]), stream);
  }
}

template <typename T>
void run_lora_sgmv_device_low_rank(int64_t num_tokens, int num_lora_modules,
                                   int64_t in_hidden_size, int max_low_rank, void const* input,
                                   int32_t const* device_lora_ranks,
                                   void const* const* device_lora_weight_ptrs, int weight_index,
                                   void* workspace, cudaStream_t stream) {
  if (num_tokens == 0 || num_lora_modules == 0 || max_low_rank == 0) {
    return;
  }

  size_t const num_rank_entries =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules);
  char* workspace_ptr = static_cast<char*>(workspace);
  auto* low_rank_workspace = reinterpret_cast<T*>(
      advance_aligned(workspace_ptr, num_rank_entries * max_low_rank * sizeof(T)));

  launch_lora_low_rank_kernel<T>(num_tokens, num_lora_modules, in_hidden_size, max_low_rank, input,
                                 device_lora_ranks, device_lora_weight_ptrs, weight_index,
                                 low_rank_workspace, stream);
}

template <typename T>
void run_lora_sgmv_device_strided_outputs(
    int64_t num_tokens, int num_lora_modules, int64_t in_hidden_size,
    int const* out_hidden_sizes, int64_t const* output_strides, int max_low_rank,
    void const* input, int32_t const* device_lora_ranks,
    void const* const* device_lora_weight_ptrs, int weight_index, void* const* outputs,
    void* workspace, cudaStream_t stream) {
  if (num_tokens == 0 || num_lora_modules == 0 || max_low_rank == 0) {
    return;
  }

  size_t const num_rank_entries =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules);
  char* workspace_ptr = static_cast<char*>(workspace);
  auto* low_rank_workspace = reinterpret_cast<T*>(
      advance_aligned(workspace_ptr, num_rank_entries * max_low_rank * sizeof(T)));

  launch_lora_low_rank_kernel<T>(num_tokens, num_lora_modules, in_hidden_size, max_low_rank, input,
                                 device_lora_ranks, device_lora_weight_ptrs, weight_index,
                                 low_rank_workspace, stream);

  for (int module = 0; module < num_lora_modules; ++module) {
    int64_t const out_hidden_size = out_hidden_sizes[module];
    int64_t const output_stride = output_strides[module];
    launch_lora_output_strided<T>(num_tokens, out_hidden_size, output_stride, max_low_rank, module,
                                  low_rank_workspace, device_lora_ranks, device_lora_weight_ptrs,
                                  weight_index, static_cast<T*>(outputs[module]), stream);
  }
}

}  // namespace lora
}  // namespace group_gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_LORA_CUH_
