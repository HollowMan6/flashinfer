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

template <typename T>
__global__ void lora_low_rank_kernel(T const* input, int64_t num_tokens, int64_t hidden_size,
                                     int max_low_rank, int num_lora_modules,
                                     int32_t const* lora_ranks, void const* const* lora_weight_ptrs,
                                     int weight_index, T* low_rank_workspace) {
  size_t const total = static_cast<size_t>(num_tokens) * static_cast<size_t>(num_lora_modules) *
                       static_cast<size_t>(max_low_rank);
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += blockDim.x * gridDim.x) {
    int const rank_idx = static_cast<int>(idx % max_low_rank);
    size_t const token_module = idx / max_low_rank;
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
      low_rank[rank_idx] = from_float<T>(0.0f);
      continue;
    }

    lora_a += static_cast<size_t>(weight_index) * static_cast<size_t>(hidden_size) *
              static_cast<size_t>(max_low_rank);

    float acc = 0.0f;
    T const* input_row = input + token * hidden_size;
    T const* lora_a_row = lora_a + static_cast<size_t>(rank_idx) * hidden_size;
    for (int64_t hidden_idx = 0; hidden_idx < hidden_size; ++hidden_idx) {
      acc += to_float(input_row[hidden_idx]) * to_float(lora_a_row[hidden_idx]);
    }
    low_rank[rank_idx] = from_float<T>(acc);
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

  int constexpr threads = 256;
  size_t const low_rank_total = num_rank_entries * static_cast<size_t>(max_low_rank);
  int const low_rank_blocks =
      static_cast<int>(std::min<size_t>((low_rank_total + threads - 1) / threads, 65535));
  lora_low_rank_kernel<T><<<low_rank_blocks, threads, 0, stream>>>(
      static_cast<T const*>(input), num_tokens, in_hidden_size, max_low_rank, num_lora_modules,
      device_lora_ranks, device_lora_weight_ptrs, weight_index, low_rank_workspace);

  for (int module = 0; module < num_lora_modules; ++module) {
    int64_t const out_hidden_size = out_hidden_sizes[module];
    size_t const output_total =
        static_cast<size_t>(num_tokens) * static_cast<size_t>(out_hidden_size);
    int const output_blocks =
        static_cast<int>(std::min<size_t>((output_total + threads - 1) / threads, 65535));
    lora_output_kernel<T><<<output_blocks, threads, 0, stream>>>(
        low_rank_workspace, num_tokens, out_hidden_size, max_low_rank, module, device_lora_ranks,
        device_lora_weight_ptrs, weight_index, static_cast<T*>(outputs[module]));
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

  int constexpr threads = 256;
  size_t const low_rank_total = num_rank_entries * static_cast<size_t>(max_low_rank);
  int const low_rank_blocks =
      static_cast<int>(std::min<size_t>((low_rank_total + threads - 1) / threads, 65535));
  lora_low_rank_kernel<T><<<low_rank_blocks, threads, 0, stream>>>(
      static_cast<T const*>(input), num_tokens, in_hidden_size, max_low_rank, num_lora_modules,
      device_lora_ranks, device_lora_weight_ptrs, weight_index, low_rank_workspace);

  for (int module = 0; module < num_lora_modules; ++module) {
    int64_t const out_hidden_size = out_hidden_sizes[module];
    size_t const output_total =
        static_cast<size_t>(num_tokens) * static_cast<size_t>(out_hidden_size);
    int const output_blocks =
        static_cast<int>(std::min<size_t>((output_total + threads - 1) / threads, 65535));
    lora_output_kernel<T><<<output_blocks, threads, 0, stream>>>(
        low_rank_workspace, num_tokens, out_hidden_size, max_low_rank, module, device_lora_ranks,
        device_lora_weight_ptrs, weight_index, static_cast<T*>(outputs[module]));
  }
}

}  // namespace lora
}  // namespace group_gemm
}  // namespace flashinfer

#endif  // FLASHINFER_GROUP_GEMM_LORA_CUH_
