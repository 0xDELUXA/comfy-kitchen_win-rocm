// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Grouped INT4 -> INT8 weight decode for the W4A8 path. The per-group scale is
// folded in here so the result feeds the WMMA int8 GEMM directly; the
// per-channel scale stays for that GEMM's epilogue. Memory-bound.
#include <stdexcept>

#include "../fp8_utils.h"
#include "../launchers.h"

namespace comfy::hip_backend {

// Bytes per element of a DTYPE_TO_CODE value. The chunk loop strides three
// different operands, so one table keeps them from drifting apart.
inline int64_t dtype_size(int code) {
    switch (code) {
        case 0: return 4;          // float32
        case 1:
        case 2: return 2;          // float16, bfloat16
        default: return 1;         // uint8 storage, which is how fp8 crosses
    }
}

// The group scale is stored as fp32 or as raw e4m3 bytes.
template <typename ScaleT>
__forceinline__ __device__ float load_group_scale(ScaleT v);

template <>
__forceinline__ __device__ float load_group_scale<float>(float v) {
    return v;
}

template <>
__forceinline__ __device__ float load_group_scale<uint8_t>(uint8_t v) {
    return fp8_to_float(v);
}

__forceinline__ __device__ int8_t dequant_round(float v) {
    // rintf is round-half-to-even, matching torch.round.
    int q = static_cast<int>(rintf(v));
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    return static_cast<int8_t>(q);
}

// One thread reads 8 packed bytes and writes 16 int8. group_size is 4, 8, 16 or
// a multiple of 16, so those 16 columns span at most four groups: load each of
// those scales once instead of re-reading one per output pair.
template <typename ScaleT>
__global__ void dequant_int4_grouped_to_int8_kernel(
    const int8_t* __restrict__ qw, const ScaleT* __restrict__ s_rel,
    const float* __restrict__ codebook, int8_t* __restrict__ out,
    int64_t vec_count, int k_half, int k, int group_size) {

    __shared__ float cb[16];
    if (codebook && threadIdx.x < 16) {
        cb[threadIdx.x] = codebook[threadIdx.x];
    }
    if (codebook) {
        __syncthreads();
    }

    const int64_t vec = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (vec >= vec_count) {
        return;
    }

    const int vec_per_row = k_half / 8;
    const int64_t row = vec / vec_per_row;
    const int byte_base = static_cast<int>(vec % vec_per_row) * 8;
    const int col_base = byte_base * 2;

    const uint2 packed = *reinterpret_cast<const uint2*>(&qw[row * k_half + byte_base]);
    const unsigned words[2] = {packed.x, packed.y};

    const int64_t scale_row = row * (k / group_size);
    const int group_base = col_base / group_size;
    const float sc0 = load_group_scale<ScaleT>(s_rel[scale_row + group_base]);
    float sc1 = sc0, sc2 = sc0, sc3 = sc0;
    if (group_size < 16) {
        sc1 = load_group_scale<ScaleT>(s_rel[scale_row + group_base + 1]);
        if (group_size < 8) {
            sc2 = load_group_scale<ScaleT>(s_rel[scale_row + group_base + 2]);
            sc3 = load_group_scale<ScaleT>(s_rel[scale_row + group_base + 3]);
        }
    }

    alignas(16) int8_t decoded[16];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
#pragma unroll
        for (int b = 0; b < 4; ++b) {
            const int pair = w * 4 + b;
            const int local = (group_size >= 16) ? 0 : ((pair * 2) / group_size);
            const float s = (local == 0) ? sc0 : (local == 1 ? sc1 : (local == 2 ? sc2 : sc3));
            const unsigned byte = (words[w] >> (b * 8)) & 0xFFu;
            const unsigned lo = byte & 0xFu;
            const unsigned hi = (byte >> 4) & 0xFu;
            const float v0 = codebook ? cb[lo] : (static_cast<float>(lo) - 8.0f);
            const float v1 = codebook ? cb[hi] : (static_cast<float>(hi) - 8.0f);
            decoded[pair * 2] = dequant_round(v0 * s);
            decoded[pair * 2 + 1] = dequant_round(v1 * s);
        }
    }
    *reinterpret_cast<uint4*>(&out[row * k + col_base]) =
        *reinterpret_cast<const uint4*>(decoded);
}

}  // namespace comfy::hip_backend

// Decode the weight in column chunks, running the INT8 GEMM on each chunk before
// the next is decoded, so a chunk is still cached when the GEMM reads it back
// instead of the whole [N, K] INT8 weight round-tripping through global memory.
extern "C" void launch_w4a8_int8_gemm_chunked_kernel(
    const void* xq, const void* qw, const void* s_rel, int scale_code, const void* codebook,
    const void* s_channel, const void* xs, const void* bias, int bias_code, void* workspace,
    void* out, int M, int N, int K, int group_size, int chunk_cols, int out_code,
    hipStream_t stream) {

    using namespace comfy::hip_backend;
    if (chunk_cols <= 0) {
        throw std::runtime_error("w4a8_int8_gemm_chunked: chunk_cols must be positive");
    }
    const int64_t k_half = K / 2;
    const int64_t scale_row = (group_size > 0) ? K / group_size : 0;

    for (int n0 = 0; n0 < N; n0 += chunk_cols) {
        const int cols = (chunk_cols < N - n0) ? chunk_cols : (N - n0);
        launch_dequant_int4_grouped_to_int8_kernel(
            static_cast<const int8_t*>(qw) + n0 * k_half,
            static_cast<const char*>(s_rel) + n0 * scale_row * dtype_size(scale_code), scale_code,
            codebook, workspace, cols, K, group_size, stream);
        launch_int8_gemm_kernel(
            xq, workspace, static_cast<char*>(out) + n0 * dtype_size(out_code), xs,
            static_cast<const float*>(s_channel) + n0, 1,
            bias ? static_cast<const char*>(bias) + n0 * dtype_size(bias_code) : nullptr,
            bias_code, M, cols, K, N /*ldc*/, out_code, stream);
    }
}

// scale_code is a DTYPE_TO_CODE value: 0 float32, 5 e4m3 (passed as raw bytes).
// codebook is 16 floats, or null for the uniform levels (code - 8).
extern "C" void launch_dequant_int4_grouped_to_int8_kernel(
    const void* qw, const void* s_rel, int scale_code, const void* codebook, void* out,
    int64_t n, int64_t k, int group_size, hipStream_t stream) {

    using namespace comfy::hip_backend;
    // The vector load reads 8 packed bytes and the store writes 16 columns, so a
    // row that is not a whole number of vectors would run into the next one.
    if (k % 16 != 0) {
        throw std::runtime_error("dequant_int4_grouped_to_int8: K must be divisible by 16");
    }
    // Four scales are loaded per vector, so a narrower group would span more.
    if (group_size < 4 || k % group_size != 0 ||
        (16 % group_size != 0 && group_size % 16 != 0)) {
        throw std::runtime_error(
            "dequant_int4_grouped_to_int8: group_size must be >=4 and divide 16 or be a "
            "multiple of 16");
    }
    if (n == 0 || k == 0) {
        return;  // a zero-block launch is hipErrorInvalidConfiguration
    }

    const int k_half = static_cast<int>(k / 2);
    const int64_t vec_count = n * (k_half / 8);
    constexpr int threads = 256;
    const int64_t blocks = (vec_count + threads - 1) / threads;

    if (scale_code == kFp8E4M3Code) {
        dequant_int4_grouped_to_int8_kernel<uint8_t>
            <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                static_cast<const int8_t*>(qw), static_cast<const uint8_t*>(s_rel),
                static_cast<const float*>(codebook), static_cast<int8_t*>(out), vec_count,
                k_half, static_cast<int>(k), group_size);
    } else {
        dequant_int4_grouped_to_int8_kernel<float>
            <<<static_cast<unsigned int>(blocks), threads, 0, stream>>>(
                static_cast<const int8_t*>(qw), static_cast<const float*>(s_rel),
                static_cast<const float*>(codebook), static_cast<int8_t*>(out), vec_count,
                k_half, static_cast<int>(k), group_size);
    }
}
