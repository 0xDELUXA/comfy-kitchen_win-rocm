// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Shared INT4 -> INT8 weight decode for the W4A8 paths. The chunked decode
// writes the result to a workspace the WMMA GEMM reads back; the decode GEMV
// keeps it in registers. Both must land on the same int8 grid, so the rounding
// and the level table live here rather than in either kernel.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "fp8_utils.h"

namespace comfy::hip_backend {

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

// 8 packed bytes (16 int4 codes, low nibble = even column) -> 16 int8 packed as
// four dwords, column 0 in the low byte of x. cb is the 16-entry level table or
// null for the uniform (code - 8) levels. The 16 columns span one group when
// group_size >= 16, two at 8 and four at 4, so the caller hands in up to four
// scales; all four are equal in the common case.
__forceinline__ __device__ uint4 dequant16_int4_to_int8(
    uint2 packed, const float* __restrict__ cb, float sc0, float sc1, float sc2, float sc3,
    int group_size) {

    const unsigned words[2] = {packed.x, packed.y};
    unsigned d[4] = {0u, 0u, 0u, 0u};
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
            const float v0 = cb ? cb[lo] : (static_cast<float>(lo) - 8.0f);
            const float v1 = cb ? cb[hi] : (static_cast<float>(hi) - 8.0f);
            const unsigned q0 = static_cast<unsigned>(dequant_round(v0 * s)) & 0xFFu;
            const unsigned q1 = static_cast<unsigned>(dequant_round(v1 * s)) & 0xFFu;
            d[pair >> 1] |= (q0 << ((pair & 1) * 16)) | (q1 << ((pair & 1) * 16 + 8));
        }
    }
    return make_uint4(d[0], d[1], d[2], d[3]);
}

}  // namespace comfy::hip_backend
