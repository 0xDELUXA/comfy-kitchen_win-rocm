// SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Launchers called from more than one translation unit. They are extern "C", so a
// declaration that drifts from its definition still links and then corrupts the
// call frame; declaring them once where the definition can see it makes the
// compiler check instead.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

extern "C" {

// ldc is c's row stride, so a caller writing an N-column slice of a wider output
// passes that output's width; a whole GEMM passes N.
void launch_int8_gemm_kernel(const void* a, const void* b, void* c, const void* scale_a,
                             const void* scale_b, int scale_b_stride, const void* bias,
                             int bias_code, int M, int N, int K, int ldc, int out_code,
                             hipStream_t stream);

// scale_code is a DTYPE_TO_CODE value: 0 float32, 5 e4m3 (passed as raw bytes).
// codebook is 16 floats, or null for the uniform levels.
void launch_dequant_int4_grouped_to_int8_kernel(const void* qw, const void* s_rel, int scale_code,
                                                const void* codebook, void* out, int64_t n,
                                                int64_t k, int group_size, hipStream_t stream);

void launch_w4a8_int8_gemm_chunked_kernel(const void* xq, const void* qw, const void* s_rel,
                                          int scale_code, const void* codebook,
                                          const void* s_channel, const void* xs, const void* bias,
                                          int bias_code, void* workspace, void* out, int M, int N,
                                          int K, int group_size, int chunk_cols, int out_code,
                                          hipStream_t stream);

}  // extern "C"
