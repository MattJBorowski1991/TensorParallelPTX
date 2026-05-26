#pragma once
#include <cuda_fp16.h>
#include <cstring>

// ── 2D mesh rank decomposition ────────────────────────────────────────────────
// Works for any (tp_rows x tp_cols) mesh: 1x1, 2x2, 4x4, 8x8, etc.
//
// Global rank layout (row-major numbering):
//
//          col 0      col 1    ...   col (tp_cols-1)
//  row 0 |  rank 0  |  rank 1  | ... |  rank tp_cols-1
//  row 1 |  rank tc |  ...     | ...
//  ...
//
// rank_row = gpu_rank / tp_cols    <- which M-strip this GPU owns
// rank_col = gpu_rank % tp_cols    <- which N/K-strip this GPU owns

struct RankCoord {
    int row;   // 0 .. tp_rows-1
    int col;   // 0 .. tp_cols-1
};

inline RankCoord rank_to_coord(int gpu_rank, int tp_cols) {
    return { gpu_rank / tp_cols, gpu_rank % tp_cols };
}

inline int coord_to_rank(int rank_row, int rank_col, int tp_cols) {
    return rank_row * tp_cols + rank_col;
}

// ── Shard geometry ─────────────────────────────────────────────────────────────
// For a global problem (M, N, K) on a (tp_rows x tp_cols) mesh:
//
//   A  [M  x K ] → GPU(i,j) owns rows [i*lM, (i+1)*lM), K-cols [j*lK, (j+1)*lK)
//   B  [K  x N ] → GPU(i,j) owns K-rows [j*lK, (j+1)*lK), N-cols [j*lN, (j+1)*lN)
//   C  [M  x N ] → GPU(i,j) owns rows [i*lM, (i+1)*lM), N-cols [j*lN, (j+1)*lN)
//
// where lM = M/tp_rows, lN = N/tp_cols, lK = K/tp_cols.
//
// A and B shards are non-contiguous submatrices in row-major layout, so they
// must be packed into contiguous buffers before being uploaded to device.

// ── pack_shard_A ──────────────────────────────────────────────────────────────
// Copies A[rank_row*lM : (rank_row+1)*lM,  rank_col*lK : (rank_col+1)*lK]
// (non-contiguous rows from global A) into a contiguous buffer dst[lM x lK].
// Handles batches: A_global is [num_batches x M x K].
inline void pack_shard_A(
    const half* A_global,
    half*        dst,
    int M, int K, int num_batches,
    int local_M, int local_K,
    RankCoord coord)
{
    const int row_offset = coord.row * local_M;
    const int col_offset = coord.col * local_K;
    for (int b = 0; b < num_batches; ++b) {
        const half* A_b = A_global + (size_t)b * M * K;
        half*        D_b = dst       + (size_t)b * local_M * local_K;
        for (int m = 0; m < local_M; ++m) {
            // Source row in global A, starting at the K-shard column
            const half* src = A_b + (size_t)(row_offset + m) * K + col_offset;
            half*        d   = D_b + (size_t)m * local_K;
            memcpy(d, src, local_K * sizeof(half));
        }
    }
}

// ── pack_shard_B ──────────────────────────────────────────────────────────────
// Copies B[rank_col*lK : (rank_col+1)*lK,  rank_col*lN : (rank_col+1)*lN]
// into a contiguous buffer dst[lK x lN].
// Handles batches: B_global is [num_batches x K x N].
inline void pack_shard_B(
    const half* B_global,
    half*        dst,
    int K, int N, int num_batches,
    int local_K, int local_N,
    RankCoord coord)
{
    const int row_offset = coord.col * local_K;   // B's row = K-axis, split by col rank
    const int col_offset = coord.col * local_N;   // B's col = N-axis, split by col rank
    for (int b = 0; b < num_batches; ++b) {
        const half* B_b = B_global + (size_t)b * K * N;
        half*        D_b = dst       + (size_t)b * local_K * local_N;
        for (int k = 0; k < local_K; ++k) {
            const half* src = B_b + (size_t)(row_offset + k) * N + col_offset;
            half*        d   = D_b + (size_t)k * local_N;
            memcpy(d, src, local_N * sizeof(half));
        }
    }
}

// ── unpack_shard_C ────────────────────────────────────────────────────────────
// Writes a contiguous C shard [lM x lN] back into its position in the global
// C [M x N] output.  Handles batches: C_global is [num_batches x M x N].
inline void unpack_shard_C(
    const float* src,
    float*        C_global,
    int M, int N, int num_batches,
    int local_M, int local_N,
    RankCoord coord)
{
    const int row_offset = coord.row * local_M;
    const int col_offset = coord.col * local_N;
    for (int b = 0; b < num_batches; ++b) {
        const float* S_b = src      + (size_t)b * local_M * local_N;
        float*        C_b = C_global + (size_t)b * M * N;
        for (int m = 0; m < local_M; ++m) {
            const float* s = S_b + (size_t)m * local_N;
            float*        d = C_b + (size_t)(row_offset + m) * N + col_offset;
            memcpy(d, s, local_N * sizeof(float));
        }
    }
}
