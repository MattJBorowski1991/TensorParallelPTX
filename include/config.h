#pragma once

// ── Warp / tile constants (compile-time) ────────────────────────────────────
constexpr int THREADS_PER_WARP = 32;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define PAD    0

#define WARP_TILES_X    4
#define WARP_TILES_Y    2
#define WARPS_PER_BLOCK (WARP_TILES_X * WARP_TILES_Y)

// ── 2D Tensor Parallelism configuration ───────────────────────────────────────
// For 2D TP: GPU mesh is (TP_ROW x TP_COL).
// Each GPU (rank) holds tensors partitioned as:
//   - A:  (M / TP_ROW) x (K / TP_COL)
//   - B:  (K / TP_COL) x (N / TP_COL)  
//   - C:  (M / TP_ROW) x (N / TP_COL)
// This requires:
//   1. Allgather/Allreduce on K-dimension (for A and B shards)
//   2. Allreduce on C output to sum partial results

// ── Runtime problem / benchmark configuration ────────────────────────────────────
struct GemmConfig {
    // Global problem dimensions
    int M, N, K;
    int num_batches;
    int warmups, runs;
    
    // Tensor Parallelism configuration
    int tp_rows;    // number of ranks in row dimension (splitting M and A)
    int tp_cols;    // number of ranks in column dimension (splitting N, B, and K)
    int gpu_rank;   // global rank (0 to tp_rows*tp_cols - 1)
    
    // Derived: local per-GPU dimensions
    // These are computed from global dims and TP rank:
    //   local_M = M / tp_rows
    //   local_N = N / tp_cols
    //   local_K = K / tp_cols
};
