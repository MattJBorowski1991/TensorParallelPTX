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

// ── Default profiling settings ───────────────────────────────────────────────
constexpr int DEFAULT_M = 16384;
constexpr int DEFAULT_N = 16384;
constexpr int DEFAULT_K = 16384;
constexpr int DEFAULT_TP_ROWS = 2;
constexpr int DEFAULT_TP_COLS = 2;
constexpr int DEFAULT_NUM_BATCHES = 4;
constexpr int DEFAULT_PROFILE_RUNS = 5;

// ── 2D Tensor Parallelism configuration ───────────────────────────────────────
// For 2D TP: GPU mesh is (TP_ROW x TP_COL).
// Each GPU (rank) holds tensors partitioned as:
//   - A:  (M / TP_ROW) x (K / TP_COL)
//   - B:  (K / TP_COL) x (N / TP_COL)  
//   - C:  (M / TP_ROW) x (N / TP_COL)
// This requires SUMMA-style K-panel exchange per step:
//   1. Broadcast A panel across each row communicator
//   2. Broadcast B panel across each column communicator
//   3. Accumulate partial C over all K panels

// ── Runtime problem / benchmark configuration ────────────────────────────────────
struct GemmConfig {
    // Global problem dimensions (informational)
    int M, N, K;
    int num_batches;

    // Per-rank GEMM dims — the kernel contract. The runner decides how these
    // relate to the global dims (SUMMA 2D shards, 1D col/row shards, or 1:1
    // for single-GPU verify). Kernels are TP-agnostic.
    int local_M, local_N, local_K;
};
