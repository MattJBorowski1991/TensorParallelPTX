#pragma once

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

// ── Shard geometry ─────────────────────────────────────────────────────────────
// For a global problem (M, N, K) on a (tp_rows x tp_cols) mesh:
//
//   A  [M  x K ] → GPU(i,j) owns rows [i*lM, (i+1)*lM), K-cols [j*lK, (j+1)*lK)
//   B  [K  x N ] → GPU(i,j) owns K-rows [i*lK, (i+1)*lK), N-cols [j*lN, (j+1)*lN)
//   C  [M  x N ] → GPU(i,j) owns rows [i*lM, (i+1)*lM), N-cols [j*lN, (j+1)*lN)
//
// where lM = M/tp_rows, lN = N/tp_cols, lK = K/tp_cols.
//
// Shards are generated directly into contiguous per-rank buffers by
// fill_rank_batch_shards (src/tp/sharding.cu) — the global matrices are
// never materialized.
