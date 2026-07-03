#pragma once

#include <cuda_fp16.h>

#include "include/tp_utils.h"

void fill_rank_batch_shards(
    half* A_shard,
    half* B_shard,
    int M,
    int N,
    int K,
    int local_M,
    int local_N,
    int local_K,
    int batch,
    RankCoord coord);

// ── 1D (Megatron-style) shard generation ─────────────────────────────────────
// Column-parallel: A replicated (full M×K), B column shard (K × local_N)
// owned by rank p at N-offset p*local_N.
void fill_rank_batch_shards_1d_col(
    half* A_full, half* B_shard,
    int M, int N, int K, int local_N,
    int batch, int rank_p);

// Row-parallel: A K-shard (M × local_K) and B K-row shard (local_K × N),
// both owned by rank p at K-offset p*local_K.
void fill_rank_batch_shards_1d_row(
    half* A_shard, half* B_shard,
    int M, int N, int K, int local_K,
    int batch, int rank_p);

// CPU reference for one element of the global C (batch, global row/col):
// full-K dot product from the deterministic generator. Used by --verify-tp.
float cpu_ref_c_value(int batch, int global_m, int global_n, int K, int N);
