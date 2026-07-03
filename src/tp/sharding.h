#pragma once

#include <cstdint>

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

// Compile-time-selected TP kernel storage. Integer kernels
// consume B transposed as N×K; INT4 stores two signed values per byte.
enum class TpDataKind { Fp16, Int8, Int4 };

void fill_rank_batch_shards_typed(
    void* A_shard, void* B_shard,
    int M, int N, int K,
    int local_M, int local_N, int local_K,
    int batch, RankCoord coord, TpDataKind kind);

void fill_rank_batch_shards_1d_col_typed(
    void* A_full, void* B_shard,
    int M, int N, int K, int local_N,
    int batch, int rank_p, TpDataKind kind);

void fill_rank_batch_shards_1d_row_typed(
    void* A_shard, void* B_shard,
    int M, int N, int K, int local_K,
    int batch, int rank_p, TpDataKind kind);

// Exact full-K reference for the integer generators used by typed 1D shards.
int32_t cpu_ref_c_value_int(
    int batch, int global_m, int global_n, int K, int N, TpDataKind kind);
