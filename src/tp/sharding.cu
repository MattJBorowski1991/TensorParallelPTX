#include "src/tp/sharding.h"

#include <cuda_fp16.h>

static inline half gen_fp16_val(int batch, int r, int c, int ld) {
    unsigned x = (unsigned)(batch * 1315423911u) ^ (unsigned)(r * 2654435761u)
               ^ (unsigned)(c * 40503u) ^ (unsigned)(ld * 2166136261u);
    x ^= x >> 13;
    x *= 1274126177u;
    x ^= x >> 16;
    float v = ((x & 0xFFFFu) / 65535.0f) * 0.1f - 0.05f;
    return __float2half(v);
}

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
    RankCoord coord) {
    const int A_row_offset = coord.row * local_M;
    const int A_col_offset = coord.col * local_K;
    // K not split (tp_cols==1, local_K==K): every rank needs the full B rows,
    // regardless of its row coord. Otherwise B K-rows are indexed by row coord.
    const int B_row_offset = (local_K == K) ? 0 : coord.row * local_K;
    const int B_col_offset = coord.col * local_N;

    for (int m = 0; m < local_M; ++m) {
        int gr = A_row_offset + m;
        for (int k = 0; k < local_K; ++k) {
            int gc = A_col_offset + k;
            A_shard[(size_t)m * local_K + k] = gen_fp16_val(batch, gr, gc, K);
        }
    }

    for (int k = 0; k < local_K; ++k) {
        int gr = B_row_offset + k;
        for (int n = 0; n < local_N; ++n) {
            int gc = B_col_offset + n;
            B_shard[(size_t)k * local_N + n] = gen_fp16_val(batch, gr, gc, N);
        }
    }
}

void fill_rank_batch_shards_1d_col(
    half* A_full, half* B_shard,
    int M, int N, int K, int local_N,
    int batch, int rank_p) {
    // A replicated: every rank generates the identical full matrix locally —
    // the deterministic generator makes replication free (no broadcast).
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            A_full[(size_t)m * K + k] = gen_fp16_val(batch, m, k, K);

    const int n_off = rank_p * local_N;
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < local_N; ++n)
            B_shard[(size_t)k * local_N + n] = gen_fp16_val(batch, k, n_off + n, N);
}

void fill_rank_batch_shards_1d_row(
    half* A_shard, half* B_shard,
    int M, int N, int K, int local_K,
    int batch, int rank_p) {
    const int k_off = rank_p * local_K;
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < local_K; ++k)
            A_shard[(size_t)m * local_K + k] = gen_fp16_val(batch, m, k_off + k, K);

    for (int k = 0; k < local_K; ++k)
        for (int n = 0; n < N; ++n)
            B_shard[(size_t)k * N + n] = gen_fp16_val(batch, k_off + k, n, N);
}

float cpu_ref_c_value(int batch, int global_m, int global_n, int K, int N) {
    // Full-K dot product using the same deterministic generator as the shard
    // fill — one global C element, no global matrices materialized.
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += __half2float(gen_fp16_val(batch, global_m, k, K)) *
               __half2float(gen_fp16_val(batch, k, global_n, N));
    }
    return acc;
}
