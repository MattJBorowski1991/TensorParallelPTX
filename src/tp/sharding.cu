// ── Purpose of this file ──────────────────────────────────────────────────────
// - Generate each rank's full shard of the (never-materialized) global A/B
//   matrices, in the right dtype and layout for the chosen kernel and TP mode.
// - Values are a deterministic hash of GLOBAL (batch, row, col) coordinates,
//   so every rank agrees on any element without storing or communicating it.
// - Recompute single reference C elements on the CPU (fp16 approximate,
//   int exact) so --verify-tp can spot-check the distributed GPU result.
// ──────────────────────────────────────────────────────────────────────────────

#include "src/tp/sharding.h"
#include <cuda_fp16.h>

// Deterministic hash of (batch, row, col) → small fp16 value in [-0.05, 0.05]; every rank can regenerate any global A/B element without storing the matrix.
static inline half gen_fp16_val(int batch, int r, int c, int ld) {
    unsigned x = (unsigned)(batch * 1315423911u) ^ (unsigned)(r * 2654435761u)
               ^ (unsigned)(c * 40503u) ^ (unsigned)(ld * 2166136261u);
    x ^= x >> 13;
    x *= 1274126177u;
    x ^= x >> 16;
    float v = ((x & 0xFFFFu) / 65535.0f) * 0.1f - 0.05f;
    return __float2half(v);
}

// SUMMA (2D mesh), fp16: fills this rank's A [lM×lK] and B [lK×lN] shards for one batch, based on its (row, col) mesh coordinate.
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

// 1d-col, fp16: fills the full replicated A [M×K] and this rank's B column shard [K×lN].
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

// 1d-row, fp16: fills this rank's K-slice of A [M×lK] and matching K-rows of B [lK×N].
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

// CPU reference for ONE element of the global fp16 C (full-K dot product); what --verify-tp compares sampled GPU results against.
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

// Same deterministic hash as gen_fp16_val, returned raw for the integer generators below.
static inline unsigned gen_hash(int batch, int r, int c, int ld) {
    unsigned x = (unsigned)(batch * 1315423911u) ^ (unsigned)(r * 2654435761u)
               ^ (unsigned)(c * 40503u) ^ (unsigned)(ld * 2166136261u);
    x ^= x >> 13;
    x *= 1274126177u;
    x ^= x >> 16;
    return x;
}

// Hash → small signed integer: [-8,7] for int4, [-7,7] for int8.
static inline int8_t gen_integer_val(
    int batch, int r, int c, int ld, TpDataKind kind) {
    const unsigned x = gen_hash(batch, r, c, ld);
    if (kind == TpDataKind::Int4) return static_cast<int8_t>((x & 0xFu) - 8);
    // Keep the learning workload comfortably inside int32 accumulation range.
    return static_cast<int8_t>((x % 15u) - 7);
}

// Packs two int4 values into one byte (lo nibble first) — the storage format the int4 kernels expect.
static inline int8_t pack_int4(int8_t lo, int8_t hi) {
    const uint8_t packed = (static_cast<uint8_t>(lo) & 0xFu)
                         | ((static_cast<uint8_t>(hi) & 0xFu) << 4);
    return static_cast<int8_t>(packed);
}

// 1d-col, any data kind: dispatches to fp16 fill or generates int8 / packed-int4 shards (B transposed to N×K, as the int kernels expect).
void fill_rank_batch_shards_1d_col_typed(
    void* A_full, void* B_shard,
    int M, int N, int K, int local_N,
    int batch, int rank_p, TpDataKind kind) {
    if (kind == TpDataKind::Fp16) {
        fill_rank_batch_shards_1d_col(
            static_cast<half*>(A_full), static_cast<half*>(B_shard),
            M, N, K, local_N, batch, rank_p);
        return;
    }

    int8_t* A = static_cast<int8_t*>(A_full);
    int8_t* BT = static_cast<int8_t*>(B_shard);
    const int n_off = rank_p * local_N;

    if (kind == TpDataKind::Int8) {
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                A[(size_t)m * K + k] = gen_integer_val(batch, m, k, K, kind);

        // Integer kernels consume the local B columns transposed: local_N×K.
        for (int n = 0; n < local_N; ++n)
            for (int k = 0; k < K; ++k)
                BT[(size_t)n * K + k] = gen_integer_val(batch, k, n_off + n, N, kind);
        return;
    }

    const int k_bytes = K / 2;
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; k += 2) {
            const int8_t v0 = gen_integer_val(batch, m, k, K, kind);
            const int8_t v1 = gen_integer_val(batch, m, k + 1, K, kind);
            A[(size_t)m * k_bytes + k / 2] = pack_int4(v0, v1);
        }
    }
    for (int n = 0; n < local_N; ++n) {
        for (int k = 0; k < K; k += 2) {
            const int8_t v0 = gen_integer_val(batch, k, n_off + n, N, kind);
            const int8_t v1 = gen_integer_val(batch, k + 1, n_off + n, N, kind);
            BT[(size_t)n * k_bytes + k / 2] = pack_int4(v0, v1);
        }
    }
}

// 1d-row, any data kind: same dispatch as above for the K-sliced A/B shards.
void fill_rank_batch_shards_1d_row_typed(
    void* A_shard, void* B_shard,
    int M, int N, int K, int local_K,
    int batch, int rank_p, TpDataKind kind) {
    if (kind == TpDataKind::Fp16) {
        fill_rank_batch_shards_1d_row(
            static_cast<half*>(A_shard), static_cast<half*>(B_shard),
            M, N, K, local_K, batch, rank_p);
        return;
    }

    int8_t* A = static_cast<int8_t*>(A_shard);
    int8_t* BT = static_cast<int8_t*>(B_shard);
    const int k_off = rank_p * local_K;

    if (kind == TpDataKind::Int8) {
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < local_K; ++k)
                A[(size_t)m * local_K + k] = gen_integer_val(batch, m, k_off + k, K, kind);

        // Transposed local K slice: N×local_K.
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < local_K; ++k)
                BT[(size_t)n * local_K + k] = gen_integer_val(batch, k_off + k, n, N, kind);
        return;
    }

    const int k_bytes = local_K / 2;
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < local_K; k += 2) {
            const int8_t v0 = gen_integer_val(batch, m, k_off + k, K, kind);
            const int8_t v1 = gen_integer_val(batch, m, k_off + k + 1, K, kind);
            A[(size_t)m * k_bytes + k / 2] = pack_int4(v0, v1);
        }
    }
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < local_K; k += 2) {
            const int8_t v0 = gen_integer_val(batch, k_off + k, n, N, kind);
            const int8_t v1 = gen_integer_val(batch, k_off + k + 1, n, N, kind);
            BT[(size_t)n * k_bytes + k / 2] = pack_int4(v0, v1);
        }
    }
}

// CPU reference for ONE element of the global int32 C — exact (integer math), used by --verify-tp for int8/int4 variants.
int32_t cpu_ref_c_value_int(
    int batch, int global_m, int global_n, int K, int N, TpDataKind kind) {
    int32_t acc = 0;
    for (int k = 0; k < K; ++k) {
        acc += static_cast<int32_t>(gen_integer_val(batch, global_m, k, K, kind))
             * static_cast<int32_t>(gen_integer_val(batch, k, global_n, N, kind));
    }
    return acc;
}


// SUMMA (2D mesh), any data kind: dispatches to fp16 fill or generates int8 / packed-int4 shards from the rank's mesh coordinate.
void fill_rank_batch_shards_typed(
    void* A_shard, void* B_shard,
    int M, int N, int K,
    int local_M, int local_N, int local_K,
    int batch, RankCoord coord, TpDataKind kind) {
    if (kind == TpDataKind::Fp16) {
        fill_rank_batch_shards(
            static_cast<half*>(A_shard), static_cast<half*>(B_shard),
            M, N, K, local_M, local_N, local_K, batch, coord);
        return;
    }

    int8_t* A = static_cast<int8_t*>(A_shard);
    int8_t* BT = static_cast<int8_t*>(B_shard);
    const int m_off = coord.row * local_M;
    const int a_k_off = coord.col * local_K;
    const int b_k_off = (local_K == K) ? 0 : coord.row * local_K;
    const int n_off = coord.col * local_N;

    if (kind == TpDataKind::Int8) {
        for (int m = 0; m < local_M; ++m)
            for (int k = 0; k < local_K; ++k)
                A[(size_t)m * local_K + k] =
                    gen_integer_val(batch, m_off + m, a_k_off + k, K, kind);

        // Integer kernels consume B transposed: local_N×local_K.
        for (int n = 0; n < local_N; ++n)
            for (int k = 0; k < local_K; ++k)
                BT[(size_t)n * local_K + k] =
                    gen_integer_val(batch, b_k_off + k, n_off + n, N, kind);
        return;
    }

    const int k_bytes = local_K / 2;
    for (int m = 0; m < local_M; ++m) {
        for (int k = 0; k < local_K; k += 2) {
            const int8_t v0 = gen_integer_val(batch, m_off + m, a_k_off + k, K, kind);
            const int8_t v1 = gen_integer_val(batch, m_off + m, a_k_off + k + 1, K, kind);
            A[(size_t)m * k_bytes + k / 2] = pack_int4(v0, v1);
        }
    }
    for (int n = 0; n < local_N; ++n) {
        for (int k = 0; k < local_K; k += 2) {
            const int8_t v0 = gen_integer_val(batch, b_k_off + k, n_off + n, N, kind);
            const int8_t v1 = gen_integer_val(batch, b_k_off + k + 1, n_off + n, N, kind);
            BT[(size_t)n * k_bytes + k / 2] = pack_int4(v0, v1);
        }
    }
}
