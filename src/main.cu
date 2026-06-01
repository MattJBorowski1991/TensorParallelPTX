#include <stdio.h>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "include/config.h"
#include "include/cuda_utils.h"
#include "include/tp_utils.h"
#include "src/solver.h"
#include "src/data.h"

#ifdef USE_NCCL
#include <nccl.h>
#define CHECK_NCCL(call) \
    do { \
        ncclResult_t r_ = (call); \
        if (r_ != ncclSuccess) { \
            fprintf(stderr, "NCCL Error at %s:%d: %s\n", __FILE__, __LINE__, ncclGetErrorString(r_)); \
            exit(1); \
        } \
    } while (0)
#endif

// Implemented in kernels/*.cu
void launch_kernel(const half* A, const half* B, float* C,
                   const GemmConfig& cfg, cudaStream_t stream);

// ── Command-line parsing ──────────────────────────────────────────────────────
struct Args {
    int M = 8192, N = 8192, K = 8192;
    int num_batches = 4;
    int tp_rows = 2, tp_cols = 2;          // Default 2D TP mesh: 2x2 (4 GPUs)
    bool verify = true;
    bool profile = true;
    int profile_runs = 5;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--M" && i + 1 < argc) args.M = atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) args.N = atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) args.K = atoi(argv[++i]);
        else if (arg == "--B" && i + 1 < argc) args.num_batches = atoi(argv[++i]);
        else if (arg == "--tp-rows" && i + 1 < argc) args.tp_rows = atoi(argv[++i]);
        else if (arg == "--tp-cols" && i + 1 < argc) args.tp_cols = atoi(argv[++i]);
        else if (arg == "--no-verify") args.verify = false;
        else if (arg == "--no-profile") args.profile = false;
        else if (arg == "--profile-runs" && i + 1 < argc) args.profile_runs = atoi(argv[++i]);
    }
    return args;
}

// ── GPU availability check ────────────────────────────────────────────────────
static int query_num_gpus() {
    int n = 0;
    CHECK_CUDA(cudaGetDeviceCount(&n));
    return n;
}

static bool validate_runtime(const Args& args, int num_gpus) {
    if (args.M <= 0 || args.N <= 0 || args.K <= 0 || args.num_batches <= 0 || args.profile_runs <= 0) {
        fprintf(stderr, "[error] M/N/K, num_batches, and profile_runs must be > 0.\n");
        return false;
    }
    if (args.tp_rows <= 0 || args.tp_cols <= 0) {
        fprintf(stderr, "[error] tp_rows/tp_cols must be > 0.\n");
        return false;
    }
    if (args.M % args.tp_rows != 0 || args.N % args.tp_cols != 0 || args.K % args.tp_cols != 0) {
        fprintf(stderr,
                "[error] divisibility check failed: require M%%tp_rows==0, N%%tp_cols==0, K%%tp_cols==0. "
                "Got M=%d N=%d K=%d tp_rows=%d tp_cols=%d\n",
                args.M, args.N, args.K, args.tp_rows, args.tp_cols);
        return false;
    }

    const int num_ranks = args.tp_rows * args.tp_cols;
    if (num_gpus < num_ranks) {
        fprintf(stderr,
                "[error] this path requires one physical GPU per rank. requested ranks=%d available_gpus=%d\n",
                num_ranks, num_gpus);
        return false;
    }

#ifndef USE_NCCL
    if (args.tp_cols > 1) {
        fprintf(stderr, "[error] tp_cols > 1 requires NCCL build support (USE_NCCL).\n");
        return false;
    }
#endif

    return true;
}

// ── Out-of-core shard generation (no global host A/B materialization) ───────
// Deterministic value generator from global indices so shards are reproducible
// without allocating full matrices in host RAM.
static inline half gen_fp16_val(int batch, int r, int c, int ld) {
    unsigned x = (unsigned)(batch * 1315423911u) ^ (unsigned)(r * 2654435761u)
               ^ (unsigned)(c * 40503u) ^ (unsigned)(ld * 2166136261u);
    x ^= x >> 13;
    x *= 1274126177u;
    x ^= x >> 16;
    float v = ((x & 0xFFFFu) / 65535.0f) * 0.1f - 0.05f;
    return __float2half(v);
}

static void fill_rank_batch_shards(
    half* A_shard,
    half* B_shard,
    int M, int N, int K,
    int local_M, int local_N, int local_K,
    int batch,
    RankCoord coord)
{
    const int A_row_offset = coord.row * local_M;
    const int A_col_offset = coord.col * local_K;
    const int B_row_offset = coord.col * local_K;
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

// ── Verification: single GPU (1x1 mesh), confirms kernel correctness ──────────
static void run_verify() {
    constexpr int Mv = 512, Nv = 512, Kv = 512;

    std::vector<half>  h_A(Mv * Kv), h_B(Kv * Nv);
    std::vector<float> h_C_ref(Mv * Nv, 0.f), h_C_out(Mv * Nv, 0.f);

    generate_fp16(h_A.data(), h_B.data(), Mv, Nv, Kv, 1);
    cpu_gemm_fp16(h_A.data(), h_B.data(), h_C_ref.data(), Mv, Nv, Kv);

    DeviceBuffer<half>  d_A(Mv * Kv);
    DeviceBuffer<half>  d_B(Kv * Nv);
    DeviceBuffer<float> d_C(Mv * Nv);
    CHECK_CUDA(cudaMemcpy(d_A.get(), h_A.data(), Mv * Kv * sizeof(half),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B.get(), h_B.data(), Kv * Nv * sizeof(half),  cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C.get(), 0, Mv * Nv * sizeof(float)));

    GemmConfig vcfg{};
    vcfg.M = Mv;
    vcfg.N = Nv;
    vcfg.K = Kv;
    vcfg.num_batches = 1;
    vcfg.warmups = 0;
    vcfg.runs = 1;
    vcfg.tp_rows = 1;
    vcfg.tp_cols = 1;
    vcfg.gpu_rank = 0;
    Solver solver;
    solver.configure(vcfg);
    solver.run(d_A.get(), d_B.get(), d_C.get());

    CHECK_CUDA(cudaMemcpy(h_C_out.data(), d_C.get(), Mv * Nv * sizeof(float), cudaMemcpyDeviceToHost));
    AccuracyResult acc = measure_accuracy(h_C_ref.data(), h_C_out.data(), Mv, Nv);
    printf("[verify]  M=%d N=%d K=%d  %s  max_abs=%.4e  rmse=%.4e  rel=%.3f%%\n",
           Mv, Nv, Kv, acc.pass ? "PASS" : "FAIL",
           acc.max_abs_err, acc.rmse, acc.real_err_pct);
}

// ── Multi-GPU profiling: one rank per GPU, 2D mesh tile mapping ───────────────
//
// For each gpu_rank in [0, tp_rows*tp_cols):
//   - Decompose rank → (rank_row, rank_col)
//   - Pack the non-contiguous A shard [local_M x local_K] and
//     B shard [local_K x local_N] into contiguous host buffers
//   - Upload shards to the corresponding GPU (cudaSetDevice(gpu_rank))
//   - Launch GEMM kernel → produces C shard [local_M x local_N]
//   - Download C shard and unpack it back into the global output matrix
//
// This is correct for any (tp_rows x tp_cols) mesh: 2x2, 4x4, 8x8, etc.
// For meshes where K is split (local_K < K), each GPU computes a partial C;
// a future AllReduce across the column group is needed to get exact results.
static float run_profile(const Args& args) {
    const int tp_rows      = args.tp_rows;
    const int tp_cols      = args.tp_cols;
    const int num_ranks    = tp_rows * tp_cols;
    const int num_gpus     = query_num_gpus();
    const int M = args.M, N = args.N, K = args.K;
    const int B = args.num_batches;
    const int local_M = M / tp_rows;
    const int local_N = N / tp_cols;
    const int local_K = K / tp_cols;

    if (!validate_runtime(args, num_gpus)) return 0.f;

    // Point 1: out-of-core mode.
    // Keep only one batch-chunk shard in host memory at a time.
    size_t szA_l = (size_t)local_M * local_K;
    size_t szB_l = (size_t)local_K * local_N;
    size_t szC_l = (size_t)local_M * local_N;

    // ── Timing across all ranks ──────────────────────────────────────────────
    float total_ms = 0.f;
    int total_launches = 0;

    for (int row = 0; row < tp_rows; ++row) {
        std::vector<int> devices(tp_cols);
        std::vector<int> ranks(tp_cols);
        for (int col = 0; col < tp_cols; ++col) {
            int rank = row * tp_cols + col;
            ranks[col] = rank;
            devices[col] = rank; // validated: one device per rank
        }

#ifdef USE_NCCL
        ncclComm_t row_comm;
        CHECK_NCCL(ncclCommInitAll(&row_comm, tp_cols, devices.data()));
#endif

        std::vector<PinnedBuffer<half>> h_A_stage;
        std::vector<PinnedBuffer<half>> h_B_stage;
        std::vector<PinnedBuffer<float>> h_C_stage;
        std::vector<DeviceBuffer<half>> d_A;
        std::vector<DeviceBuffer<half>> d_B;
        std::vector<DeviceBuffer<float>> d_C;
        std::vector<cudaStream_t> streams(tp_cols);

        h_A_stage.reserve(tp_cols);
        h_B_stage.reserve(tp_cols);
        h_C_stage.reserve(tp_cols);
        d_A.reserve(tp_cols);
        d_B.reserve(tp_cols);
        d_C.reserve(tp_cols);

        for (int col = 0; col < tp_cols; ++col) {
            h_A_stage.emplace_back(szA_l);
            h_B_stage.emplace_back(szB_l);
            h_C_stage.emplace_back(szC_l);
            d_A.emplace_back(szA_l);
            d_B.emplace_back(szB_l);
            d_C.emplace_back(szC_l);
            CHECK_CUDA(cudaSetDevice(devices[col]));
            CHECK_CUDA(cudaStreamCreate(&streams[col]));
        }

        std::vector<float> rank_ms_accum(tp_cols, 0.f);

        for (int b = 0; b < B; ++b) {
            std::vector<cudaEvent_t> ev_start(tp_cols), ev_stop(tp_cols);

            // H2D + kernel launch for each rank in this row-group
            for (int col = 0; col < tp_cols; ++col) {
                const int rank = ranks[col];
                RankCoord coord = rank_to_coord(rank, tp_cols);
                CHECK_CUDA(cudaSetDevice(devices[col]));

                CHECK_CUDA(cudaEventCreate(&ev_start[col]));
                CHECK_CUDA(cudaEventCreate(&ev_stop[col]));

                fill_rank_batch_shards(h_A_stage[col].get(), h_B_stage[col].get(),
                                       M, N, K, local_M, local_N, local_K, b, coord);

                CHECK_CUDA(cudaMemcpyAsync(d_A[col].get(), h_A_stage[col].get(),
                                           szA_l * sizeof(half), cudaMemcpyHostToDevice, streams[col]));
                CHECK_CUDA(cudaMemcpyAsync(d_B[col].get(), h_B_stage[col].get(),
                                           szB_l * sizeof(half), cudaMemcpyHostToDevice, streams[col]));
                CHECK_CUDA(cudaMemsetAsync(d_C[col].get(), 0, szC_l * sizeof(float), streams[col]));

                GemmConfig cfg{};
                cfg.M = M;
                cfg.N = N;
                cfg.K = K;
                cfg.num_batches = 1;
                cfg.warmups = 0;
                cfg.runs = args.profile_runs;
                cfg.tp_rows = tp_rows;
                cfg.tp_cols = tp_cols;
                cfg.gpu_rank = rank;

                CHECK_CUDA(cudaEventRecord(ev_start[col], streams[col]));
                for (int i = 0; i < args.profile_runs; ++i) {
                    launch_kernel(d_A[col].get(), d_B[col].get(), d_C[col].get(), cfg, streams[col]);
                }
            }

#ifdef USE_NCCL
            // Point 2: K-split correctness via row-group allreduce across tp_cols.
            CHECK_NCCL(ncclGroupStart());
            for (int col = 0; col < tp_cols; ++col) {
                CHECK_NCCL(ncclAllReduce((const void*)d_C[col].get(), (void*)d_C[col].get(),
                                         szC_l, ncclFloat, ncclSum, row_comm, streams[col]));
            }
            CHECK_NCCL(ncclGroupEnd());
#endif

            for (int col = 0; col < tp_cols; ++col) {
                CHECK_CUDA(cudaSetDevice(devices[col]));
                CHECK_CUDA(cudaEventRecord(ev_stop[col], streams[col]));
                CHECK_CUDA(cudaEventSynchronize(ev_stop[col]));

                float ms = 0.f;
                CHECK_CUDA(cudaEventElapsedTime(&ms, ev_start[col], ev_stop[col]));
                ms /= args.profile_runs;
                rank_ms_accum[col] += ms;
                total_ms += ms;
                total_launches += 1;

                CHECK_CUDA(cudaMemcpyAsync(h_C_stage[col].get(), d_C[col].get(),
                                           szC_l * sizeof(float), cudaMemcpyDeviceToHost, streams[col]));
            }

            for (int col = 0; col < tp_cols; ++col) {
                CHECK_CUDA(cudaSetDevice(devices[col]));
                CHECK_CUDA(cudaStreamSynchronize(streams[col]));
                CHECK_CUDA(cudaEventDestroy(ev_start[col]));
                CHECK_CUDA(cudaEventDestroy(ev_stop[col]));
            }
        }

        for (int col = 0; col < tp_cols; ++col) {
            int rank = ranks[col];
            RankCoord coord = rank_to_coord(rank, tp_cols);
            printf("  rank %d (GPU %d) [row=%d col=%d]  A[%d×%d] B[%d×%d] -> C[%d×%d]  %.3f ms avg over %d batches\n",
                   rank, devices[col], coord.row, coord.col,
                   local_M, local_K, local_K, local_N, local_M, local_N,
                   rank_ms_accum[col] / B, B);
        }

        for (int col = 0; col < tp_cols; ++col) {
            CHECK_CUDA(cudaSetDevice(devices[col]));
            CHECK_CUDA(cudaStreamDestroy(streams[col]));
        }

#ifdef USE_NCCL
        CHECK_NCCL(ncclCommDestroy(row_comm));
#endif
    }

    return total_launches ? (total_ms / total_launches) : 0.f;
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    printf("\n=== TensorParallelPTX Baseline (fp16_wmma) ===\n");
    printf("Global problem:  M=%d  N=%d  K=%d  B=%d\n", args.M, args.N, args.K, args.num_batches);
    printf("TP mesh:         %d×%d  (%d GPUs)\n", args.tp_rows, args.tp_cols, args.tp_rows * args.tp_cols);
    printf("Per-GPU shard:   A[%d×%d]  B[%d×%d]  C[%d×%d]\n",
           args.M / args.tp_rows, args.K / args.tp_cols,
           args.K / args.tp_cols, args.N / args.tp_cols,
           args.M / args.tp_rows, args.N / args.tp_cols);

    if (args.verify) {
        printf("\n--- Verification (1×1 baseline) ---\n");
        run_verify();
    }

    if (args.profile) {
        printf("\n--- Profiling (%d×%d mesh) ---\n", args.tp_rows, args.tp_cols);
        float avg_ms = run_profile(args);

        double tflops = 2.0 * args.num_batches * (double)args.M * args.N * args.K
                        / (avg_ms * 1e-3) / 1e12;
        printf("[profile] avg across ranks: %.3f ms | %.2f TFLOPS (compute only, no comms)\n",
               avg_ms, tflops);
    }

    return 0;
}
