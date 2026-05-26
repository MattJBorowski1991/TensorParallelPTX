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

    GemmConfig vcfg{ .M=Mv, .N=Nv, .K=Kv, .num_batches=1, .warmups=0, .runs=1,
                     .tp_rows=1, .tp_cols=1, .gpu_rank=0 };
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

    if (num_gpus < num_ranks) {
        printf("[warn] requested %d GPUs but only %d available — "
               "mapping multiple ranks to GPU 0 (simulation mode)\n",
               num_ranks, num_gpus);
    }

    // ── Allocate & fill global host matrices ─────────────────────────────────
    size_t szA_g = (size_t)B * M * K;
    size_t szB_g = (size_t)B * K * N;
    size_t szC_g = (size_t)B * M * N;

    std::vector<half>  h_A(szA_g), h_B(szB_g);
    std::vector<float> h_C_global(szC_g, 0.f);

    generate_fp16(h_A.data(), h_B.data(), M, N, K, B);

    // ── Per-rank shard buffers (host side, contiguous) ───────────────────────
    size_t szA_l = (size_t)B * local_M * local_K;
    size_t szB_l = (size_t)B * local_K * local_N;
    size_t szC_l = (size_t)B * local_M * local_N;

    std::vector<half>  h_A_shard(szA_l);
    std::vector<half>  h_B_shard(szB_l);
    std::vector<float> h_C_shard(szC_l);

    // ── Timing across all ranks ──────────────────────────────────────────────
    float total_ms = 0.f;

    for (int rank = 0; rank < num_ranks; ++rank) {
        RankCoord coord = rank_to_coord(rank, tp_cols);
        int device = rank % num_gpus;         // graceful fallback for sim mode
        CHECK_CUDA(cudaSetDevice(device));

        // Pack non-contiguous A and B submatrices into contiguous shards
        pack_shard_A(h_A.data(), h_A_shard.data(), M, K, B, local_M, local_K, coord);
        pack_shard_B(h_B.data(), h_B_shard.data(), K, N, B, local_K, local_N, coord);

        DeviceBuffer<half>  d_A(szA_l);
        DeviceBuffer<half>  d_B(szB_l);
        DeviceBuffer<float> d_C(szC_l);
        CHECK_CUDA(cudaMemcpy(d_A.get(), h_A_shard.data(), szA_l * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B.get(), h_B_shard.data(), szB_l * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C.get(), 0, szC_l * sizeof(float)));

        GemmConfig cfg{
            .M = M, .N = N, .K = K,
            .num_batches = B,
            .warmups = 1, .runs = args.profile_runs,
            .tp_rows = tp_rows, .tp_cols = tp_cols,
            .gpu_rank = rank
        };

        Solver solver;
        solver.configure(cfg);
        float ms = solver.run(d_A.get(), d_B.get(), d_C.get());
        total_ms += ms;

        // Collect C shard and place it back into the global output
        CHECK_CUDA(cudaMemcpy(h_C_shard.data(), d_C.get(), szC_l * sizeof(float), cudaMemcpyDeviceToHost));
        unpack_shard_C(h_C_shard.data(), h_C_global.data(), M, N, B, local_M, local_N, coord);

        printf("  rank %d (GPU %d) [row=%d col=%d]  A[%d×%d] B[%d×%d] → C[%d×%d]  %.3f ms\n",
               rank, device, coord.row, coord.col,
               local_M, local_K, local_K, local_N, local_M, local_N, ms);
    }

    return total_ms / num_ranks;   // average across ranks
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
