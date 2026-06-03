#include "src/app/logging.h"

#include <cstdio>
#include <ctime>

void print_run_header(const Args& args, int rank) {
    if (rank != 0) return;
#ifdef TP_KERNEL_VARIANT
    printf("\n=== TensorParallelPTX (%s) ===\n", TP_KERNEL_VARIANT);
#else
    printf("\n=== TensorParallelPTX (unknown kernel variant) ===\n");
#endif
    printf("Global problem:  M=%d  N=%d  K=%d  B=%d\n", args.M, args.N, args.K, args.num_batches);
    printf("TP mesh:         %dx%d  (%d GPUs)\n", args.tp_rows, args.tp_cols, args.tp_rows * args.tp_cols);
    printf("Per-GPU shard:   A[%dx%d]  B[%dx%d]  C[%dx%d]\n",
           args.M / args.tp_rows, args.K / args.tp_cols,
           args.K / args.tp_cols, args.N / args.tp_cols,
           args.M / args.tp_rows, args.N / args.tp_cols);
    printf("Chunking:        %d batch(es) per pipeline chunk\n", args.chunk_batches);
}

void print_verify_header(int rank) {
    if (rank == 0) printf("\n--- Verification (1x1 baseline) ---\n");
}

void print_profile_header(const Args& args, int rank) {
    if (rank == 0) printf("\n--- Profiling (%dx%d mesh) ---\n", args.tp_rows, args.tp_cols);
}

void print_profile_summary(const Args& args, const ProfileStats& stats, int rank) {
    if (rank != 0) return;
    double tflops_avg_rank = 2.0 * args.num_batches * (double)args.M * args.N * args.K
                             / (stats.avg_rank_ms * 1e-3) / 1e12;
    double tflops_wall = 2.0 * args.num_batches * (double)args.M * args.N * args.K
                         / (stats.wall_ms * 1e-3) / 1e12;
    printf("[profile] avg across ranks: %.3f ms | %.2f TFLOPS (includes comms when USE_NCCL is enabled)\n",
           stats.avg_rank_ms, tflops_avg_rank);
    printf("[profile] wall time (critical path across %d GPUs): %.3f ms | %.2f TFLOPS\n",
           args.tp_rows * args.tp_cols, stats.wall_ms, tflops_wall);
}

void append_walltime_log(const Args& args, const ProfileStats& stats) {
    FILE* f = fopen(args.walltime_file.c_str(), "a");
    if (!f) {
        fprintf(stderr, "[warn] could not open walltime log file: %s\n", args.walltime_file.c_str());
        return;
    }

    time_t now = time(nullptr);
    char ts[64] = {0};
    struct tm tmv;
#if defined(_WIN32)
    localtime_s(&tmv, &now);
#else
    localtime_r(&now, &tmv);
#endif
    strftime(ts, sizeof(ts), "%Y-%m-%d %H:%M:%S", &tmv);

#ifdef TP_KERNEL_VARIANT
    const char* kernel_variant = TP_KERNEL_VARIANT;
#else
    const char* kernel_variant = "unknown";
#endif

    fprintf(f,
            "%s kernel=%s M=%d N=%d K=%d B=%d tp=%dx%d runs=%d avg_rank_ms=%.3f wall_ms=%.3f\n",
            ts,
            kernel_variant,
            args.M,
            args.N,
            args.K,
            args.num_batches,
            args.tp_rows,
            args.tp_cols,
            args.profile_runs,
            stats.avg_rank_ms,
            stats.wall_ms);
    fclose(f);
}
