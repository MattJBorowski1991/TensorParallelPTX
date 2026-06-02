#include <stdio.h>
#include <vector>
#include <string>
#include <time.h>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
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
    int M = DEFAULT_M, N = DEFAULT_N, K = DEFAULT_K;
    int num_batches = DEFAULT_NUM_BATCHES;
    int chunk_batches = 1;
    int tp_rows = DEFAULT_TP_ROWS, tp_cols = DEFAULT_TP_COLS; // Default 2D TP mesh: 2x2 (4 GPUs)
    bool verify = false;
    bool profile = true;
    int profile_runs = DEFAULT_PROFILE_RUNS;
    std::string walltime_file = "profile_walltime.txt";
    int rank = -1;
    int world_size = -1;
    int local_rank = -1;
    std::string nccl_id_prefix;
};

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--M" && i + 1 < argc) args.M = atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) args.N = atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) args.K = atoi(argv[++i]);
        else if (arg == "--B" && i + 1 < argc) args.num_batches = atoi(argv[++i]);
        else if (arg == "--chunk-batches" && i + 1 < argc) args.chunk_batches = atoi(argv[++i]);
        else if (arg == "--tp-rows" && i + 1 < argc) args.tp_rows = atoi(argv[++i]);
        else if (arg == "--tp-cols" && i + 1 < argc) args.tp_cols = atoi(argv[++i]);
        else if (arg == "--no-verify") args.verify = false;
        else if (arg == "--no-profile") args.profile = false;
        else if (arg == "--profile-runs" && i + 1 < argc) args.profile_runs = atoi(argv[++i]);
        else if (arg == "--walltime-file" && i + 1 < argc) args.walltime_file = argv[++i];
        else if (arg == "--rank" && i + 1 < argc) args.rank = atoi(argv[++i]);
        else if (arg == "--world-size" && i + 1 < argc) args.world_size = atoi(argv[++i]);
        else if (arg == "--local-rank" && i + 1 < argc) args.local_rank = atoi(argv[++i]);
        else if (arg == "--nccl-id-prefix" && i + 1 < argc) args.nccl_id_prefix = argv[++i];
    }
    return args;
}

struct ProfileStats {
    float avg_rank_ms = 0.f;
    float wall_ms = 0.f;
};

struct DistContext {
    bool enabled = false;
    int rank = 0;
    int world_size = 1;
    int local_rank = 0;
    std::string nccl_id_prefix;
};

static int env_to_int(const char* name, int fallback) {
    const char* v = getenv(name);
    return (v && v[0] != '\0') ? atoi(v) : fallback;
}

static std::string detect_default_nccl_prefix() {
    const char* run_id = getenv("TORCHELASTIC_RUN_ID");
    if (run_id && run_id[0] != '\0') {
        return std::string("/tmp/tpx_nccl_id_") + run_id;
    }
    return "/tmp/tpx_nccl_id";
}

static DistContext resolve_dist_context(const Args& args, int num_ranks, int num_gpus) {
    DistContext dc;
    dc.rank = (args.rank >= 0) ? args.rank : env_to_int("RANK", -1);
    dc.world_size = (args.world_size > 0) ? args.world_size : env_to_int("WORLD_SIZE", -1);
    dc.local_rank = (args.local_rank >= 0) ? args.local_rank : env_to_int("LOCAL_RANK", -1);
    dc.nccl_id_prefix = args.nccl_id_prefix.empty() ? detect_default_nccl_prefix() : args.nccl_id_prefix;

    if (dc.rank >= 0 || dc.world_size > 0 || dc.local_rank >= 0) {
        dc.enabled = true;
    }

    if (dc.enabled) {
        if (dc.rank < 0 || dc.world_size <= 0) {
            fprintf(stderr, "[error] distributed mode requires rank/world-size (args or env RANK/WORLD_SIZE).\n");
            exit(1);
        }
        if (dc.world_size != num_ranks) {
            fprintf(stderr, "[error] world_size (%d) must equal tp_rows*tp_cols (%d).\n", dc.world_size, num_ranks);
            exit(1);
        }
        if (dc.rank < 0 || dc.rank >= dc.world_size) {
            fprintf(stderr, "[error] rank (%d) must be in [0, %d).\n", dc.rank, dc.world_size);
            exit(1);
        }
        if (dc.local_rank < 0) dc.local_rank = dc.rank % num_gpus;
        if (dc.local_rank < 0 || dc.local_rank >= num_gpus) {
            fprintf(stderr, "[error] local_rank (%d) must be in [0, %d).\n", dc.local_rank, num_gpus);
            exit(1);
        }
        return dc;
    }

    // Default: single-process single-rank mode.
    dc.rank = 0;
    dc.world_size = 1;
    dc.local_rank = 0;
    if (num_ranks > 1) {
        fprintf(stderr,
                "[error] tp_rows*tp_cols=%d requires one process per GPU. Launch with torchrun/mpirun and set RANK/WORLD_SIZE/LOCAL_RANK.\n",
                num_ranks);
        exit(1);
    }
    return dc;
}

#ifdef USE_NCCL
static std::string make_id_path(const std::string& prefix, const std::string& tag) {
    return prefix + "_" + tag + ".bin";
}

static void write_nccl_id_file(const std::string& path, const ncclUniqueId& id) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[error] failed to write NCCL id file: %s\n", path.c_str());
        exit(1);
    }
    size_t nw = fwrite(&id, 1, sizeof(ncclUniqueId), f);
    fclose(f);
    if (nw != sizeof(ncclUniqueId)) {
        fprintf(stderr, "[error] short write on NCCL id file: %s\n", path.c_str());
        exit(1);
    }
}

static void read_nccl_id_file_retry(const std::string& path, ncclUniqueId* id_out) {
    constexpr int kMaxRetries = 10000;
    for (int i = 0; i < kMaxRetries; ++i) {
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            size_t nr = fread(id_out, 1, sizeof(ncclUniqueId), f);
            fclose(f);
            if (nr == sizeof(ncclUniqueId)) return;
        }
        usleep(10000);
    }
    fprintf(stderr, "[error] timed out waiting for NCCL id file: %s\n", path.c_str());
    exit(1);
}
#endif

static void append_walltime_log(const Args& args, const ProfileStats& stats) {
    FILE* f = fopen(args.walltime_file.c_str(), "a");
    if (!f) {
        fprintf(stderr, "[warn] could not open walltime log file: %s\n", args.walltime_file.c_str());
        return;
    }

    time_t now = time(nullptr);
    char ts[64] = {0};
    struct tm tmv;
    localtime_r(&now, &tmv);
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
static ProfileStats run_profile(const Args& args, const DistContext& dist) {
    const int tp_rows      = args.tp_rows;
    const int tp_cols      = args.tp_cols;
    const int num_ranks    = tp_rows * tp_cols;
    const int num_gpus     = query_num_gpus();
    const int M = args.M, N = args.N, K = args.K;
    const int B = args.num_batches;
    const int local_M = M / tp_rows;
    const int local_N = N / tp_cols;
    const int local_K = K / tp_cols;
    const int chunk_B = args.chunk_batches > 0 ? args.chunk_batches : 1;

    if (!validate_runtime(args, num_gpus)) return {};
    if (chunk_B > B) {
        fprintf(stderr, "[warn] chunk-batches (%d) > B (%d), clamping to B.\n", chunk_B, B);
    }
    const int eff_chunk_B = chunk_B > B ? B : chunk_B;
    const int num_chunks = (B + eff_chunk_B - 1) / eff_chunk_B;

    // Point 1: out-of-core mode.
    // Stage all local batch shards once, then pipeline compute/comm in batch chunks.
    size_t szA_l = (size_t)local_M * local_K;
    size_t szB_l = (size_t)local_K * local_N;
    size_t szC_l = (size_t)local_M * local_N;
    size_t szA_all = (size_t)B * szA_l;
    size_t szB_all = (size_t)B * szB_l;
    size_t szC_chunk_max = (size_t)eff_chunk_B * szC_l;

    const int rank = dist.rank;
    const int device = dist.local_rank;
    RankCoord coord = rank_to_coord(rank, tp_cols);

    CHECK_CUDA(cudaSetDevice(device));

#ifdef USE_NCCL
    ncclComm_t world_comm = nullptr;
    ncclComm_t row_comm = nullptr;

    // Global communicator for cross-rank metric aggregation.
    {
        ncclUniqueId id{};
        const std::string id_path = make_id_path(dist.nccl_id_prefix, "world");
        if (rank == 0) {
            CHECK_NCCL(ncclGetUniqueId(&id));
            write_nccl_id_file(id_path, id);
        }
        read_nccl_id_file_retry(id_path, &id);
        CHECK_NCCL(ncclCommInitRank(&world_comm, num_ranks, id, rank));
    }

    if (tp_cols > 1) {
        const int row = coord.row;
        const int col = coord.col;
        ncclUniqueId id{};
        const std::string row_tag = std::string("row") + std::to_string(row);
        const std::string id_path = make_id_path(dist.nccl_id_prefix, row_tag);
        if (col == 0) {
            CHECK_NCCL(ncclGetUniqueId(&id));
            write_nccl_id_file(id_path, id);
        }
        read_nccl_id_file_retry(id_path, &id);
        CHECK_NCCL(ncclCommInitRank(&row_comm, tp_cols, id, col));
    }
#endif

    // One host process manages exactly one GPU/rank.
    PinnedBuffer<half> h_A_stage(szA_all);
    PinnedBuffer<half> h_B_stage(szB_all);
    DeviceBuffer<half> d_A(szA_all);
    DeviceBuffer<half> d_B(szB_all);
    DeviceBuffer<float> d_C_ping(szC_chunk_max);
    DeviceBuffer<float> d_C_pong(szC_chunk_max);

    cudaStream_t compute_stream{}, comm_stream{};
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&comm_stream));

    cudaEvent_t ev_start{}, ev_stop{};
    cudaEvent_t ev_compute_done_ping{}, ev_compute_done_pong{};
    cudaEvent_t ev_comm_done_ping{}, ev_comm_done_pong{};
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventCreate(&ev_compute_done_ping));
    CHECK_CUDA(cudaEventCreate(&ev_compute_done_pong));
    CHECK_CUDA(cudaEventCreate(&ev_comm_done_ping));
    CHECK_CUDA(cudaEventCreate(&ev_comm_done_pong));

    for (int b = 0; b < B; ++b) {
        half* A_dst = h_A_stage.get() + (size_t)b * szA_l;
        half* B_dst = h_B_stage.get() + (size_t)b * szB_l;
        fill_rank_batch_shards(A_dst, B_dst,
                               M, N, K, local_M, local_N, local_K, b, coord);
    }

    CHECK_CUDA(cudaMemcpyAsync(d_A.get(), h_A_stage.get(),
                               szA_all * sizeof(half), cudaMemcpyHostToDevice, compute_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B.get(), h_B_stage.get(),
                               szB_all * sizeof(half), cudaMemcpyHostToDevice, compute_stream));
    CHECK_CUDA(cudaStreamSynchronize(compute_stream));

    CHECK_CUDA(cudaEventRecord(ev_start, compute_stream));

    for (int iter = 0; iter < args.profile_runs; ++iter) {
        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            const int global_chunk_idx = iter * num_chunks + chunk_idx;
            const int batch_start = chunk_idx * eff_chunk_B;
            const int cur_chunk_B = ((batch_start + eff_chunk_B) <= B) ? eff_chunk_B : (B - batch_start);
            const int buf = global_chunk_idx & 1;
            const size_t c_chunk_elems = (size_t)cur_chunk_B * szC_l;

            // Reuse guard must span iteration boundaries too.
            if (global_chunk_idx >= 2) {
                if (buf == 0) {
                    CHECK_CUDA(cudaStreamWaitEvent(compute_stream, ev_comm_done_ping, 0));
                } else {
                    CHECK_CUDA(cudaStreamWaitEvent(compute_stream, ev_comm_done_pong, 0));
                }
            }

            GemmConfig cfg{};
            cfg.M = M;
            cfg.N = N;
            cfg.K = K;
            cfg.num_batches = cur_chunk_B;
            cfg.warmups = 0;
            cfg.runs = 1;
            cfg.tp_rows = tp_rows;
            cfg.tp_cols = tp_cols;
            cfg.gpu_rank = rank;

            const half* A_ptr = d_A.get() + (size_t)batch_start * szA_l;
            const half* B_ptr = d_B.get() + (size_t)batch_start * szB_l;
            float* C_ptr = (buf == 0) ? d_C_ping.get() : d_C_pong.get();

            CHECK_CUDA(cudaMemsetAsync(C_ptr, 0, c_chunk_elems * sizeof(float), compute_stream));
            launch_kernel(A_ptr, B_ptr, C_ptr, cfg, compute_stream);

            if (buf == 0) {
                CHECK_CUDA(cudaEventRecord(ev_compute_done_ping, compute_stream));
                CHECK_CUDA(cudaStreamWaitEvent(comm_stream, ev_compute_done_ping, 0));
            } else {
                CHECK_CUDA(cudaEventRecord(ev_compute_done_pong, compute_stream));
                CHECK_CUDA(cudaStreamWaitEvent(comm_stream, ev_compute_done_pong, 0));
            }

#ifdef USE_NCCL
            if (tp_cols > 1) {
                CHECK_NCCL(ncclAllReduce((const void*)C_ptr, (void*)C_ptr,
                                         c_chunk_elems, ncclFloat, ncclSum,
                                         row_comm, comm_stream));
            }
#endif

            if (buf == 0) {
                CHECK_CUDA(cudaEventRecord(ev_comm_done_ping, comm_stream));
            } else {
                CHECK_CUDA(cudaEventRecord(ev_comm_done_pong, comm_stream));
            }
        }
    }

    CHECK_CUDA(cudaEventRecord(ev_stop, comm_stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float local_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&local_ms, ev_start, ev_stop));
    local_ms /= args.profile_runs;

    printf("  rank %d (GPU %d) [row=%d col=%d]  A[%d×%d] B[%d×%d] -> C[%d×%d]  %.3f ms avg over %d batches\n",
           rank, device, coord.row, coord.col,
           local_M, local_K, local_K, local_N, local_M, local_N,
           local_ms, B);
    fflush(stdout);

    ProfileStats stats{};
    stats.avg_rank_ms = local_ms;
    stats.wall_ms = local_ms;

#ifdef USE_NCCL
    if (dist.world_size > 1) {
        DeviceBuffer<float> d_send(1), d_sum(1), d_max(1);
        float host_sum = 0.f;
        float host_max = 0.f;
        CHECK_CUDA(cudaMemcpyAsync(d_send.get(), &local_ms, sizeof(float), cudaMemcpyHostToDevice, comm_stream));
        CHECK_NCCL(ncclAllReduce((const void*)d_send.get(), (void*)d_sum.get(), 1,
                                 ncclFloat, ncclSum, world_comm, comm_stream));
        CHECK_NCCL(ncclAllReduce((const void*)d_send.get(), (void*)d_max.get(), 1,
                                 ncclFloat, ncclMax, world_comm, comm_stream));
        CHECK_CUDA(cudaMemcpyAsync(&host_sum, d_sum.get(), sizeof(float), cudaMemcpyDeviceToHost, comm_stream));
        CHECK_CUDA(cudaMemcpyAsync(&host_max, d_max.get(), sizeof(float), cudaMemcpyDeviceToHost, comm_stream));
        CHECK_CUDA(cudaStreamSynchronize(comm_stream));
        stats.avg_rank_ms = host_sum / dist.world_size;
        stats.wall_ms = host_max;
    }
#endif

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaEventDestroy(ev_compute_done_ping));
    CHECK_CUDA(cudaEventDestroy(ev_compute_done_pong));
    CHECK_CUDA(cudaEventDestroy(ev_comm_done_ping));
    CHECK_CUDA(cudaEventDestroy(ev_comm_done_pong));
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaStreamDestroy(comm_stream));

#ifdef USE_NCCL
    if (row_comm) CHECK_NCCL(ncclCommDestroy(row_comm));
    if (world_comm) CHECK_NCCL(ncclCommDestroy(world_comm));
#endif

    return stats;
}

// ── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    const int num_ranks = args.tp_rows * args.tp_cols;
    const int num_gpus = query_num_gpus();
    DistContext dist = resolve_dist_context(args, num_ranks, num_gpus);

    if (dist.rank == 0) {
        #ifdef TP_KERNEL_VARIANT
        printf("\n=== TensorParallelPTX (%s) ===\n", TP_KERNEL_VARIANT);
        #else
        printf("\n=== TensorParallelPTX (unknown kernel variant) ===\n");
        #endif
        printf("Global problem:  M=%d  N=%d  K=%d  B=%d\n", args.M, args.N, args.K, args.num_batches);
        printf("TP mesh:         %d×%d  (%d GPUs)\n", args.tp_rows, args.tp_cols, args.tp_rows * args.tp_cols);
        printf("Per-GPU shard:   A[%d×%d]  B[%d×%d]  C[%d×%d]\n",
               args.M / args.tp_rows, args.K / args.tp_cols,
               args.K / args.tp_cols, args.N / args.tp_cols,
               args.M / args.tp_rows, args.N / args.tp_cols);
        printf("Chunking:        %d batch(es) per pipeline chunk\n", args.chunk_batches);
    }

    if (args.verify) {
        if (dist.rank == 0) printf("\n--- Verification (1×1 baseline) ---\n");
        run_verify();
    }

    if (args.profile) {
        if (dist.rank == 0) printf("\n--- Profiling (%d×%d mesh) ---\n", args.tp_rows, args.tp_cols);
        ProfileStats stats = run_profile(args, dist);
        if (stats.avg_rank_ms <= 0.0f || stats.wall_ms <= 0.0f) {
            fprintf(stderr, "[profile] skipped: runtime validation failed or no successful launches.\n");
            return 1;
        }

        double tflops_avg_rank = 2.0 * args.num_batches * (double)args.M * args.N * args.K
                                 / (stats.avg_rank_ms * 1e-3) / 1e12;
        double tflops_wall = 2.0 * args.num_batches * (double)args.M * args.N * args.K
                             / (stats.wall_ms * 1e-3) / 1e12;
         if (dist.rank == 0) {
             printf("[profile] avg across ranks: %.3f ms | %.2f TFLOPS (includes comms when USE_NCCL is enabled)\n",
                 stats.avg_rank_ms, tflops_avg_rank);
             printf("[profile] wall time (critical path across %d GPUs): %.3f ms | %.2f TFLOPS\n",
                 args.tp_rows * args.tp_cols, stats.wall_ms, tflops_wall);

             append_walltime_log(args, stats);
             printf("[profile] wall-time log appended to: %s\n", args.walltime_file.c_str());
         }
    }

    return 0;
}
