#include "src/tp/summa_runner.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
// NVTX v3 (header-only): named ranges for the nsys timeline. nsys projects
// ranges onto the GPU rows via the kernels/collectives launched inside them,
// which makes the compute/comm overlap directly visible.
#include <nvtx3/nvToolsExt.h>

#include "include/config.h"
#include "include/cuda_utils.h"
#include "include/tp_utils.h"
#include "src/app/dist_context.h"
#include "src/comm/nccl_utils.h"
#include "src/tp/sharding.h"

void launch_kernel(const half* A, const half* B, float* C,
                   const GemmConfig& cfg, cudaStream_t stream);

__global__ void accumulate_inplace(float* dst, const float* src, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

bool validate_runtime(const Args& args, int num_gpus) {
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
    // SUMMA panel indexing (B K-rows indexed by row coord with lK=K/tp_cols,
    // panel roots p over tp_cols) is only correct on a square mesh.
    if (args.tp_cols > 1 && args.tp_rows != args.tp_cols) {
        fprintf(stderr, "[error] SUMMA path requires a square mesh (tp_rows == tp_cols) when tp_cols > 1. "
                        "Got %dx%d\n", args.tp_rows, args.tp_cols);
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

ProfileStats run_profile(const Args& args, const DistContext& dist) {
    const int tp_rows = args.tp_rows;
    const int tp_cols = args.tp_cols;
    const int num_ranks = tp_rows * tp_cols;
    const int num_gpus = query_num_gpus();
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
    ncclComm_t col_comm = nullptr;

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

        ncclUniqueId row_id{};
        const std::string row_tag = std::string("row") + std::to_string(row);
        const std::string row_path = make_id_path(dist.nccl_id_prefix, row_tag);
        if (col == 0) {
            CHECK_NCCL(ncclGetUniqueId(&row_id));
            write_nccl_id_file(row_path, row_id);
        }
        read_nccl_id_file_retry(row_path, &row_id);
        CHECK_NCCL(ncclCommInitRank(&row_comm, tp_cols, row_id, col));

        ncclUniqueId col_id{};
        const std::string col_tag = std::string("col") + std::to_string(col);
        const std::string col_path = make_id_path(dist.nccl_id_prefix, col_tag);
        if (row == 0) {
            CHECK_NCCL(ncclGetUniqueId(&col_id));
            write_nccl_id_file(col_path, col_id);
        }
        read_nccl_id_file_retry(col_path, &col_id);
        CHECK_NCCL(ncclCommInitRank(&col_comm, tp_rows, col_id, row));
    }
#endif

    PinnedBuffer<half> h_A_stage(szA_all);
    PinnedBuffer<half> h_B_stage(szB_all);
    DeviceBuffer<half> d_A(szA_all);
    DeviceBuffer<half> d_B(szB_all);
    DeviceBuffer<half> d_A_panel_ping((size_t)eff_chunk_B * szA_l);
    DeviceBuffer<half> d_A_panel_pong((size_t)eff_chunk_B * szA_l);
    DeviceBuffer<half> d_B_panel_ping((size_t)eff_chunk_B * szB_l);
    DeviceBuffer<half> d_B_panel_pong((size_t)eff_chunk_B * szB_l);
    DeviceBuffer<float> d_C_accum(szC_chunk_max);
    DeviceBuffer<float> d_C_partial_ping(szC_chunk_max);
    DeviceBuffer<float> d_C_partial_pong(szC_chunk_max);

    cudaStream_t compute_stream{};
    cudaStream_t comm_stream{};
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&comm_stream));

    cudaEvent_t ev_start{}, ev_stop{};
    cudaEvent_t ev_panel_ready_ping{}, ev_panel_ready_pong{};
    // Reverse handshake: comm must not overwrite a panel buffer before the
    // compute stream has finished reading it (next chunk / next profile run).
    cudaEvent_t ev_panel_consumed_ping{}, ev_panel_consumed_pong{};
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    CHECK_CUDA(cudaEventCreate(&ev_panel_ready_ping));
    CHECK_CUDA(cudaEventCreate(&ev_panel_ready_pong));
    CHECK_CUDA(cudaEventCreate(&ev_panel_consumed_ping));
    CHECK_CUDA(cudaEventCreate(&ev_panel_consumed_pong));

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

    // One chunk = panel-broadcast loop + GEMM + accumulate. Shared by the
    // timed profiling loop and the post-timing verify pass.
    auto run_chunk = [&](int chunk_idx) {
            char nvtx_name[64];
            snprintf(nvtx_name, sizeof(nvtx_name), "chunk %d", chunk_idx);
            nvtxRangePushA(nvtx_name);

            const int batch_start = chunk_idx * eff_chunk_B;
            const int cur_chunk_B = ((batch_start + eff_chunk_B) <= B) ? eff_chunk_B : (B - batch_start);
            const size_t c_chunk_elems = (size_t)cur_chunk_B * szC_l;

            GemmConfig cfg{};
            cfg.M = M;
            cfg.N = N;
            cfg.K = K;
            cfg.num_batches = cur_chunk_B;
            cfg.local_M = local_M;
            cfg.local_N = local_N;
            cfg.local_K = local_K;

            const half* A_own = d_A.get() + (size_t)batch_start * szA_l;
            const half* B_own = d_B.get() + (size_t)batch_start * szB_l;
            CHECK_CUDA(cudaMemsetAsync(d_C_accum.get(), 0, c_chunk_elems * sizeof(float), compute_stream));

            if (tp_cols > 1) {
#ifdef USE_NCCL
                nvtxRangePushA("bcast p=0 (ping)");
                // Don't overwrite ping until compute finished reading it
                // (previous chunk / previous profile run).
                CHECK_CUDA(cudaStreamWaitEvent(comm_stream, ev_panel_consumed_ping, 0));
                CHECK_NCCL(ncclBroadcast((const void*)A_own,
                                         (void*)d_A_panel_ping.get(),
                                         (size_t)cur_chunk_B * szA_l,
                                         ncclHalf,
                                         0,
                                         row_comm,
                                         comm_stream));
                CHECK_NCCL(ncclBroadcast((const void*)B_own,
                                         (void*)d_B_panel_ping.get(),
                                         (size_t)cur_chunk_B * szB_l,
                                         ncclHalf,
                                         0,
                                         col_comm,
                                         comm_stream));
                CHECK_CUDA(cudaEventRecord(ev_panel_ready_ping, comm_stream));
                nvtxRangePop();
#endif
            }

            for (int p = 0; p < tp_cols; ++p) {
                const int buf = p & 1;
                const half* A_panel = A_own;
                const half* B_panel = B_own;
                float* C_partial = (buf == 0) ? d_C_partial_ping.get() : d_C_partial_pong.get();

#ifdef USE_NCCL
                if (tp_cols > 1) {
                    CHECK_CUDA(cudaStreamWaitEvent(compute_stream,
                                                   (buf == 0) ? ev_panel_ready_ping : ev_panel_ready_pong,
                                                   0));

                    A_panel = (buf == 0) ? d_A_panel_ping.get() : d_A_panel_pong.get();
                    B_panel = (buf == 0) ? d_B_panel_ping.get() : d_B_panel_pong.get();

                    const int next_p = p + 1;
                    if (next_p < tp_cols) {
                        snprintf(nvtx_name, sizeof(nvtx_name), "bcast p=%d (prefetch)", next_p);
                        nvtxRangePushA(nvtx_name);
                        const int next_buf = next_p & 1;
                        half* A_next = (next_buf == 0) ? d_A_panel_ping.get() : d_A_panel_pong.get();
                        half* B_next = (next_buf == 0) ? d_B_panel_ping.get() : d_B_panel_pong.get();
                        CHECK_CUDA(cudaStreamWaitEvent(comm_stream,
                                                       (next_buf == 0) ? ev_panel_consumed_ping : ev_panel_consumed_pong,
                                                       0));
                        CHECK_NCCL(ncclBroadcast((const void*)A_own,
                                                 (void*)A_next,
                                                 (size_t)cur_chunk_B * szA_l,
                                                 ncclHalf,
                                                 next_p,
                                                 row_comm,
                                                 comm_stream));
                        CHECK_NCCL(ncclBroadcast((const void*)B_own,
                                                 (void*)B_next,
                                                 (size_t)cur_chunk_B * szB_l,
                                                 ncclHalf,
                                                 next_p,
                                                 col_comm,
                                                 comm_stream));
                        CHECK_CUDA(cudaEventRecord((next_buf == 0) ? ev_panel_ready_ping : ev_panel_ready_pong,
                                                   comm_stream));
                        nvtxRangePop();
                    }
                }
#endif

                snprintf(nvtx_name, sizeof(nvtx_name), "gemm p=%d", p);
                nvtxRangePushA(nvtx_name);
                CHECK_CUDA(cudaMemsetAsync(C_partial, 0, c_chunk_elems * sizeof(float), compute_stream));
                launch_kernel(A_panel, B_panel, C_partial, cfg, compute_stream);
                nvtxRangePop();
#ifdef USE_NCCL
                if (tp_cols > 1) {
                    // Panels are last read by launch_kernel: mark buffer reusable.
                    CHECK_CUDA(cudaEventRecord((buf == 0) ? ev_panel_consumed_ping : ev_panel_consumed_pong,
                                               compute_stream));
                }
#endif

                snprintf(nvtx_name, sizeof(nvtx_name), "accum p=%d", p);
                nvtxRangePushA(nvtx_name);
                constexpr int kAccThreads = 256;
                const int acc_blocks = (int)((c_chunk_elems + kAccThreads - 1) / kAccThreads);
                accumulate_inplace<<<acc_blocks, kAccThreads, 0, compute_stream>>>(
                    d_C_accum.get(), C_partial, c_chunk_elems);
                CHECK_CUDA(cudaGetLastError());
                nvtxRangePop();
            }

            nvtxRangePop(); // chunk
    };

    CHECK_CUDA(cudaEventRecord(ev_start, compute_stream));

    for (int iter = 0; iter < args.profile_runs; ++iter)
        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
            run_chunk(chunk_idx);

    CHECK_CUDA(cudaEventRecord(ev_stop, compute_stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float local_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&local_ms, ev_start, ev_stop));
    local_ms /= args.profile_runs;

    printf("  rank %d (GPU %d) [row=%d col=%d]  A[%d×%d] B[%d×%d] -> C[%d×%d]  %.3f ms avg over %d batches\n",
           rank, device, coord.row, coord.col,
           local_M, local_K, local_K, local_N, local_M, local_N,
           local_ms, B);
    fflush(stdout);

    // ── TP correctness check (untimed) ────────────────────────────────────────
    // Re-run chunk 0 through the full SUMMA path, then sample-check C elements
    // of batch 0 against CPU dot products over the *global* K (deterministic
    // generator — no global matrices needed). Catches broadcast-root, panel
    // ordering, and accumulation bugs end-to-end.
    // fp16-only: the CPU reference assumes half inputs / float output; int
    // variants reinterpret these buffers and need their own reference.
#ifdef TP_KERNEL_VARIANT
    const bool fp16_variant = strstr(TP_KERNEL_VARIANT, "fp16") != nullptr;
#else
    const bool fp16_variant = true;
#endif
    if (args.verify_tp && !fp16_variant && rank == 0) {
        printf("  [verify-tp] skipped: only implemented for fp16 variants\n");
    }
    if (args.verify_tp && fp16_variant) {
        nvtxRangePushA("verify-tp");
        run_chunk(0);
        std::vector<float> h_C(szC_l);
        CHECK_CUDA(cudaMemcpyAsync(h_C.data(), d_C_accum.get(), szC_l * sizeof(float),
                                   cudaMemcpyDeviceToHost, compute_stream));
        CHECK_CUDA(cudaStreamSynchronize(compute_stream));

        constexpr int kSamples = 64;
        unsigned s = 0x9E3779B9u ^ (unsigned)rank;   // per-rank LCG seed
        int fails = 0;
        float max_rel = 0.0f;
        for (int i = 0; i < kSamples; ++i) {
            s = s * 1664525u + 1013904223u;
            const int m = (int)((s >> 8) % (unsigned)local_M);
            s = s * 1664525u + 1013904223u;
            const int n = (int)((s >> 8) % (unsigned)local_N);
            const float ref = cpu_ref_c_value(/*batch=*/0,
                                              coord.row * local_M + m,
                                              coord.col * local_N + n, K, N);
            const float out = h_C[(size_t)m * local_N + n];
            const float abs_err = fabsf(ref - out);
            const float rel = abs_err / fmaxf(fabsf(ref), 1e-6f);
            if (rel > max_rel) max_rel = rel;
            if (abs_err > 2e-3f * fabsf(ref) + 2e-3f) ++fails;
        }
        printf("  rank %d [verify-tp] %s  (%d/%d samples ok, max_rel=%.3e)\n",
               rank, fails == 0 ? "PASS" : "FAIL", kSamples - fails, kSamples, max_rel);
        fflush(stdout);
        nvtxRangePop();
    }

    ProfileStats stats{};
    stats.avg_rank_ms = local_ms;
    stats.wall_ms = local_ms;

#ifdef USE_NCCL
    if (dist.world_size > 1) {
        DeviceBuffer<float> d_send(1), d_sum(1), d_max(1);
        float host_sum = 0.f;
        float host_max = 0.f;
        CHECK_CUDA(cudaMemcpyAsync(d_send.get(), &local_ms, sizeof(float), cudaMemcpyHostToDevice, compute_stream));
        CHECK_NCCL(ncclAllReduce((const void*)d_send.get(), (void*)d_sum.get(), 1,
                                 ncclFloat, ncclSum, world_comm, compute_stream));
        CHECK_NCCL(ncclAllReduce((const void*)d_send.get(), (void*)d_max.get(), 1,
                                 ncclFloat, ncclMax, world_comm, compute_stream));
        CHECK_CUDA(cudaMemcpyAsync(&host_sum, d_sum.get(), sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
        CHECK_CUDA(cudaMemcpyAsync(&host_max, d_max.get(), sizeof(float), cudaMemcpyDeviceToHost, compute_stream));
        CHECK_CUDA(cudaStreamSynchronize(compute_stream));
        stats.avg_rank_ms = host_sum / dist.world_size;
        stats.wall_ms = host_max;
    }
#endif

    CHECK_CUDA(cudaEventDestroy(ev_start));
    CHECK_CUDA(cudaEventDestroy(ev_stop));
    CHECK_CUDA(cudaEventDestroy(ev_panel_ready_ping));
    CHECK_CUDA(cudaEventDestroy(ev_panel_ready_pong));
    CHECK_CUDA(cudaEventDestroy(ev_panel_consumed_ping));
    CHECK_CUDA(cudaEventDestroy(ev_panel_consumed_pong));
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaStreamDestroy(comm_stream));

#ifdef USE_NCCL
    if (col_comm) CHECK_NCCL(ncclCommDestroy(col_comm));
    if (row_comm) CHECK_NCCL(ncclCommDestroy(row_comm));
    if (world_comm) CHECK_NCCL(ncclCommDestroy(world_comm));
#endif

    return stats;
}
