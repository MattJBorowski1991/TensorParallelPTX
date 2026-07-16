// 1D tensor parallelism (Megatron-style) — an isolated alternative to the
// SUMMA path (summa_runner.cu). Same kernels, same NCCL bootstrap, different
// distribution. Flat mesh: --tp-rows 1 --tp-cols P.
//
// TLDR:
// Every rank runs a normal local GEMM. The difference is how A/B/C are sharded:
//   - 1d-col: split output columns, then optionally allgather C.
//   - 1d-row: split the K dimension, then allreduce partial C.
// No input-panel broadcasts happen here. Unlike SUMMA, communication is on the
// output C buffer, not on A/B panels.
//
//   1d-col (column-parallel):  C[:, j] = A · B[:, j]
//     A replicated, B split by N columns. Each rank's GEMM produces its own
//     C column shard [M x N/P] — the math needs NO communication.
//     ncclAllGather then materializes the full C (real transformer stacks
//     skip this by feeding the shard straight into a row-parallel layer).
//
//   1d-row (row-parallel):     C = sum_p A[:, p-th K slice] · B[p-th K slice, :]
//     A split by K columns, B split by K rows. Each rank's GEMM produces a
//     full-size PARTIAL C [M x N]; ncclAllReduce sums the partials — the
//     collective IS the accumulation (contrast with SUMMA's accumulate kernel).
//
// 1d-row: Comm/compute overlap: C (output) is double-buffered. 
// Ping-pong allows for one partial result to be accumulated (comm_stream, chunk c) while
// the next partial result is being GEMMed (compute_stream, chunk c+1)
// Events hand each buffer back and forth: c_ready (compute→comm, GEMM wrote it) and
// c_free (comm→compute, collective is done with it).

#include "src/tp/oned_runner.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <nvtx3/nvToolsExt.h>

#include "include/config.h"
#include "include/cuda_utils.h"
#include "src/app/dist_context.h"
#include "src/comm/nccl_utils.h"
#include "src/tp/sharding.h"

void launch_kernel(const half* A, const half* B, float* C,
                   const GemmConfig& cfg, cudaStream_t stream);

static TpDataKind selected_data_kind() {
#ifdef TP_KERNEL_VARIANT
    if (strstr(TP_KERNEL_VARIANT, "int4") != nullptr) return TpDataKind::Int4;
    if (strstr(TP_KERNEL_VARIANT, "int8") != nullptr) return TpDataKind::Int8;
#endif
    return TpDataKind::Fp16;
}

static size_t input_storage_bytes(size_t logical_elements, TpDataKind kind) {
    if (kind == TpDataKind::Fp16) return logical_elements * sizeof(half);
    if (kind == TpDataKind::Int8) return logical_elements * sizeof(int8_t);
    return logical_elements / 2; // packed signed INT4 - SHOULDNT HERE BE logical_elements * sizeof(int8_t) / 2 ??????
}

static bool validate_runtime_1d(const Args& args, int num_gpus, TpDataKind kind) {
    if (args.M <= 0 || args.N <= 0 || args.K <= 0 || args.num_batches <= 0 || args.profile_runs <= 0) {
        fprintf(stderr, "[error] M/N/K, num_batches, and profile_runs must be > 0.\n");
        return false;
    }
    if (args.tp_rows != 1 || args.tp_cols <= 0) {
        fprintf(stderr, "[error] 1D TP uses a flat mesh: --tp-rows 1 --tp-cols <P>. Got %dx%d\n",
                args.tp_rows, args.tp_cols);
        return false;
    }
    const int P = args.tp_cols;
    if (args.tp_mode == TpMode::OneDCol && args.N % P != 0) {
        fprintf(stderr, "[error] 1d-col requires N %% P == 0 (N=%d P=%d).\n", args.N, P);
        return false;
    }
    if (args.tp_mode == TpMode::OneDRow && args.K % P != 0) {
        fprintf(stderr, "[error] 1d-row requires K %% P == 0 (K=%d P=%d).\n", args.K, P);
        return false;
    }
    if (kind == TpDataKind::Int4 &&
        (args.K % 2 != 0 || (args.tp_mode == TpMode::OneDRow && (args.K / P) % 2 != 0))) {
        fprintf(stderr, "[error] INT4 1D TP requires even global and local K dimensions.\n");
        return false;
    }
    if (num_gpus < P) {
        fprintf(stderr, "[error] one physical GPU per rank required: P=%d available_gpus=%d\n", P, num_gpus);
        return false;
    }
#ifndef USE_NCCL
    if (P > 1) {
        fprintf(stderr, "[error] 1D TP with P > 1 requires NCCL build support (USE_NCCL).\n");
        return false;
    }
#endif
    return true;
}

// run_profile_1d runs once per rank/process. With torchrun --nproc_per_node=4
// and --tp-cols 4, this function executes 4 times in parallel, once per GPU.
// Each rank sees the same global M/N/K, but picks a different local shard by
// using `rank` as its 1D partition index.
ProfileStats run_profile_1d(const Args& args, const DistContext& dist) {
    const int P = args.tp_cols;
    const bool col_mode = (args.tp_mode == TpMode::OneDCol);
    const int M = args.M, N = args.N, K = args.K;
    const int B = args.num_batches;
    const TpDataKind data_kind = selected_data_kind();
    const bool integer_output = data_kind != TpDataKind::Fp16;

    // Per-rank GEMM dims:
    //
    // 1d-col:
    //   A is replicated:    [M x K]
    //   B is column shard:  [K x N/P]
    //   C is column shard:  [M x N/P]
    //
    // 1d-row:
    //   A is K shard:       [M x K/P]
    //   B is K-row shard:   [K/P x N]
    //   C is full partial:  [M x N], then AllReduce sums all ranks' partials.
    const int local_M = M;
    const int local_N = col_mode ? N / P : N;
    const int local_K = col_mode ? K : K / P;

    if (!validate_runtime_1d(args, query_num_gpus(), data_kind)) return {}; //what is query_num_gpus????
    const int chunk_B = args.chunk_batches > 0 ? args.chunk_batches : 1; // WHAT IS THE PURPOSE OF THIS CHUNK_B HERE???
    if (chunk_B > B) {
        fprintf(stderr, "[warn] chunk-batches (%d) > B (%d), clamping to B.\n", chunk_B, B);
    }
    const int eff_chunk_B = chunk_B > B ? B : chunk_B;
    const int num_chunks = (B + eff_chunk_B - 1) / eff_chunk_B;

    const size_t szA_l = (size_t)local_M * local_K;
    const size_t szB_l = (size_t)local_K * local_N;
    const size_t szC_l = (size_t)local_M * local_N;
    const size_t bytesA_l = input_storage_bytes(szA_l, data_kind);
    const size_t bytesB_l = input_storage_bytes(szB_l, data_kind);
    const size_t szC_chunk_max = (size_t)eff_chunk_B * szC_l;

    const int rank = dist.rank;
    const int device = dist.local_rank;
    CHECK_CUDA(cudaSetDevice(device)); //WHAT DOES THIS DO?

    // -- 1. SETUP -------------------------------------------------------------
    // 1D TP only needs one NCCL communicator: the flat world of P ranks.
    // It is used for output collectives:
    //   - 1d-col: AllGather C shards if we want full C materialized.
    //   - 1d-row: AllReduce partial C values to form final C.
#ifdef USE_NCCL
    ncclComm_t world_comm = nullptr;
    if (P > 1) {
        ncclUniqueId id{};
        const std::string id_path = make_id_path(dist.nccl_id_prefix, "world1d");
        if (rank == 0) {
            CHECK_NCCL(ncclGetUniqueId(&id));
            write_nccl_id_file(id_path, id);
        }
        read_nccl_id_file_retry(id_path, &id);
        CHECK_NCCL(ncclCommInitRank(&world_comm, P, id, rank));
    }
#endif

    // Allocate my local A/B shards for every batch, plus two C output buffers.
    //
    // Important contrast with SUMMA:
    //   SUMMA ping/pong buffers hold incoming A/B input panels.
    //   1D ping/pong buffers hold outgoing C chunks.
    //
    // Why output ping/pong? While NCCL communicates C_ping from chunk c, the
    // compute stream can write the next chunk into C_pong.
    PinnedBuffer<uint8_t> h_A_stage((size_t)B * bytesA_l);
    PinnedBuffer<uint8_t> h_B_stage((size_t)B * bytesB_l);
    DeviceBuffer<uint8_t> d_A((size_t)B * bytesA_l);
    DeviceBuffer<uint8_t> d_B((size_t)B * bytesB_l);
    // Double-buffered GEMM output: collective on buffer `b` overlaps the next
    // chunk's GEMM into buffer `1-b`.
    DeviceBuffer<float> d_C_ping(szC_chunk_max);
    DeviceBuffer<float> d_C_pong(szC_chunk_max);
    // 1d-col only: allgather destination (rank-major: [P][chunk_B][lM][lN]).
    DeviceBuffer<float> d_C_gathered(col_mode ? szC_chunk_max * P : 1);

    cudaStream_t compute_stream{}, comm_stream{};
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&comm_stream));

    // Two streams plus a two-way event handshake:
    //   c_ready: compute -> comm, "GEMM finished writing C; NCCL may read."
    //   c_free:  comm -> compute, "NCCL finished with C; GEMM may overwrite."
    cudaEvent_t ev_start{}, ev_stop{};
    cudaEvent_t ev_c_ready[2] = {};  // compute→comm: GEMM finished writing C[b]
    cudaEvent_t ev_c_free[2]  = {};  // comm→compute: collective done with C[b]
    CHECK_CUDA(cudaEventCreate(&ev_start));
    CHECK_CUDA(cudaEventCreate(&ev_stop));
    for (int b = 0; b < 2; ++b) {
        CHECK_CUDA(cudaEventCreate(&ev_c_ready[b]));
        CHECK_CUDA(cudaEventCreate(&ev_c_free[b]));
    }

    // -- 2. DATA --------------------------------------------------------------
    // Generate only this rank's local inputs, then upload them once before the
    // timed loop. The global A/B matrices are never materialized.
    //
    // 1d-col example with P=4:
    //   rank0 gets B columns 0..N/4-1 and a full copy of A.
    //   rank1 gets B columns N/4..N/2-1 and a full copy of A.
    //
    // 1d-row example with P=4:
    //   rank0 gets K slice 0 of A/B.
    //   rank1 gets K slice 1 of A/B.
    //   each rank later produces a full-size partial C.
    for (int b = 0; b < B; ++b) {
        void* A_dst = h_A_stage.get() + (size_t)b * bytesA_l;
        void* B_dst = h_B_stage.get() + (size_t)b * bytesB_l;
        if (col_mode) {
            fill_rank_batch_shards_1d_col_typed(
                A_dst, B_dst, M, N, K, local_N, b, rank, data_kind);
        } else {
            fill_rank_batch_shards_1d_row_typed(
                A_dst, B_dst, M, N, K, local_K, b, rank, data_kind);
        }
    }

    CHECK_CUDA(cudaMemcpyAsync(d_A.get(), h_A_stage.get(),
                               (size_t)B * bytesA_l, cudaMemcpyHostToDevice, compute_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B.get(), h_B_stage.get(),
                               (size_t)B * bytesB_l, cudaMemcpyHostToDevice, compute_stream));
    CHECK_CUDA(cudaStreamSynchronize(compute_stream));

    // -- 3. run_chunk ---------------------------------------------------------
    // One chunk = GEMM into C[buf] + collective on C[buf]. `pass` alternates
    // buffers across the whole run (chunks AND profile iterations).
    //
    // Output ping/pong mental model:
    //   compute stream: [GEMM chunk0 -> C_ping] [GEMM chunk1 -> C_pong]
    //   comm stream:                            [NCCL on C_ping       ] [NCCL on C_pong]
    //
    // The handshake:
    //   c_ready: compute -> comm, "GEMM finished writing; NCCL may read."
    //   c_free:  comm -> compute, "NCCL finished; GEMM may overwrite."
    int pass = 0;
    auto run_chunk = [&](int chunk_idx) {
        char nvtx_name[64];
        snprintf(nvtx_name, sizeof(nvtx_name), "chunk %d", chunk_idx);
        nvtxRangePushA(nvtx_name);

        const int buf = pass & 1;
        ++pass;
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

        const uint8_t* A_own = d_A.get() + (size_t)batch_start * bytesA_l;
        const uint8_t* B_own = d_B.get() + (size_t)batch_start * bytesB_l;
        float* C_out = (buf == 0) ? d_C_ping.get() : d_C_pong.get();

        // Wait until the collective from two chunks ago released this buffer.
        // Without this, GEMM could overwrite C_ping while NCCL is still reading
        // C_ping from an earlier chunk.
        // (No memset: the kernel overwrites every element of its C region.)
        CHECK_CUDA(cudaStreamWaitEvent(compute_stream, ev_c_free[buf], 0));

        nvtxRangePushA("gemm");
        launch_kernel(reinterpret_cast<const half*>(A_own),
                      reinterpret_cast<const half*>(B_own),
                      C_out, cfg, compute_stream);
        nvtxRangePop();
        // Signal the comm stream: C_out now contains a complete GEMM result.
        CHECK_CUDA(cudaEventRecord(ev_c_ready[buf], compute_stream));

#ifdef USE_NCCL
        if (P > 1) {
            // NCCL must not read C_out until the GEMM above has finished.
            CHECK_CUDA(cudaStreamWaitEvent(comm_stream, ev_c_ready[buf], 0));
            if (col_mode) {
                nvtxRangePushA("allgather C");
                // 1d-col: every rank owns different C columns. AllGather packs
                // all ranks' C shards into d_C_gathered so full C exists.
                // The local math itself did not need this collective.
                CHECK_NCCL(ncclAllGather((const void*)C_out,
                                         (void*)d_C_gathered.get(),
                                         c_chunk_elems, integer_output ? ncclInt32 : ncclFloat,
                                         world_comm, comm_stream));
            } else {
                nvtxRangePushA("allreduce C");
                // In-place: sums the P partial C's; the collective IS the accumulation.
                // After this call, each rank's C_out contains the same final C
                // for this chunk.
                CHECK_NCCL(ncclAllReduce((const void*)C_out, (void*)C_out,
                                         c_chunk_elems, integer_output ? ncclInt32 : ncclFloat, ncclSum,
                                         world_comm, comm_stream));
            }
            nvtxRangePop();
            // Signal the compute stream: this C buffer is safe to reuse.
            CHECK_CUDA(cudaEventRecord(ev_c_free[buf], comm_stream));
        }
#endif
        nvtxRangePop(); // chunk
    };

    // -- 4. TIMED LOOP --------------------------------------------------------
    // profile_runs x chunks x run_chunk(), bracketed by cuda events -> ms.
    CHECK_CUDA(cudaEventRecord(ev_start, compute_stream));

    for (int iter = 0; iter < args.profile_runs; ++iter)
        for (int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx)
            run_chunk(chunk_idx);

    // Fold the trailing collectives into the timed region.
    CHECK_CUDA(cudaStreamWaitEvent(compute_stream, ev_c_free[0], 0));
    CHECK_CUDA(cudaStreamWaitEvent(compute_stream, ev_c_free[1], 0));
    CHECK_CUDA(cudaEventRecord(ev_stop, compute_stream));
    CHECK_CUDA(cudaEventSynchronize(ev_stop));

    float local_ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&local_ms, ev_start, ev_stop));
    local_ms /= args.profile_runs;

    printf("  rank %d (GPU %d) [%s P=%d]  A[%d×%d] B[%d×%d] -> C[%d×%d]  %.3f ms avg over %d batches\n",
           rank, device, tp_mode_name(args.tp_mode), P,
           local_M, local_K, local_K, local_N, local_M, local_N,
           local_ms, B);
    fflush(stdout);

    // ── TP correctness check (untimed, all kernel storage variants) ──────────
    // 1d-col checks the local C shard; 1d-row checks C after allreduce.
    // Re-runs chunk0 through the same GEMM + output collective path.
    // 1d-col samples global columns owned by this rank.
    // 1d-row samples C after AllReduce, so it should match full global C.
    if (args.verify_tp) {
        nvtxRangePushA("verify-tp");
        const int vbuf = pass & 1;   // buffer run_chunk(0) will use next
        run_chunk(0);
        CHECK_CUDA(cudaStreamSynchronize(comm_stream));
        CHECK_CUDA(cudaStreamSynchronize(compute_stream));

        float* C_check = (vbuf == 0) ? d_C_ping.get() : d_C_pong.get();
        constexpr int kSamples = 64;
        unsigned s = 0x9E3779B9u ^ (unsigned)rank;
        int fails = 0;
        float max_rel = 0.0f;
        int32_t max_abs_int = 0;

        if (integer_output) {
            std::vector<int32_t> h_C(szC_l);
            CHECK_CUDA(cudaMemcpy(h_C.data(), C_check,
                                  szC_l * sizeof(int32_t), cudaMemcpyDeviceToHost));
            for (int i = 0; i < kSamples; ++i) {
                s = s * 1664525u + 1013904223u;
                const int m = (int)((s >> 8) % (unsigned)local_M);
                s = s * 1664525u + 1013904223u;
                const int n = (int)((s >> 8) % (unsigned)local_N);
                const int gn = col_mode ? rank * local_N + n : n;
                const int32_t ref = cpu_ref_c_value_int(0, m, gn, K, N, data_kind);
                const int32_t out = h_C[(size_t)m * local_N + n];
                const int64_t diff = (int64_t)ref - out;
                const int32_t abs_err = (int32_t)(diff < 0 ? -diff : diff);
                if (abs_err > max_abs_int) max_abs_int = abs_err;
                if (abs_err != 0) ++fails;
            }
        } else {
            std::vector<float> h_C(szC_l);
            CHECK_CUDA(cudaMemcpy(h_C.data(), C_check,
                                  szC_l * sizeof(float), cudaMemcpyDeviceToHost));
            for (int i = 0; i < kSamples; ++i) {
                s = s * 1664525u + 1013904223u;
                const int m = (int)((s >> 8) % (unsigned)local_M);
                s = s * 1664525u + 1013904223u;
                const int n = (int)((s >> 8) % (unsigned)local_N);
                const int gn = col_mode ? rank * local_N + n : n;
                const float ref = cpu_ref_c_value(0, m, gn, K, N);
                const float out = h_C[(size_t)m * local_N + n];
                const float abs_err = fabsf(ref - out);
                const float rel = abs_err / fmaxf(fabsf(ref), 1e-6f);
                if (rel > max_rel) max_rel = rel;
                if (abs_err > 2e-3f * fabsf(ref) + 2e-3f) ++fails;
            }
        }

        if (integer_output) {
            printf("  rank %d [verify-tp] %s  (%d/%d samples ok, max_abs=%d)\n",
                   rank, fails == 0 ? "PASS" : "FAIL",
                   kSamples - fails, kSamples, (int)max_abs_int);
        } else {
            printf("  rank %d [verify-tp] %s  (%d/%d samples ok, max_rel=%.3e)\n",
                   rank, fails == 0 ? "PASS" : "FAIL",
                   kSamples - fails, kSamples, max_rel);
        }
        fflush(stdout);
        nvtxRangePop();
    }

    // -- 6. STATS -------------------------------------------------------------
    // Purpose: turns each rank's measured local_ms into reported timing numbers.
    // avg_rank_ms: average of local_ms across ranks, via AllReduce sum / world_size.
    // wall_ms: max local_ms across ranks, via AllReduce max.
    // wall_ms is the number that matters for end-to-end wall time: the slowest rank.
    // If single-rank: both are just local_ms.
    ProfileStats stats{};
    stats.avg_rank_ms = local_ms;
    stats.wall_ms = local_ms;

#ifdef USE_NCCL
    if (dist.world_size > 1) {
        DeviceBuffer<float> d_send(1), d_sum(1), d_max(1);
        float host_sum = 0.f, host_max = 0.f;
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
    for (int b = 0; b < 2; ++b) {
        CHECK_CUDA(cudaEventDestroy(ev_c_ready[b]));
        CHECK_CUDA(cudaEventDestroy(ev_c_free[b]));
    }
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaStreamDestroy(comm_stream));

#ifdef USE_NCCL
    if (world_comm) CHECK_NCCL(ncclCommDestroy(world_comm));
#endif

    return stats;
}
