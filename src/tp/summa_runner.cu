// heart of the SUMMA path
// TLDR:
// Take rank (i,j) on a 2×2 mesh and turns it into a timed, verified distributed GEMM.
// Each rank repeatedly receives the panels it doesn't own, multiplies, and accumulates 
// — with communication and compute on separate streams so they overlap.
//
// NTLDR:
// 1. SETUP at the NCCL block — 3 communicators, file-based bootstrap, plus what each comm is for (row→A panels, col→B panels, world→timing stats)
// Buffer allocation — what each buffer's role is (own shards / panel landing zones / C_partial vs C_accum)
// Streams + the two-direction event handshake ("ready" vs "consumed")
// 2. DATA at the shard fill + one-time upload
// 3. run_chunk — THE ALGORITHM above the lambda, with the per-step comm/compute choreography; two inline comments inside mark the pipeline kick-off broadcast and the ping/pong panel loop
// 4. TIMED LOOP, 5. VERIFY-TP, 6. STATS (with why max = critical path) at their blocks

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
    return logical_elements / 2; // packed signed INT4
}

__global__ void accumulate_inplace(float* dst, const float* src, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

__global__ void accumulate_inplace_int32(int32_t* dst, const int32_t* src, size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] += src[idx];
    }
}

// Sanity: divisibility, square mesh, enough GPUs, NCCL present

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
    const TpDataKind kind = selected_data_kind();
    const int local_K = args.K / args.tp_cols;
    if (kind == TpDataKind::Int4 && (args.K % 2 != 0 || local_K % 2 != 0)) {
        fprintf(stderr, "[error] INT4 SUMMA requires even global and local K dimensions.\n");
        return false;
    }

    const int num_ranks = args.tp_rows * args.tp_cols;
    if (num_gpus < num_ranks) {
        fprintf(stderr,
                "[error] this path requires one physical GPU per rank. requested ranks=%d available_gpus=%d\n",
                num_ranks, num_gpus);
        return false;
    }
// NCCL (NVIDIA Collective Communications Library) is NVIDIA's library 
// for fast GPU-to-GPU communication — broadcast, allreduce, allgather, etc.
// that automatically routes data over the best available path (NVLink, PCIe, network)
#ifndef USE_NCCL
    if (args.tp_cols > 1) {
        fprintf(stderr, "[error] tp_cols > 1 requires NCCL build support (USE_NCCL).\n");
        return false;
    }
#endif

    return true;
}

// run_profile (and in fact the whole binary) runs once per rank
// torchrun starts 4 identical processes/ranks; each runs main() → run_profile() 
// top to bottom with only dist.rank differing. So every line you read in this 
// file is executing (tp_rows x tp_cols = ) 4× in parallel, once per GPU 
ProfileStats run_profile(const Args& args, const DistContext& dist) {
    const int tp_rows = args.tp_rows;
    const int tp_cols = args.tp_cols;
    const int num_ranks = tp_rows * tp_cols;
    const int num_gpus = query_num_gpus();
    const int M = args.M, N = args.N, K = args.K;
    const int B = args.num_batches;
    const TpDataKind data_kind = selected_data_kind();
    const bool integer_output = data_kind != TpDataKind::Fp16;
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
    size_t bytesA_l = input_storage_bytes(szA_l, data_kind);
    size_t bytesB_l = input_storage_bytes(szB_l, data_kind);
    size_t bytesA_all = (size_t)B * bytesA_l;
    size_t bytesB_all = (size_t)B * bytesB_l;
    size_t szC_chunk_max = (size_t)eff_chunk_B * szC_l;

    const int rank = dist.rank;
    const int device = dist.local_rank;
    RankCoord coord = rank_to_coord(rank, tp_cols);

    CHECK_CUDA(cudaSetDevice(device));

    // ── 1. SETUP ──────────────────────────────────────────────────────────────

    // NCCL: create 3 communicators* — world, my-row, my-column.
    // *communicator = named group of ranks, over which NCCL calls run (eg ncclAllReduce)

    // Bootstrap: to form a communicator, every rank must present the same ncclUniqueId 
    // (a ~128-byte token that identifies the group being formed)
    // Rank0 of each group writes the ncclUniqueId to a file, the others poll-read it 
    // (thats all the nccl_utils.cu does)
    // (ranks cant yet send it to each other over NCCL as NCCL isnt yet setup)
    // (actual frameworks use MPI for this)
    // Row comm  → broadcasts A panels across my mesh row.
    // Col comm  → broadcasts B panels down my mesh column.
    // World comm → only used at the end to average/max the timings.
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

    // Allocate: my own A/B shards (all batches), ping/pong panel receive
    // buffers (the "other ranks' panels" landing zone), and C buffers
    // (C_partial = one panel step's output, C_accum = running sum).
    PinnedBuffer<uint8_t> h_A_stage(bytesA_all);
    PinnedBuffer<uint8_t> h_B_stage(bytesB_all);
    DeviceBuffer<uint8_t> d_A(bytesA_all);
    DeviceBuffer<uint8_t> d_B(bytesB_all);
    // PINGPONG (double buffering):
    // overlapping comm with compute
    // comm stream:    [bcast p0 panels → PING][bcast p1 panels (A01,B10) → PONG]
    // compute stream:                         [ GEMM reads PING: A00·B00       ][ GEMM reads PONG: A01·B10 ]
    //                                          ▲ while this GEMM runs, p1's data is already arriving
    // where p0/p1 = whose turn it is to share
    // e.g. p=0 means: in every row, the member sitting at column 0 shares its A; in every column, the member sitting at row 0 shares its B.
    // to be more precise:
    // i.  p=0: sharers are: A = rank00, rank10; B = rank00, rank01
    // ii. p=1: sharers are: A = rank01, rank11; B = rank10, rank11
    // limited advantage here as we only have 4 GPUs and hence there will be 1 single overlap
    // but with a larger number of GPUs this would be v advantageous

    //without pingpong we would have: bcast -> GEMM -> wait for bcast -> GEMM (serial)
    DeviceBuffer<uint8_t> d_A_panel_ping((size_t)eff_chunk_B * bytesA_l);
    DeviceBuffer<uint8_t> d_A_panel_pong((size_t)eff_chunk_B * bytesA_l);
    DeviceBuffer<uint8_t> d_B_panel_ping((size_t)eff_chunk_B * bytesB_l);
    DeviceBuffer<uint8_t> d_B_panel_pong((size_t)eff_chunk_B * bytesB_l);
    DeviceBuffer<float> d_C_accum(szC_chunk_max);
    DeviceBuffer<float> d_C_partial_ping(szC_chunk_max);
    DeviceBuffer<float> d_C_partial_pong(szC_chunk_max);

    // Two streams so communication and compute can overlap, plus handshake
    // events in both directions: "panel ready" (comm→compute: safe to read)
    // and "panel consumed" (compute→comm: safe to overwrite).
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

    // ── 2. DATA ───────────────────────────────────────────────────────────────
    // Purpose: get all input data resident on the GPU before the timed loop, 
    // so the benchmark measures GEMM + NCCL only — never host→device transfers 
    // or data generation.

    // Fill my A/B shards (sharding.cu, deterministic hash) into pinned host
    // memory, then upload to the GPU once — inputs never move again.
    for (int b = 0; b < B; ++b) {
        void* A_dst = h_A_stage.get() + (size_t)b * bytesA_l;
        void* B_dst = h_B_stage.get() + (size_t)b * bytesB_l;
        fill_rank_batch_shards_typed(
            A_dst, B_dst, M, N, K, local_M, local_N, local_K, b, coord, data_kind);
    }

    CHECK_CUDA(cudaMemcpyAsync(d_A.get(), h_A_stage.get(),
                               bytesA_all, cudaMemcpyHostToDevice, compute_stream));
    CHECK_CUDA(cudaMemcpyAsync(d_B.get(), h_B_stage.get(),
                               bytesB_all, cudaMemcpyHostToDevice, compute_stream));
    CHECK_CUDA(cudaStreamSynchronize(compute_stream));

    // ── 3. run_chunk — THE ALGORITHM ──────────────────────────────────────────
    // One chunk = panel-broadcast loop + GEMM + accumulate. Shared by the
    // timed profiling loop (4) and the post-timing verify pass (5).
    // For each panel step p = 0..tp_cols-1:
    //   comm stream:    broadcast A panel (row comm, root col=p) and
    //                   B panel (col comm, root row=p) — prefetched into the
    //                   ping/pong buffer for step p+1 while step p computes
    //   compute stream: wait "panel ready" → GEMM → C_partial
    //                   → accumulate C_partial into C_accum
    auto run_chunk = [&](int chunk_idx) {
            char nvtx_name[64];
            snprintf(nvtx_name, sizeof(nvtx_name), "chunk %d", chunk_idx);
            nvtxRangePushA(nvtx_name);

            // Which batches this chunk covers (last chunk may be smaller).
            const int batch_start = chunk_idx * eff_chunk_B;
            const int cur_chunk_B = ((batch_start + eff_chunk_B) <= B) ? eff_chunk_B : (B - batch_start);
            const size_t c_chunk_elems = (size_t)cur_chunk_B * szC_l;
            const size_t a_bcast_count = (size_t)cur_chunk_B *
                (data_kind == TpDataKind::Fp16 ? szA_l : bytesA_l);
            const size_t b_bcast_count = (size_t)cur_chunk_B *
                (data_kind == TpDataKind::Fp16 ? szB_l : bytesB_l);
#ifdef USE_NCCL
            const ncclDataType_t input_nccl_type =
                data_kind == TpDataKind::Fp16 ? ncclHalf :
                data_kind == TpDataKind::Int8 ? ncclInt8 : ncclUint8;
#endif

            // Kernel launch config: global dims + this rank's local GEMM dims.
            GemmConfig cfg{};
            cfg.M = M;
            cfg.N = N;
            cfg.K = K;
            cfg.num_batches = cur_chunk_B;
            cfg.local_M = local_M;
            cfg.local_N = local_N;
            cfg.local_K = local_K;

            // My own shard slice for this chunk's batches (what I send when I'm root).
            const uint8_t* A_own = d_A.get() + (size_t)batch_start * bytesA_l;
            const uint8_t* B_own = d_B.get() + (size_t)batch_start * bytesB_l;
            // Zero the running sum: C_accum will collect all panel steps' partials.
            CHECK_CUDA(cudaMemsetAsync(d_C_accum.get(), 0, c_chunk_elems * sizeof(float), compute_stream));

            // Kick off the pipeline: broadcast step 0's panels into ping.
            if (tp_cols > 1) {
#ifdef USE_NCCL
                nvtxRangePushA("bcast p=0 (ping)");
                // Don't overwrite ping until compute finished reading it
                // (previous chunk / previous profile run).
                CHECK_CUDA(cudaStreamWaitEvent(comm_stream, ev_panel_consumed_ping, 0));
                CHECK_NCCL(ncclBroadcast((const void*)A_own, // send buffer (every rank calls this with its own A_own!!!)
                                         (void*)d_A_panel_ping.get(), //receive buffer - where the panel lands (ping or pong) on every rank
                                         a_bcast_count, // number of elements to transfer
                                         input_nccl_type, // element type (ncclHalf / ncclInt8 / ncclUint8 for packed int4)
                                         0, // root: WHICH rank in this communicator is the sender (its index within the row comm = mesh column)
                                         row_comm, // the group: my row's 2 ranks — defines who participates
                                         comm_stream)); // // CUDA stream the transfer is enqueued on (the comm stream, so it overlaps compute)
                CHECK_NCCL(ncclBroadcast((const void*)B_own,
                                         (void*)d_B_panel_ping.get(),
                                         b_bcast_count,
                                         input_nccl_type,
                                         0,
                                         col_comm,
                                         comm_stream));
                CHECK_CUDA(cudaEventRecord(ev_panel_ready_ping, comm_stream));
                nvtxRangePop();
#endif
            }

            // Panel loop: compute step p from one buffer while the comm
            // stream prefetches step p+1 into the other (ping/pong).
            for (int p = 0; p < tp_cols; ++p) {
                // Even steps read ping, odd steps read pong.
                const int buf = p & 1;
                // Defaults for the no-TP case: compute directly from my own shards.
                const uint8_t* A_panel = A_own;
                const uint8_t* B_panel = B_own;
                float* C_partial = (buf == 0) ? d_C_partial_ping.get() : d_C_partial_pong.get();

#ifdef USE_NCCL
                if (tp_cols > 1) {
                    // Block compute until this step's panels have fully arrived.
                    CHECK_CUDA(cudaStreamWaitEvent(compute_stream,
                                                   (buf == 0) ? ev_panel_ready_ping : ev_panel_ready_pong,
                                                   0));

                    // Compute from the received panels instead of my own shards.
                    A_panel = (buf == 0) ? d_A_panel_ping.get() : d_A_panel_pong.get();
                    B_panel = (buf == 0) ? d_B_panel_ping.get() : d_B_panel_pong.get();

                    // Prefetch: broadcast step p+1's panels into the OTHER buffer
                    // (on comm stream) while this step's GEMM runs below.
                    const int next_p = p + 1;
                    if (next_p < tp_cols) {
                        snprintf(nvtx_name, sizeof(nvtx_name), "bcast p=%d (prefetch)", next_p);
                        nvtxRangePushA(nvtx_name);
                        const int next_buf = next_p & 1;
                        uint8_t* A_next = (next_buf == 0) ? d_A_panel_ping.get() : d_A_panel_pong.get();
                        uint8_t* B_next = (next_buf == 0) ? d_B_panel_ping.get() : d_B_panel_pong.get();
                        CHECK_CUDA(cudaStreamWaitEvent(comm_stream,
                                                       (next_buf == 0) ? ev_panel_consumed_ping : ev_panel_consumed_pong,
                                                       0));
                        CHECK_NCCL(ncclBroadcast((const void*)A_own,
                                                 (void*)A_next,
                                                 a_bcast_count,
                                                 input_nccl_type,
                                                 next_p,
                                                 row_comm,
                                                 comm_stream));
                        CHECK_NCCL(ncclBroadcast((const void*)B_own,
                                                 (void*)B_next,
                                                 b_bcast_count,
                                                 input_nccl_type,
                                                 next_p,
                                                 col_comm,
                                                 comm_stream));
                        // Signal compute: next step's panels are complete.
                        CHECK_CUDA(cudaEventRecord((next_buf == 0) ? ev_panel_ready_ping : ev_panel_ready_pong,
                                                   comm_stream));
                        nvtxRangePop();
                    }
                }
#endif

                // GEMM: C_partial = A_panel · B_panel (this step's product).
                snprintf(nvtx_name, sizeof(nvtx_name), "gemm p=%d", p);
                nvtxRangePushA(nvtx_name);
                CHECK_CUDA(cudaMemsetAsync(C_partial, 0, c_chunk_elems * sizeof(float), compute_stream));
                launch_kernel(reinterpret_cast<const half*>(A_panel),
                              reinterpret_cast<const half*>(B_panel),
                              C_partial, cfg, compute_stream);
                nvtxRangePop();
#ifdef USE_NCCL
                if (tp_cols > 1) {
                    // Panels are last read by launch_kernel: mark buffer reusable.
                    CHECK_CUDA(cudaEventRecord((buf == 0) ? ev_panel_consumed_ping : ev_panel_consumed_pong,
                                               compute_stream));
                }
#endif

                // Accumulate: C_accum += C_partial (sums the tp_cols panel steps).
                snprintf(nvtx_name, sizeof(nvtx_name), "accum p=%d", p);
                nvtxRangePushA(nvtx_name);
                constexpr int kAccThreads = 256;
                const int acc_blocks = (int)((c_chunk_elems + kAccThreads - 1) / kAccThreads);
                if (integer_output) {
                    accumulate_inplace_int32<<<acc_blocks, kAccThreads, 0, compute_stream>>>(
                        reinterpret_cast<int32_t*>(d_C_accum.get()),
                        reinterpret_cast<const int32_t*>(C_partial), c_chunk_elems);
                } else {
                    accumulate_inplace<<<acc_blocks, kAccThreads, 0, compute_stream>>>(
                        d_C_accum.get(), C_partial, c_chunk_elems);
                }
                CHECK_CUDA(cudaGetLastError());
                nvtxRangePop();
            }

            nvtxRangePop(); // chunk
    };

    // ── 4. TIMED LOOP ─────────────────────────────────────────────────────────
    // profile_runs × chunks × run_chunk(), bracketed by cuda events → ms.
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

    // ── 5. VERIFY-TP (untimed, all kernel storage variants) ──────────────────
    // Re-run chunk 0 through the complete panel-broadcast and accumulation path,
    // then compare 64 sampled local C elements against full-global-K CPU dots
    // recomputed from the hash generator (sharding.cu).
    if (args.verify_tp) {
        nvtxRangePushA("verify-tp");
        run_chunk(0);

        constexpr int kSamples = 64;
        unsigned s = 0x9E3779B9u ^ (unsigned)rank;
        int fails = 0;
        float max_rel = 0.0f;
        int32_t max_abs_int = 0;

        if (integer_output) {
            std::vector<int32_t> h_C(szC_l);
            CHECK_CUDA(cudaMemcpyAsync(h_C.data(), d_C_accum.get(),
                                       szC_l * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, compute_stream));
            CHECK_CUDA(cudaStreamSynchronize(compute_stream));
            for (int i = 0; i < kSamples; ++i) {
                s = s * 1664525u + 1013904223u;
                const int m = (int)((s >> 8) % (unsigned)local_M);
                s = s * 1664525u + 1013904223u;
                const int n = (int)((s >> 8) % (unsigned)local_N);
                const int gm = coord.row * local_M + m;
                const int gn = coord.col * local_N + n;
                const int32_t ref = cpu_ref_c_value_int(0, gm, gn, K, N, data_kind);
                const int32_t out = h_C[(size_t)m * local_N + n];
                const int64_t diff = (int64_t)ref - out;
                const int32_t abs_err = (int32_t)(diff < 0 ? -diff : diff);
                if (abs_err > max_abs_int) max_abs_int = abs_err;
                if (abs_err != 0) ++fails;
            }
        } else {
            std::vector<float> h_C(szC_l);
            CHECK_CUDA(cudaMemcpyAsync(h_C.data(), d_C_accum.get(),
                                       szC_l * sizeof(float),
                                       cudaMemcpyDeviceToHost, compute_stream));
            CHECK_CUDA(cudaStreamSynchronize(compute_stream));
            for (int i = 0; i < kSamples; ++i) {
                s = s * 1664525u + 1013904223u;
                const int m = (int)((s >> 8) % (unsigned)local_M);
                s = s * 1664525u + 1013904223u;
                const int n = (int)((s >> 8) % (unsigned)local_N);
                const float ref = cpu_ref_c_value(0,
                                                  coord.row * local_M + m,
                                                  coord.col * local_N + n, K, N);
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

    // ── 6. STATS ──────────────────────────────────────────────────────────────
    // Allreduce the per-rank times over the world comm → avg across ranks and
    // max (= critical path, the number that matters for wall time).
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
