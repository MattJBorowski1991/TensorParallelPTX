# TensorParallelPTX

CUDA/PTX learning project: large GEMMs with 2D tensor parallelism on NVIDIA L4 (sm_89), profiled with Nsight Systems. (Per-kernel NCU profiling was already done single-GPU; the kernels are unchanged under TP, so only nsys runs here.)

See [CURR.md](CURR.md) for how the implementation works today and the planned improvements, and [LEARN.md](LEARN.md) for the learning experiments to run against this code.

## Layout

- `kernels/` — one GEMM kernel per build variant:
  - `fp16_wmma` — WMMA m16n16k16, fp16 in / fp32 out (default)
  - `int8_wmma` — WMMA m16n16k16, int8 in / int32 out (B transposed)
  - `int8_ptx_mma_k32` — raw PTX `ldmatrix` + `mma.sync` m16n8k32
  - `int4_wmma` — WMMA m8n8k32, packed int4
  - `int4_ptx_mma_k64_x4_x2nontrans_ca` — raw PTX m16n8k64, packed int4
- `src/tp/` — SUMMA runner (NCCL panel broadcasts, dual streams, ping/pong buffers), shard generation
- `src/verify/` — correctness checks (CPU reference + cuBLAS cross-check)
- `src/app/` — CLI, torchrun-style rank resolution, logging

## Tensor parallelism (short version)

For global `M×K @ K×N` on a `tp_rows × tp_cols` mesh (square when `tp_cols > 1`), rank `(i,j)` owns:

- `A: lM × lK`, `B: lK × lN`, `C: lM × lN` with `lM = M/tp_rows`, `lN = N/tp_cols`, `lK = K/tp_cols`

When `tp_cols > 1`, each rank runs `tp_cols` SUMMA panel steps per chunk: NCCL broadcasts the A panel across the row communicator (root `col=p`) and the B panel down the column communicator (root `row=p`), computes the partial GEMM, and accumulates into `C`. Communication and compute run on separate streams with ping/pong panel buffers and event handshakes in both directions.

Shards are generated out-of-core from a deterministic hash — global matrices are never materialized.

## Launch model

One process per GPU (`WORLD_SIZE == tp_rows * tp_cols`); `RANK`/`LOCAL_RANK` come from the launcher (e.g. `torchrun`). `tp_cols > 1` requires NCCL (auto-detected by CMake).

## CLI flags

| Flag | Default | Meaning |
|---|---|---|
| `--M --N --K` | 16384 | global GEMM dims |
| `--B` | 4 | batches |
| `--chunk-batches` | 1 | batches per pipeline chunk (1 = max overlap) |
| `--tp-rows --tp-cols` | 2×2 | TP mesh |
| `--tp-mode` | `summa` | `summa` (2D panel broadcasts) \| `1d-col` (allgather C) \| `1d-row` (allreduce C); 1D modes need `--tp-rows 1`, P = `--tp-cols` |
| `--profile-runs` | 5 | timed iterations |
| `--verify` | off | rank 0: single-GPU 1024³ kernel vs cached CPU ref + cuBLAS |
| `--verify-tp` / `--no-verify-tp` | on | untimed end-to-end TP sample check for every mode and kernel variant |
| `--walltime-file` | `profile_walltime.txt` | append wall-time log (rank 0) |

## Build

```bash
cmake -DKERNEL_VARIANT=fp16_wmma -S . -B build   # or any variant name from Layout above
cmake --build build -j
```

## Smoke test (run this first on a fresh machine)

```bash
for K in fp16_wmma int8_wmma int8_ptx_mma_k32 int4_wmma int4_ptx_mma_k64_x4_x2nontrans_ca; do
  cmake -DKERNEL_VARIANT=$K -S . -B build > /dev/null && cmake --build build -j > /dev/null || break
  ./build/bin/tensor_parallel_ptx --tp-rows 1 --tp-cols 1 --verify --no-profile
done
# small 4-GPU TP run (verify-tp prints PASS/FAIL per rank):
cmake -DKERNEL_VARIANT=fp16_wmma -S . -B build && cmake --build build -j
torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 4096 --N 4096 --K 4096 --tp-rows 2 --tp-cols 2 --B 1 --profile-runs 1
# same problem, 1D Megatron-style modes:
torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 4096 --N 4096 --K 4096 --tp-mode 1d-row --tp-rows 1 --tp-cols 4 --B 1 --profile-runs 1
torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 4096 --N 4096 --K 4096 --tp-mode 1d-col --tp-rows 1 --tp-cols 4 --B 1 --profile-runs 1
```

## Profile

### Wall-time comparison (4×L4)

Critical-path wall time (`wall_ms`) for `M=N=K=16384`, `B=4`, and two
profile runs. SUMMA uses a 2×2 mesh; the 1D modes use a 1×4 group. These
measurements include Nsight Systems instrumentation overhead.

| Kernel | SUMMA 2×2 | 1D row 1×4 | 1D col 1×4 | Fastest |
|---|---:|---:|---:|---|
| `fp16_wmma` | **3038.494 ms** | 3466.801 ms | 3139.312 ms | SUMMA |
| `int8_wmma` | **1432.520 ms** | 1818.683 ms | 1556.154 ms | SUMMA |
| `int8_ptx_mma_k32` | 1015.542 ms | 2110.969 ms | **981.308 ms** | 1D col |
| `int4_wmma` | **823.104 ms** | 1401.107 ms | 1128.168 ms | SUMMA |
| `int4_ptx_mma_k64_x4_x2nontrans_ca` | **315.784 ms** | 1213.218 ms | 791.391 ms | SUMMA |

SUMMA is fastest for four of the five kernels. The 1D-row path is slowest
throughout because it all-reduces the full 32-bit output, while 1D-col uses
the smaller all-gather.

### Nsight Systems (timeline: kernels + NCCL + streams; use the big shape)

```bash
KERNEL=fp16_wmma
OUTDIR=prof/nsys/${KERNEL} && mkdir -p ${OUTDIR}
cmake -DKERNEL_VARIANT=${KERNEL} -S . -B build > /dev/null && cmake --build build -j > /dev/null
nsys profile --force-overwrite true --trace cuda,nvtx,osrt --sample=none --cuda-memory-usage=true \
  -o ${OUTDIR}/nsys_${KERNEL} \
  torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 16384 --N 16384 --K 16384 --tp-rows 2 --tp-cols 2 --B 4 --chunk-batches 1 --profile-runs 2 \
  --walltime-file ${OUTDIR}/walltime_${KERNEL}.txt
nsys stats --report cuda_api_gpu_sum,cuda_api_sum,cuda_gpu_kern_sum,osrt_sum,nvtx_sum --format table \
  ${OUTDIR}/nsys_${KERNEL}.nsys-rep > ${OUTDIR}/nsys_${KERNEL}_summary.txt
```

Optional nsys flags: `--trace cuda,nvtx,osrt,nccl` (NCCL API rows), `--cudabacktrace=true`, `--sample=cpu`, `--gpu-metrics-device=all`.

Each profiling run appends a wall-time line (timestamp, kernel, mode, shape, tp, avg_rank_ms, wall_ms) to the `--walltime-file`.
