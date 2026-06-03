# TensorParallelPTX

TensorParallelPTX is a CUDA prototype for profiling large GEMMs with 2D tensor parallelism on NVIDIA GPUs (target: L4, sm_89).

## What this codebase does

- Runs an FP16 WMMA kernel (`kernels/fp16_wmma.cu`) for large GEMMs.
- Splits the global GEMM into a 2D TP mesh (`tp_rows x tp_cols`), default `2x2` (4 ranks / 4 GPUs).
- Uses out-of-core shard generation (no full global A/B/C allocation on host RAM).
- Uses pinned host buffers and async copies for profiling flow.
- If built with NCCL and `tp_cols > 1`, performs SUMMA-style panel broadcasts (A across rows, B across columns).

## Current workflow (implemented)

- One CPU host process manages one GPU/rank (`torchrun`/MPI style launch).
- Each rank uses two CUDA streams:
  - Compute stream for GEMM + accumulation.
  - Communication stream for NCCL panel broadcasts.
- Work is processed in batch chunks (`--chunk-batches`).
- For each chunk and each K-panel step `p`, NCCL broadcasts:
  - A panel from row root `col=p` to the row communicator.
  - B panel from column root `row=p` to the column communicator.
- Panel traffic is ping/pong double-buffered to overlap communication with compute.
- Partial C results are accumulated over all panel steps.

## How tensor parallelism is done

For global `M x K` times `K x N`:

- `local_M = M / tp_rows`
- `local_N = N / tp_cols`
- `local_K = K / tp_cols`

Each rank `(row, col)` computes its local shard with:

- `A_shard: local_M x local_K`
- `B_shard: local_K x local_N`
- `C_shard: local_M x local_N`

When `tp_cols > 1`, each rank performs `tp_cols` panel steps and accumulates local partial C contributions (SUMMA-style).

## SUMMA details (current implementation)

For rank `(i, j)` in a `tp_rows x tp_cols` mesh:

- Local ownership:
  - `A_ij`: rows `[i*lM, (i+1)*lM)`, K panel for local col ownership.
  - `B_ij`: K rows `[i*lK, (i+1)*lK)`, cols `[j*lN, (j+1)*lN)`.
  - `C_ij`: rows `[i*lM, (i+1)*lM)`, cols `[j*lN, (j+1)*lN)`.
- For each panel step `p = 0 .. tp_cols-1`:
  - Row communicator broadcast: `A_ip` is broadcast across row `i` (root `col=p`).
  - Column communicator broadcast: `B_pj` is broadcast down column `j` (root `row=p`).
  - Rank computes `C_ij += A_ip * B_pj`.
- Communication is issued on `comm_stream`, compute/accumulation on `compute_stream`.
- A/B panel traffic uses ping/pong buffers to overlap prefetch of step `p+1` with compute on step `p`.

## Launch model

- For TP meshes with multiple ranks, launch one process per GPU (for example via `torchrun`).
- `WORLD_SIZE` must equal `tp_rows * tp_cols`.
- `RANK` and `LOCAL_RANK` are consumed from launcher environment variables.

## Defaults (profiling-ready)

Current defaults in `src/main.cu`:

- `M=N=K=16384`
- `tp_rows=2`, `tp_cols=2` (4 GPUs)
- `verify=false`
- `profile=true`

## Chunk size (simple explanation)

`--chunk-batches` is the chunk size. It means how many batches are grouped into one panel-broadcast + GEMM accumulation chunk.

- If `B=4` and `--chunk-batches 1`: 4 chunks (`[0] [1] [2] [3]`) for maximum overlap opportunity.
- If `B=4` and `--chunk-batches 2`: 2 chunks (`[0,1] [2,3]`) less launch overhead, less fine-grained overlap.
- If `B=4` and `--chunk-batches 4`: 1 chunk (no chunk-level pipeline overlap).

Rule of thumb:

- Start with `--chunk-batches 1` for latency-focused profiling.
- Increase to `2` or `4` if kernel launch/collective overhead becomes dominant.

Summary: each GPU is managed by one host process. For each chunk, the rank runs `tp_cols` panel steps, broadcasts A/B panels with NCCL (if enabled), runs GEMM for each step, and accumulates partial C.

### Concrete pipeline steps (per single GPU)

Example assumes `B=4`, `--chunk-batches 1` (chunks 0,1,2,3), and one GPU process:

1. Preload stage:
  - Build local `A`/`B` shards for all batches.
  - Copy `A`/`B` to device once.
2. Start timing event.
3. Chunk 0:
  - Zero `C_accum`.
  - For `p=0..tp_cols-1`: broadcast A/B panels for step `p`, run GEMM, accumulate into `C_accum`.
  - While computing step `p`, the comm stream prefetches panels for step `p+1` into the other ping/pong buffer.
4. Chunk 1:
  - Repeat the same panel loop and accumulation.
5. Continue for all chunks and stop timing.

In the current implementation, communication and compute run on separate streams with event synchronization and ping/pong panel buffers.

## Build and run notes

- NCCL is auto-detected in CMake.
- For `2x2` TP (`tp_cols=2`), NCCL must be found at build time.
- To switch kernels, rebuild with `-DKERNEL_VARIANT=<variant>` flag.

## Commands and examples

### Build

Linux / WSL / environments with `cmake` and `nvcc` in `PATH`.

#### Default (FP16 WMMA)
```bash
cmake -S . -B build
cmake --build build -j
```

#### Specify kernel variant (fp16_wmma, int8_wmma, int8_ptx, int4_wmma, int4_ptx)
```bash
cmake -DKERNEL_VARIANT=int8_wmma -S . -B build
cmake --build build -j
```

### Run profiling (2x2, 4 GPUs, 16384)

```bash
torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 16384 --N 16384 --K 16384 --tp-rows 2 --tp-cols 2 --B 4 --chunk-batches 1 --profile-runs 5 --no-verify
```

### Quick sanity run (smaller)

```bash
torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 4096 --N 4096 --K 4096 --tp-rows 2 --tp-cols 2 --B 1 --chunk-batches 1 --profile-runs 3 --no-verify
```

### Compile and run (fp16_wmma, 16384, Nsight Systems)

```bash
KERNEL=fp16_wmma
OUTDIR=prof/nsys/fp16
mkdir -p ${OUTDIR}
cmake -DKERNEL_VARIANT=${KERNEL} -S . -B build
cmake --build build -j
nsys profile --force-overwrite true --trace cuda,nvtx,osrt --sample=none --cuda-memory-usage=true -o ${OUTDIR}/nsys_${KERNEL} torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx --M 16384 --N 16384 --K 16384 --tp-rows 2 --tp-cols 2 --B 4 --chunk-batches 1 --profile-runs 2 --no-verify --walltime-file ${OUTDIR}/walltime_${KERNEL}.txt
nsys stats --report cuda_api_gpu_sum,cuda_api_sum,cuda_gpu_kern_sum,osrt_sum,nvtx_sum --format table ${OUTDIR}/nsys_${KERNEL}.nsys-rep > ${OUTDIR}/nsys_${KERNEL}_summary.txt
```

### Nsight Compute (all metrics)

Run from repo root. Use a small shape for NCU to keep collection time reasonable.

### Select kernel variant and profile
Available variants: `fp16_wmma`, `int8_wmma`, `int8_ptx`, `int4_wmma`, `int4_ptx`

```bash
# Example: profile INT8 WMMA
KERNEL=int8_wmma
cmake -DKERNEL_VARIANT=${KERNEL} -S . -B build > /dev/null
cmake --build build -j > /dev/null
ncu --set full --target-processes all -o ncu_${KERNEL} \
  torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 1024 --N 1024 --K 1024 --tp-rows 2 --tp-cols 2 --B 1 --chunk-batches 1 --profile-runs 1 --no-verify \
  --walltime-file prof/walltime_${KERNEL}.txt

# Export to text
ncu --import ncu_${KERNEL}.ncu-rep --export text -o ncu_${KERNEL}.txt
```

### Quick command for any kernel variant
```bash
KERNEL=int8_wmma && cmake -DKERNEL_VARIANT=${KERNEL} -S . -B build > /dev/null && \
cmake --build build -j > /dev/null && \
ncu --set full -o ncu_${KERNEL} \
  torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 1024 --N 1024 --K 1024 --tp-rows 2 --tp-cols 2 --B 1 --chunk-batches 1 --profile-runs 1 --no-verify --walltime-file prof/walltime_${KERNEL}.txt && \
ncu --import ncu_${KERNEL}.ncu-rep --export text -o ncu_${KERNEL}.txt
```

Each run appends robust wall-time entries to `prof/walltime_${KERNEL}.txt`.

### Nsight Systems

### Select kernel variant and profile
Available variants: `fp16_wmma`, `int8_wmma`, `int8_ptx`, `int4_wmma`, `int4_ptx`

```bash
# Example: profile INT8 WMMA with Nsight Systems
KERNEL=int8_wmma
cmake -DKERNEL_VARIANT=${KERNEL} -S . -B build > /dev/null
cmake --build build -j > /dev/null
nsys profile --force-overwrite true \
  --trace cuda,nvtx,osrt \
  --sample=none \
  --cuda-memory-usage=true \
  -o nsys_${KERNEL} \
  torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 1024 --N 1024 --K 1024 --tp-rows 2 --tp-cols 2 --B 4 --chunk-batches 1 --profile-runs 5 --no-verify \
  --walltime-file prof/walltime_${KERNEL}.txt

# Export readable summary
nsys stats --report cuda_api_gpu_sum,cuda_api_sum,cuda_gpu_kern_sum,osrt_sum,nvtx_sum \
  --format table \
  nsys_${KERNEL}.nsys-rep > nsys_${KERNEL}_summary.txt
```

### Quick command for any kernel variant
```bash
KERNEL=int8_wmma && cmake -DKERNEL_VARIANT=${KERNEL} -S . -B build > /dev/null && \
cmake --build build -j > /dev/null && \
nsys profile --force-overwrite true --trace cuda,nvtx,osrt --sample=none -o nsys_${KERNEL} \
  torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 1024 --N 1024 --K 1024 --tp-rows 2 --tp-cols 2 --B 4 --chunk-batches 1 --profile-runs 5 --no-verify --walltime-file prof/walltime_${KERNEL}.txt && \
nsys stats --report cuda_api_gpu_sum,cuda_gpu_kern_sum -f table nsys_${KERNEL}.nsys-rep > nsys_${KERNEL}_summary.txt
```

Each run appends robust wall-time entries to `prof/walltime_${KERNEL}.txt`.

### Optional flags (add to base command as needed)

```bash
# Include NCCL API tracing explicitly in the timeline
--trace cuda,nvtx,osrt,nccl

# Add CUDA backtraces for launch-site attribution (higher overhead)
--cudabacktrace=true

# Enable CPU sampling for host-side hotspot analysis
--sample=cpu

# Collect GPU metrics over time (requires supported platform/GPU)
--gpu-metrics-device=all
```
