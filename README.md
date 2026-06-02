# TensorParallelPTX

TensorParallelPTX is a CUDA prototype for profiling large GEMMs with 2D tensor parallelism on NVIDIA GPUs (target: L4, sm_89).

## What this codebase does

- Runs an FP16 WMMA kernel (`kernels/fp16_wmma.cu`) for large GEMMs.
- Splits the global GEMM into a 2D TP mesh (`tp_rows x tp_cols`), default `2x2` (4 ranks / 4 GPUs).
- Uses out-of-core shard generation (no full global A/B/C allocation on host RAM).
- Uses pinned host buffers and async copies for profiling flow.
- If built with NCCL, performs row-group all-reduce for K-split correctness when `tp_cols > 1`.

## Current workflow (implemented)

- One CPU host process manages one GPU/rank (`torchrun`/MPI style launch).
- Each rank uses two CUDA streams:
  - Compute stream for GEMM.
  - Communication stream for NCCL collectives.
- Work is pipelined in batch chunks (`--chunk-batches`) with ping/pong C buffers.
- Compute of chunk `i+1` can overlap communication of chunk `i`.
- For `tp_cols > 1`, NCCL row-group all-reduce combines K-split partial sums.

## How tensor parallelism is done

For global `M x K` times `K x N`:

- `local_M = M / tp_rows`
- `local_N = N / tp_cols`
- `local_K = K / tp_cols`

Each rank `(row, col)` computes its local shard with:

- `A_shard: local_M x local_K`
- `B_shard: local_K x local_N`
- `C_shard: local_M x local_N`

When `tp_cols > 1`, C shards are reduced across each row-group (NCCL all-reduce) to combine K-split partial sums.

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

`--chunk-batches` is the chunk size for pipelining. It means how many batches are grouped into one compute+communication chunk.

- If `B=4` and `--chunk-batches 1`: 4 chunks (`[0] [1] [2] [3]`) for maximum overlap opportunity.
- If `B=4` and `--chunk-batches 2`: 2 chunks (`[0,1] [2,3]`) less launch overhead, less fine-grained overlap.
- If `B=4` and `--chunk-batches 4`: 1 chunk (no chunk-level pipeline overlap).

Rule of thumb:

- Start with `--chunk-batches 1` for latency-focused profiling.
- Increase to `2` or `4` if kernel launch/collective overhead becomes dominant.

Summary: each GPU is managed by one host process and uses two CUDA streams. The compute stream runs GEMM for the current chunk, while the communication stream runs NCCL all-reduce for the previous/ready chunk. Double buffering (ping/pong C buffers) keeps these chunks separate in memory so compute and communication can overlap safely.

### Concrete pipeline steps (per single GPU)

Example assumes `B=4`, `--chunk-batches 1` (chunks 0,1,2,3), and one GPU process:

1. Preload stage:
  - Build local `A`/`B` shards for all batches.
  - Copy `A`/`B` to device once.
2. Start timing event.
3. Chunk 0 on `ping` buffer:
  - `stream0` (compute): memset `C_ping`, run GEMM(chunk0), record compute-done event.
  - `stream1` (comm): wait on event, run NCCL all-reduce on `C_ping`, record comm-done-ping.
4. Chunk 1 on `pong` buffer:
  - `stream0`: memset `C_pong`, GEMM(chunk1), record compute-done.
  - `stream1`: wait, all-reduce `C_pong`, record comm-done-pong.
5. Chunk 2 reuses `ping`:
  - `stream0` waits for comm-done-ping, then memset + GEMM(chunk2).
  - `stream1` waits, all-reduce `C_ping`, record comm-done-ping.
6. Chunk 3 reuses `pong`:
  - `stream0` waits for comm-done-pong, then memset + GEMM(chunk3).
  - `stream1` waits, all-reduce `C_pong`, record comm-done-pong.
7. Stop timing on comm stream and synchronize.

Overlap intuition (still per single GPU): while `stream1` communicates chunk `i`, `stream0` can compute chunk `i+1` on the other buffer.

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
