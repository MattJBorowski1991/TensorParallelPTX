# TensorParallelPTX

TensorParallelPTX is a CUDA prototype for profiling large GEMMs with 2D tensor parallelism on NVIDIA GPUs (target: L4, sm_89).

## What this codebase does

- Runs an FP16 WMMA kernel (`kernels/fp16_wmma.cu`) for large GEMMs.
- Splits the global GEMM into a 2D TP mesh (`tp_rows x tp_cols`), default `2x2` (4 ranks / 4 GPUs).
- Uses out-of-core shard generation (no full global A/B/C allocation on host RAM).
- Uses pinned host buffers and async copies for profiling flow.
- If built with NCCL, performs row-group all-reduce for K-split correctness when `tp_cols > 1`.

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

## Defaults (profiling-ready)

Current defaults in `src/main.cu`:

- `M=N=K=16384`
- `tp_rows=2`, `tp_cols=2` (4 GPUs)
- `verify=false`
- `profile=true`

## Build

### Linux / WSL / environments with `cmake` and `nvcc` in PATH

```bash
cmake -S . -B build
cmake --build build -j
```

### Notes

- NCCL is auto-detected in CMake.
- For `2x2` TP (`tp_cols=2`), NCCL must be found at build time.

## Run profiling (2x2, 4 GPUs, 16384)

```bash
./build/bin/tensor_parallel_ptx --M 16384 --N 16384 --K 16384 --tp-rows 2 --tp-cols 2 --B 4 --profile-runs 5 --no-verify
```

## Quick sanity run (smaller)

```bash
./build/bin/tensor_parallel_ptx --M 4096 --N 4096 --K 4096 --tp-rows 2 --tp-cols 2 --B 1 --profile-runs 3 --no-verify
```

## Nsight Compute (all metrics)

Run from repo root. Use a small shape for NCU to keep collection time reasonable.

### 1) Text file with all collected metrics

```bash
ncu --set full --target-processes all \
	./build/bin/tensor_parallel_ptx \
	--M 1024 --N 1024 --K 1024 --tp-rows 2 --tp-cols 2 --B 1 --profile-runs 1 --no-verify \
	> ncu_all_metrics.txt 2>&1
```

### 2) `.ncu-rep` report with all collected metrics

```bash
ncu --set full --target-processes all -o ncu_all_metrics \
	./build/bin/tensor_parallel_ptx \
	--M 1024 --N 1024 --K 1024 --tp-rows 2 --tp-cols 2 --B 1 --profile-runs 1 --no-verify
```

This creates `ncu_all_metrics.ncu-rep`.

## Nsight Systems 

```bash
nsys profile --force-overwrite true \
  --trace cuda,nvtx,osrt \
  --sample=none \
  --cuda-memory-usage=true \
  -o tp_timeline \
  ./build/bin/tensor_parallel_ptx \
  --M 1024 --N 1024 --K 1024 --tp-rows 2 --tp-cols 2 --B 4 --profile-runs 5 --no-verify
```

This is a good baseline Nsight Systems command for end-to-end TP profiling.

- Captures CUDA API + kernels, NVTX ranges (if present), and OS runtime activity.
- Writes `tp_timeline.nsys-rep` (report) and `tp_timeline.qdstrm` (raw stream).
- Uses `--sample=none` to reduce overhead and avoid CPU sampling noise.

Common optional additions (use only when needed):

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

For quick communication/overlap debugging, prefer adding `,nccl` to `--trace` first.

### Export a readable Nsight Systems text summary

```bash
nsys stats --report cuda_api_gpu_sum,cuda_api_sum,cuda_gpu_kern_sum,osrt_sum,nvtx_sum \
  --format table \
  tp_timeline.nsys-rep > nsys_summary.txt
```

If `ncclsum` appears in `nsys stats --help-reports` on your version, you can add it to include NCCL summary stats.
