# Current State (fp16 path)

Two TP strategies share the same kernels (`--tp-mode`): **SUMMA** (2D, comm on inputs) and **1D Megatron-style** (comm on output). Kernels are TP-agnostic — they receive `local_M/N/K` via `GemmConfig` and don't know the distribution.

## How tensor parallel is implemented (SUMMA, default)
- **Mesh**: 2D `tp_rows × tp_cols` (default 2×2), one host process per GPU (torchrun-style env vars). Square mesh required when `tp_cols > 1`.
- **Shards**: rank `(i,j)` owns `A[lM×lK]`, `B[lK×lN]`, `C[lM×lN]` with `lM=M/tp_rows`, `lN=N/tp_cols`, `lK=K/tp_cols`. Shards are generated out-of-core from a deterministic hash — no global matrices ever materialized.
- **SUMMA loop**: per chunk, `tp_cols` panel steps. Step `p`: NCCL broadcasts A panel across the row communicator (root `col=p`) and B panel down the column communicator (root `row=p`); kernel computes the partial GEMM; `accumulate_inplace` sums into `C_accum`.
- **Overlap**: two streams per rank (comm + compute), ping/pong panel buffers, event handshake in **both** directions: `panel_ready` (comm→compute) and `panel_consumed` (compute→comm, prevents overwrite of a buffer still being read).
- **Kernel**: WMMA m16n16k16, 8 warps/block (64×32 block tile), one 16×16 output tile per warp, per-warp smem tiles, `cp.async` double-buffered K-loop (`__syncwarp` after `wait_group` for cross-lane visibility).

## 1D TP (`--tp-mode 1d-col | 1d-row`, `src/tp/oned_runner.cu`)
- Flat mesh: `--tp-rows 1 --tp-cols P`. Isolated runner; reuses NCCL bootstrap, streams/events, verify-tp patterns.
- **1d-col** (column-parallel): A replicated (regenerated locally — free with the deterministic generator), B split by N. GEMM needs no comm; `ncclAllGather` materializes full C. Real transformers skip the gather by pairing with a row-parallel layer.
- **1d-row** (row-parallel): A/B split by K. Each rank computes a full-size *partial* C; `ncclAllReduce` sums them — the collective is the accumulation (no `accumulate_inplace` kernel).
- Overlap: C double-buffered; collective for chunk *c* on `comm_stream` overlaps GEMM of chunk *c+1* on `compute_stream` (`c_ready`/`c_free` event handshake).
- Comm volume vs SUMMA (P=4, 16384³, per rank per chunk): SUMMA receives ~512 MB fp16 inputs; 1d-row moves ~1.5 GB fp32 C (ring allreduce ≈ 2·(P−1)/P · size); 1d-col gathers ~0.75 GB — or **zero** if C stays sharded.

## Correctness checks
- `--verify` (rank 0, off by default): single-GPU 1024³ kernel run vs **two references**:
  - cached CPU GEMM (exact for int8/int4, fp32-accumulated for fp16);
  - **cuBLAS GemmEx** on GPU (fp16: fp32 accumulate; int8: exact int32; int4: no cuBLAS support — CPU reference is authoritative).
- `--verify-tp` (on by default, untimed, fp16 variants only): re-runs chunk 0 through the full SUMMA path, then each rank checks 64 sampled C elements against CPU dot products over the global K (recomputed from the deterministic generator). Catches broadcast-root, panel-order, and accumulation bugs end-to-end. Disable with `--no-verify-tp` if a capture must contain only the timed loop.

## Last measured (before fixes)
16384³, B=4, 2×2, L4: avg_rank 1447 ms ≈ 6.1 TFLOPS/rank ≈ **5% of L4 fp16 tensor peak**.

## Improvement ideas (roughly in order of expected payoff)
1. **Cooperative block tiling** — warps currently load their own smem tiles, so identical A tiles are fetched 4× and B tiles 2× per block. Load one block-wide tile cooperatively.
2. **Register tiling** — one accumulator fragment per warp gives poor latency hiding; use 2–4 output tiles per warp.
3. **Deeper cp.async pipeline** — `wait_group 0` right after `commit_group` overlaps only one `mma_sync`; use N-stage buffering with `wait_group N-1`.
4. **PAD=8** for smem tiles to kill bank conflicts (keeps 16 B cp.async alignment).
5. **Comms**: finer-grained panel sub-tiling to overlap broadcast with GEMM within a step; NVLink/P2P topology-aware communicators.
6. **Non-square meshes** — B K-split is indexed by row coord with `lK=K/tp_cols`, so only square meshes work today.
7. **cuBLAS perf roofline** — the cuBLAS reference now checks correctness; also timing it would give a realistic per-GPU target to chase (~85–95% of peak).
8. **`--verify-tp` for int variants** — profile-path buffers are fp16-typed/filled; int kernels reinterpret them, so the TP check is skipped there. Needs int shard generation + int reference.

---

# GPU playbook — run top to bottom on the rental box

Cheapest failures first. Stop at any FAIL. Approx cost of the whole playbook: ~20–30 min on 4×L4.
(No NCU here: per-kernel profiling was already done single-GPU and the kernels are unchanged under TP — nsys is where all the new TP information lives.)

```bash
# ── 0. Environment check (free) ──────────────────────────────────────────────
nvcc --version && nsys --version
nvidia-smi topo -m          # compare against the P2P matrix the binary prints
which torchrun || pip install torch --index-url https://download.pytorch.org/whl/cu121

VARIANTS="fp16_wmma int8_wmma int8_ptx_mma_k32 int4_wmma int4_ptx_mma_k64_x4_x2nontrans_ca"
```

```bash
# ── 1. Compile all variants (~5 min, catches all build trivia at once) ───────
for K in $VARIANTS; do
  echo "=== build $K ===" && cmake -DKERNEL_VARIANT=$K -S . -B build > /dev/null \
    && cmake --build build -j > /dev/null || { echo "BUILD FAIL: $K"; break; }
done
```

```bash
# ── 2. Kernel correctness, single GPU (expect 2x PASS per variant) ───────────
for K in $VARIANTS; do
  echo "=== verify $K ===" && cmake -DKERNEL_VARIANT=$K -S . -B build > /dev/null \
    && cmake --build build -j > /dev/null
  ./build/bin/tensor_parallel_ptx --tp-rows 1 --tp-cols 1 --verify --no-profile
done
```

```bash
# ── 3. TP correctness, 4 GPUs, all three modes (expect verify-tp PASS x4 ranks each) ─
cmake -DKERNEL_VARIANT=fp16_wmma -S . -B build > /dev/null && cmake --build build -j > /dev/null
RUN4="torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 4096 --N 4096 --K 4096 --B 1 --profile-runs 1"
$RUN4 --tp-rows 2 --tp-cols 2                          # summa
$RUN4 --tp-mode 1d-row --tp-rows 1 --tp-cols 4         # allreduce C
$RUN4 --tp-mode 1d-col --tp-rows 1 --tp-cols 4         # allgather C
```

```bash
# ── 4. NSYS, big shape: all 5 kernels on SUMMA 2x2 (~2-3 min each) ───────────
for K in $VARIANTS; do
  OUT=prof/nsys/$K && mkdir -p $OUT
  cmake -DKERNEL_VARIANT=$K -S . -B build > /dev/null && cmake --build build -j > /dev/null
  nsys profile --force-overwrite true --trace cuda,nvtx,osrt --sample=none --cuda-memory-usage=true \
    -o $OUT/nsys_$K \
    torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
    --M 16384 --N 16384 --K 16384 --tp-rows 2 --tp-cols 2 --B 4 --chunk-batches 1 --profile-runs 2 \
    --walltime-file $OUT/walltime_$K.txt
  nsys stats --report cuda_api_gpu_sum,cuda_gpu_kern_sum,nvtx_sum --format table \
    $OUT/nsys_$K.nsys-rep > $OUT/nsys_${K}_summary.txt
done
```

```bash
# ── 5. NSYS, big shape: fp16 on the 1D modes (SUMMA-vs-1D comparison data) ───
cmake -DKERNEL_VARIANT=fp16_wmma -S . -B build > /dev/null && cmake --build build -j > /dev/null
for MODE in 1d-row 1d-col; do
  OUT=prof/nsys/fp16_wmma_$MODE && mkdir -p $OUT
  nsys profile --force-overwrite true --trace cuda,nvtx,osrt --sample=none --cuda-memory-usage=true \
    -o $OUT/nsys_fp16_$MODE \
    torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
    --M 16384 --N 16384 --K 16384 --tp-mode $MODE --tp-rows 1 --tp-cols 4 --B 4 --profile-runs 2 \
    --walltime-file $OUT/walltime_fp16_$MODE.txt
  nsys stats --report cuda_api_gpu_sum,cuda_gpu_kern_sum,nvtx_sum --format table \
    $OUT/nsys_fp16_$MODE.nsys-rep > $OUT/nsys_fp16_${MODE}_summary.txt
done
```

```bash
# ── 6. Collect results ────────────────────────────────────────────────────────
cat prof/nsys/*/walltime_*.txt          # one line per run: kernel, mode, shape, ms
ls -la prof/nsys/*/                     # .nsys-rep for the GUI, .txt summaries
# then: tar/scp prof/ back to the laptop, or open reports in nsys-ui
```

Notes:
- If a build fails in step 1, fix and re-run step 1 only — nothing later depends on partial state.
- Step 4/5 walltime files are the input for LEARN.md experiments 1–3 and 8.
- `--verify` (step 2) writes reference caches to `prof/cache/` on first run — later runs are instant.
