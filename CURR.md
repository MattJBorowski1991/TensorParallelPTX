# Current State

Two TP strategies share the same kernels (`--tp-mode`): **SUMMA** (2D, comm on inputs) and **1D Megatron-style** (comm on output). Kernels are TP-agnostic — they receive `local_M/N/K` via `GemmConfig` and don't know the distribution.

## How tensor parallel is implemented (SUMMA, default)
- **Mesh**: 2D `tp_rows × tp_cols` (default 2×2), one host process per GPU (torchrun-style env vars). Square mesh required when `tp_cols > 1`.
- **Shards**: rank `(i,j)` owns `A[lM×lK]`, `B[lK×lN]`, `C[lM×lN]` with `lM=M/tp_rows`, `lN=N/tp_cols`, `lK=K/tp_cols`. Storage is dtype-aware: FP16, INT8, or packed INT4 inputs with FP32/INT32 outputs; integer B shards use the kernels’ transposed layout. Shards are generated directly from a deterministic hash.
- **SUMMA loop**: per chunk, `tp_cols` panel steps. Step `p`: NCCL broadcasts native representations (`ncclHalf` elements, `ncclInt8` elements, or packed-INT4 bytes via `ncclUint8`); the selected kernel computes a partial GEMM; FP32 or INT32 accumulation sums into `C_accum`.
- **Overlap**: two streams per rank (comm + compute), ping/pong panel buffers, event handshake in **both** directions: `panel_ready` (comm→compute) and `panel_consumed` (compute→comm, prevents overwrite of a buffer still being read).
- **Kernels**: the same FP16, INT8, and INT4 WMMA/PTX variants run unchanged under each TP strategy; runners supply dtype-correct buffers and local dimensions.

## 1D TP (`--tp-mode 1d-col | 1d-row`, `src/tp/oned_runner.cu`)
- Flat mesh: `--tp-rows 1 --tp-cols P`. Supports every compiled kernel variant with dtype-aware shards: FP16 inputs/FP32 outputs, INT8 inputs/INT32 outputs, and packed INT4 inputs/INT32 outputs.
- **1d-col** (column-parallel): A replicated (regenerated locally — free with the deterministic generator), B split by N. GEMM needs no comm; `ncclAllGather` materializes full C. Real transformers skip the gather by pairing with a row-parallel layer.
- **1d-row** (row-parallel): A/B split by K. Each rank computes a full-size *partial* C; `ncclAllReduce` sums them — the collective is the accumulation (no `accumulate_inplace` kernel).
- Overlap: C double-buffered; collective for chunk *c* on `comm_stream` overlaps GEMM of chunk *c+1* on `compute_stream` (`c_ready`/`c_free` event handshake).
- Comm volume (P=4, 16384³, per rank per chunk): FP16 SUMMA moves roughly 0.5 GiB of inputs (half for INT8, one quarter for INT4); 1d-row moves ~1.5 GiB of 32-bit C, while 1d-col gathers ~0.75 GiB — or **zero** if C stays sharded.

## Correctness checks
- `--verify` (rank 0, off by default): single-GPU 1024³ kernel run vs **two references**:
  - cached CPU GEMM (exact for int8/int4, fp32-accumulated for fp16);
  - **cuBLAS GemmEx** on GPU (fp16: fp32 accumulate; int8: exact int32; int4: no cuBLAS support — CPU reference is authoritative).
- `--verify-tp` (on by default, untimed): re-runs chunk 0 and checks 64 sampled C elements per rank against deterministic CPU dot products for every TP mode and kernel variant. INT8/INT4 checks are exact. Disable with `--no-verify-tp` for profiling-only captures.

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

---

# GPU playbook — run top to bottom on the rental box

Cheapest failures first. Stop at any FAIL. Approx cost of the whole playbook: ~20–30 min on 4×L4.
(No NCU here: per-kernel profiling was already done single-GPU and the kernels are unchanged under TP — nsys is where all the new TP information lives.)

```zsh
# ── 0. Environment check (free) ──────────────────────────────────────────────
nvcc --version && nsys --version
nvidia-smi topo -m          # compare against the P2P matrix the binary prints
which torchrun || pip install torch --index-url https://download.pytorch.org/whl/cu121

VARIANTS=(
  fp16_wmma
  int8_wmma
  int8_ptx_mma_k32
  int4_wmma
  int4_ptx_mma_k64_x4_x2nontrans_ca
)
```

```zsh
# ── 1. Compile all variants (~5 min, catches all build trivia at once) ───────
for K in "${VARIANTS[@]}"; do
  echo "=== build $K ==="
  cmake -DKERNEL_VARIANT="$K" -S . -B "builds/$K" >/dev/null &&
    cmake --build "builds/$K" -j >/dev/null ||
    { echo "BUILD FAIL: $K"; break; }
done
```

```zsh
# ── 2. Kernel correctness, single GPU (baseline PASS for every variant) ──────
# FP16/INT8 also check cuBLAS; INT4 reports cuBLAS skipped (unsupported).
for K in "${VARIANTS[@]}"; do
  echo "=== verify $K ==="
  cmake -DKERNEL_VARIANT="$K" -S . -B "builds/$K" >/dev/null &&
    cmake --build "builds/$K" -j >/dev/null &&
    "./builds/$K/bin/tensor_parallel_ptx" \
      --tp-rows 1 --tp-cols 1 --verify --no-profile ||
    { echo "VERIFY FAIL: $K"; break; }
done
```

```zsh
# ── 3. TP correctness, 4 GPUs, every kernel in all three modes ───────────────
for K in "${VARIANTS[@]}"; do
  cmake -DKERNEL_VARIANT="$K" -S . -B "builds/$K" >/dev/null &&
    cmake --build "builds/$K" -j >/dev/null ||
    { echo "BUILD FAIL: $K"; break; }

  RUN4=(
    torchrun --standalone --nproc_per_node=4 --no-python
    "./builds/$K/bin/tensor_parallel_ptx"
    --M 4096 --N 4096 --K 4096 --B 1 --profile-runs 1
    --walltime-file /tmp/tpptx_verify_tp.txt
  )
  echo "=== verify SUMMA $K ==="
  "${RUN4[@]}" --tp-rows 2 --tp-cols 2 &&
    echo "=== verify 1d-row $K ===" &&
    "${RUN4[@]}" --tp-mode 1d-row --tp-rows 1 --tp-cols 4 &&
    echo "=== verify 1d-col $K ===" &&
    "${RUN4[@]}" --tp-mode 1d-col --tp-rows 1 --tp-cols 4 ||
    { echo "TP RUN FAIL: $K"; break; }
done
```

```zsh
# ── 4. NSYS, big shape: all 5 kernels on SUMMA 2x2 (~2-3 min each) ───────────
for K in "${VARIANTS[@]}"; do
  OUT="prof/nsys/$K"
  mkdir -p "$OUT"
  cmake -DKERNEL_VARIANT="$K" -S . -B "builds/$K" >/dev/null &&
    cmake --build "builds/$K" -j >/dev/null ||
    { echo "BUILD FAIL: $K"; break; }
  nsys profile --force-overwrite true --trace cuda,nvtx,osrt --sample=none --cuda-memory-usage=true \
    -o "$OUT/nsys_$K" \
    torchrun --standalone --nproc_per_node=4 --no-python "./builds/$K/bin/tensor_parallel_ptx" \
    --M 16384 --N 16384 --K 16384 --tp-rows 2 --tp-cols 2 --B 4 --chunk-batches 1 \
    --profile-runs 2 --no-verify-tp \
    --walltime-file "$OUT/walltime_$K.txt" ||
    { echo "PROFILE FAIL: $K"; break; }
  nsys stats --force-export=true --report cuda_api_gpu_sum,cuda_gpu_kern_sum,nvtx_sum --format table \
    "$OUT/nsys_$K.nsys-rep" >"$OUT/nsys_${K}_summary.txt" ||
    { echo "STATS FAIL: $K"; break; }
done
```

```zsh
# ── 5. NSYS, big shape: all kernels on both 1D modes ─────────────────────────
for K in "${VARIANTS[@]}"; do
  cmake -DKERNEL_VARIANT="$K" -S . -B "builds/$K" >/dev/null &&
    cmake --build "builds/$K" -j >/dev/null ||
    { echo "BUILD FAIL: $K"; break; }

  for MODE in 1d-row 1d-col; do
    OUT="prof/nsys/${K}_${MODE}"
    mkdir -p "$OUT"
    nsys profile --force-overwrite true --trace cuda,nvtx,osrt --sample=none --cuda-memory-usage=true \
      -o "$OUT/nsys_${K}_${MODE}" \
      torchrun --standalone --nproc_per_node=4 --no-python "./builds/$K/bin/tensor_parallel_ptx" \
      --M 16384 --N 16384 --K 16384 --tp-mode "$MODE" --tp-rows 1 --tp-cols 4 \
      --B 4 --profile-runs 2 --no-verify-tp \
      --walltime-file "$OUT/walltime_${K}_${MODE}.txt" ||
      { echo "PROFILE FAIL: ${K}_${MODE}"; break 2; }
    nsys stats --force-export=true --report cuda_api_gpu_sum,cuda_gpu_kern_sum,nvtx_sum --format table \
      "$OUT/nsys_${K}_${MODE}.nsys-rep" >"$OUT/nsys_${K}_${MODE}_summary.txt" ||
      { echo "STATS FAIL: ${K}_${MODE}"; break 2; }
  done
done
```

```zsh
# ── 6. Collect results ────────────────────────────────────────────────────────
cat prof/nsys/*/walltime_*.txt          # one line per run: kernel, mode, shape, ms
ls -la prof/nsys/*/                     # .nsys-rep for the GUI, .txt summaries
# then: tar/scp prof/ back to the laptop, or open reports in nsys-ui
```

Notes:
- If a build fails in step 1, fix and re-run step 1 only — nothing later depends on partial state.
- Step 4/5 walltime files are the input for LEARN.md experiments 1–3 and 8.
- `--verify` (step 2) writes reference caches to `prof/cache/` on first run — later runs are instant.
