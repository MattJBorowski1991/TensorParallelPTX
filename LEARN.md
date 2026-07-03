# Learning experiments

Ordered lab exercises for this repo. Run everything on the GPU box; each experiment names the question it answers. README has the commands; CURR.md has the design.

## 1. Scaling efficiency — *what does TP actually buy?*
Same global shape, one GPU vs four:
```bash
./build/bin/tensor_parallel_ptx --M 16384 --N 16384 --K 16384 --tp-rows 1 --tp-cols 1 --B 4
torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 16384 --N 16384 --K 16384 --tp-rows 2 --tp-cols 2 --B 4
```
Compute `speedup = wall_1x1 / wall_2x2` and `efficiency = speedup / 4`. Then answer from the nsys timeline: where did the missing percent go — broadcast time, accumulate kernels, or exposed (non-overlapped) comm?

## 2. Overlap — *is comm hidden behind compute?*
Profile `--chunk-batches 1` vs `--chunk-batches 4` with nsys. In the timeline, find the NVTX rows: does `bcast p=1 (prefetch)` run *under* `gemm p=0`, or after it? Measure exposed comm = chunk time − gemm time. Which chunking wins, and why?

## 3. Comm roofline — *could overlap even work here?* (pen & paper)
Per panel step each rank receives an A panel (`lM×lK` fp16) and a B panel (`lK×lN` fp16). At 16384³ 2×2 that is 2 × 128 MB. PCIe Gen4 x16 ≈ 25 GB/s effective → ~10 ms per step of comm. Compare with the measured `gemm p=N` duration. Is compute long enough to hide comm? At what M/N/K does it stop being?

## 4. Topology — *how do the GPUs talk?*
Read the P2P matrix the program prints at startup. Then:
```bash
nvidia-smi topo -m
NCCL_DEBUG=INFO torchrun ... 2>&1 | grep -E "Ring|via"
```
Map NCCL's chosen rings/transports (P2P? SHM? NET?) to the matrix. On PCIe-only L4 nodes, expect host-staged copies — find them in nsys as H2D/D2H pairs inside the broadcasts.

## 5. Kernel anatomy — *what limits the fp16 kernel?*
Use your **existing single-GPU NCU reports** (kernels are unchanged under TP — no new NCU runs needed): check SM %, tensor-core utilization %, smem bank conflicts, and the warp-stall breakdown. Match what you find to CURR.md improvements #1–4 — each stall reason maps to one planned fix. Predict which fix helps most *before* implementing it.

## 6. Sabotage (trust your verifier)
Each of these should make `verify-tp` FAIL (or hang) — confirm it does, then revert:
- Remove a `cudaStreamWaitEvent(compute_stream, ev_panel_ready_*)` → compute reads a half-broadcast panel.
- Swap the broadcast roots (`next_p` → `0`) → wrong panel, wrong math, clean-looking run.
- Remove the `__syncwarp()` after `cp.async.wait_group` in fp16_wmma.cu → may still pass (it's a race, not a guarantee of failure) — good lesson in why "it passed" ≠ "it's correct".

## 7. Reading order for the kernel ladder
1. `fp16_wmma.cu` — WMMA fragments, smem tiles, cp.async double buffering
2. `int8_wmma.cu` — same skeleton; what changes when matrix_b must be col_major (BT layout)
3. `int8_ptx_mma_k32.cu` — drop to raw PTX: `ldmatrix` lane addressing, `mma.sync` fragment ownership, the x4 register reorder
4. `int4_wmma.cu` — nibble packing; strides in elements vs bytes
5. `int4_ptx_mma_k64...cu` — both at once
For 3 and 5, write out which C elements *your* lane holds (the `out_row/out_col` math at the store) — that's the fragment-layout knowledge that transfers to every sm89 PTX kernel.

## 8. SUMMA vs 1D Megatron-style TP — *where should the comm live?*
Production LLM TP is 1D (column-parallel + row-parallel), not SUMMA. All three modes are implemented — same kernels, same world size, different comm placement:
```bash
torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 16384 --N 16384 --K 16384 --B 4 --tp-rows 2 --tp-cols 2                    # summa
torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 16384 --N 16384 --K 16384 --B 4 --tp-mode 1d-row --tp-rows 1 --tp-cols 4   # allreduce C
torchrun --standalone --nproc_per_node=4 --no-python ./build/bin/tensor_parallel_ptx \
  --M 16384 --N 16384 --K 16384 --B 4 --tp-mode 1d-col --tp-rows 1 --tp-cols 4   # allgather C
```
Compare wall times, then explain them from first principles: SUMMA broadcasts fp16 *inputs* (~512 MB/rank/chunk), 1d-row allreduces fp32 *output* (~1.5 GB moved), 1d-col gathers (~0.75 GB) — or nothing, if the shard feeds the next layer. In nsys, compare where the NVTX collective ranges sit relative to `gemm`. Now explain why transformers pair col+row layers so C never materializes — you've measured the reason.
