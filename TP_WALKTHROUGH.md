# Tensor Parallel Walkthrough

This is the small mental model for the repo. The examples use a 2x2 TP mesh
because it is the easiest case to see by hand.

## Global GEMM

The full problem is:

```text
C = A @ B

A: M x K
B: K x N
C: M x N
```

In tensor parallelism, no one GPU owns all of `A`, `B`, or `C`. Each rank owns
one shard and cooperates with the other ranks.

## 2x2 Rank Layout

Ranks are arranged row-major:

```text
              col 0      col 1
            +---------+---------+
row 0       | rank00  | rank01  |
            +---------+---------+
row 1       | rank10  | rank11  |
            +---------+---------+
```

For SUMMA, each rank `(i,j)` owns:

```text
A shard: rows for mesh row i, K slice for mesh col j
B shard: K rows for mesh row i, columns for mesh col j
C shard: rows for mesh row i, columns for mesh col j
```

With a 2x2 mesh, split the global matrices into blocks:

```text
A = [ A00  A01 ]      B = [ B00  B01 ]
    [ A10  A11 ]          [ B10  B11 ]

C = [ C00  C01 ]
    [ C10  C11 ]
```

Each `Cij` needs a full-K dot product, so it is a sum of two K-panel products:

```text
C00 = A00 @ B00 + A01 @ B10
C01 = A00 @ B01 + A01 @ B11
C10 = A10 @ B00 + A11 @ B10
C11 = A10 @ B01 + A11 @ B11
```

That is the core SUMMA idea: each step computes one K-slice contribution,
then the rank accumulates those partial results.

## What Rank00 Does

Rank00 owns output block `C00`.

It starts with its own local shards:

```text
rank00 owns A00, B00, C00
```

Panel step `p=0`:

```text
rank00 uses A00 and B00
C_partial = A00 @ B00
C_accum  += C_partial
```

Panel step `p=1`:

```text
rank00 receives A01 from rank01 across its row
rank00 receives B10 from rank10 down its column
C_partial = A01 @ B10
C_accum  += C_partial
```

After both panel steps:

```text
C_accum = A00 @ B00 + A01 @ B10 = C00
```

So rank00 never materializes the full `A`, `B`, or `C`. It only builds its own
local output tile by receiving the panels needed for each K slice.

## Who Broadcasts At Each Panel

`p` selects the mesh column that broadcasts `A`, and the mesh row that
broadcasts `B`.

For a 2x2 mesh:

```text
p=0:
  A sharers: rank00, rank10
  B sharers: rank00, rank01

p=1:
  A sharers: rank01, rank11
  B sharers: rank10, rank11
```

For rank00 specifically:

```text
p=0 uses A00 and B00
p=1 uses A01 and B10
```

For rank11:

```text
p=0 uses A10 and B01
p=1 uses A11 and B11
```

Same schedule, different local output tile.

## Why Ping/Pong Buffers Exist

Each rank has two panel receive buffers:

```text
PING
PONG
```

The code alternates:

```text
even p -> PING
odd  p -> PONG
```

For rank00 on a 2x2 mesh:

```text
comm stream:    [bcast p0 panels -> PING][bcast p1 panels -> PONG]
compute stream:                         [GEMM reads PING        ][GEMM reads PONG]
                                         ^ while GEMM p0 runs, p1 data can arrive
```

Without ping/pong, the rank would do:

```text
broadcast p0 -> GEMM p0 -> broadcast p1 -> GEMM p1
```

With ping/pong, communication for the next panel can overlap compute for the
current panel.

## What The Events Mean

There are two directions of synchronization:

```text
panel_ready:
  comm stream -> compute stream
  "The broadcast finished. GEMM may read this buffer."

panel_consumed:
  compute stream -> comm stream
  "GEMM finished reading this buffer. Broadcast may overwrite it."
```

This is what keeps the two streams overlapped without racing.

## What VERIFY-TP Checks

`--verify-tp` re-runs chunk 0 through the same panel-broadcast and accumulation
path.

Then it samples 64 local `C` elements and recomputes each one on the CPU as a
full-K dot product from the deterministic hash generator.

Important detail:

```text
GPU computes: local C tile for chunk0
CPU checks:  64 sampled elements from batch0
```

So verification does not materialize the massive global matrix. It checks a few
full-K reference dots.

## 1D Tensor Parallel Walkthrough

`oned_runner.cu` implements two 1D TP modes:

```text
1d-col: split B/C by N columns
1d-row: split A/B by K, then sum partial C
```

The mesh is flat:

```text
rank0   rank1   rank2   rank3
```

For a small 2-rank example, think of the matrices as:

```text
A = [ A0  A1 ]          B = [ B0 ]
                            [ B1 ]

C = A @ B = A0 @ B0 + A1 @ B1
```

That same equation can be distributed in two different ways.

## 1D Column Parallel

Column parallel splits the output columns.

For two ranks:

```text
B = [ B_col0  B_col1 ]
C = [ C_col0  C_col1 ]
```

Rank0 owns:

```text
A full
B_col0
C_col0
```

Rank1 owns:

```text
A full
B_col1
C_col1
```

Each rank computes a complete local shard:

```text
rank0: C_col0 = A @ B_col0
rank1: C_col1 = A @ B_col1
```

The math needs no communication before or during GEMM. Communication only
happens if the code wants to materialize full `C`:

```text
AllGather C shards -> full C
```

In real transformer stacks, this gather is often skipped. The next row-parallel
layer can consume the sharded output directly.

## 1D Row Parallel

Row parallel splits the K dimension.

For two ranks:

```text
A = [ A0  A1 ]
B = [ B0 ]
    [ B1 ]
```

Rank0 owns:

```text
A0
B0
```

Rank1 owns:

```text
A1
B1
```

Each rank computes a full-size partial output:

```text
rank0: C_partial0 = A0 @ B0
rank1: C_partial1 = A1 @ B1
```

Then NCCL sums the partial outputs:

```text
AllReduce sum:
C = C_partial0 + C_partial1
```

So in 1d-row, the collective is the accumulation step. There is no separate
`C_accum += C_partial` kernel like SUMMA uses.

## 1D Ping/Pong Buffers

The same double-buffering idea appears in `oned_runner.cu`, but it buffers
outputs instead of input panels.

SUMMA:

```text
PING/PONG hold A_panel and B_panel
comm fills next input panels while GEMM reads current input panels
```

1D:

```text
PING/PONG hold C output chunks
compute writes GEMM output into one buffer
NCCL communicates a C buffer while compute can use the other buffer
```

In `oned_runner.cu`:

```text
compute stream: [GEMM chunk0 -> C_ping] [GEMM chunk1 -> C_pong]
comm stream:                         [collective C_ping] [collective C_pong]
```

The events are analogous:

```text
c_ready:
  compute stream -> comm stream
  "GEMM finished writing C buffer. NCCL may read it."

c_free:
  comm stream -> compute stream
  "NCCL finished with C buffer. GEMM may overwrite it."
```

## SUMMA vs 1D In One Sentence

```text
SUMMA communicates input panels and accumulates local partial C.
1d-col computes sharded C and optionally allgathers output.
1d-row computes full partial C and allreduces output.
```

## How This Relates To The Kernels

The GEMM kernels are mostly TP-agnostic. They do not know about ranks, NCCL,
mesh rows, mesh columns, SUMMA, or 1D TP.

The runner gives each kernel:

```text
A_panel
B_panel
C_partial
local_M
local_N
local_K
```

The kernel just computes:

```text
C_partial = A_panel @ B_panel
```

TP is the orchestration around the kernel:

```text
choose the right panels
broadcast them
run local GEMM
accumulate partial C
repeat for every K panel
```

The CUDA/PTX still matters for performance, because a faster local GEMM changes
how much communication can be hidden. But the mathematical TP algorithm lives in
the runner, not inside the GEMM kernel.
