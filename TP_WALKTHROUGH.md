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

## How This Relates To The Kernels

The GEMM kernels are mostly TP-agnostic. They do not know about ranks, NCCL,
mesh rows, mesh columns, or SUMMA.

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
