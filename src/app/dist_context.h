#pragma once

#include "src/app/types.h"

int query_num_gpus();
DistContext resolve_dist_context(const Args& args, int num_ranks, int num_gpus);

// Print the GPU P2P access matrix (can GPU i read GPU j's memory directly?).
// If P2P is unavailable (typical PCIe-only boxes, e.g. most L4 nodes), NCCL
// routes broadcasts through host memory — expect that in the nsys timeline.
void print_p2p_topology(int num_gpus);
