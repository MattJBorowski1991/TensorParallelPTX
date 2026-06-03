#pragma once

#include "src/app/types.h"

int query_num_gpus();
DistContext resolve_dist_context(const Args& args, int num_ranks, int num_gpus);
