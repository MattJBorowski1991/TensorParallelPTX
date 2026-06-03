#pragma once

#include <cuda_fp16.h>

#include "include/tp_utils.h"

void fill_rank_batch_shards(
    half* A_shard,
    half* B_shard,
    int M,
    int N,
    int K,
    int local_M,
    int local_N,
    int local_K,
    int batch,
    RankCoord coord);
