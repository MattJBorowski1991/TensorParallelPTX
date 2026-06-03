#pragma once

#include <string>

struct Args {
    int M;
    int N;
    int K;
    int num_batches;
    int chunk_batches;
    int tp_rows;
    int tp_cols;
    bool verify;
    bool profile;
    int profile_runs;
    std::string walltime_file;
    int rank;
    int world_size;
    int local_rank;
    std::string nccl_id_prefix;
};

struct ProfileStats {
    float avg_rank_ms;
    float wall_ms;
};

struct DistContext {
    bool enabled;
    int rank;
    int world_size;
    int local_rank;
    std::string nccl_id_prefix;
};
