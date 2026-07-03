#pragma once

#include <string>

// How the GEMM is distributed across ranks.
//   Summa  — 2D mesh, typed SUMMA panel broadcasts of A/B (comm on inputs)
//   OneDCol — Megatron column-parallel: A replicated, B split by N;
//             allgather of C shards (comm on output, optional)
//   OneDRow — Megatron row-parallel: A/B split by K; each rank computes a
//             full-size partial C; allreduce sums them (comm on output)
enum class TpMode { Summa, OneDCol, OneDRow };

inline const char* tp_mode_name(TpMode m) {
    switch (m) {
        case TpMode::Summa:   return "summa";
        case TpMode::OneDCol: return "1d-col";
        case TpMode::OneDRow: return "1d-row";
    }
    return "?";
}

struct Args {
    int M;
    int N;
    int K;
    int num_batches;
    int chunk_batches;
    TpMode tp_mode;
    int tp_rows;
    int tp_cols;
    bool verify;      // single-GPU kernel verify vs cached CPU reference (rank 0)
    bool verify_tp;   // sampled end-to-end TP verify on every rank (untimed)
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
