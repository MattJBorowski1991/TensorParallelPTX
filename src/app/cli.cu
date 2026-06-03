#include "src/app/cli.h"

#include <cstdlib>
#include <string>

#include "include/config.h"

Args parse_args(int argc, char** argv) {
    Args args{};
    args.M = DEFAULT_M;
    args.N = DEFAULT_N;
    args.K = DEFAULT_K;
    args.num_batches = DEFAULT_NUM_BATCHES;
    args.chunk_batches = 1;
    args.tp_rows = DEFAULT_TP_ROWS;
    args.tp_cols = DEFAULT_TP_COLS;
    args.verify = false;
    args.profile = true;
    args.profile_runs = DEFAULT_PROFILE_RUNS;
    args.walltime_file = "profile_walltime.txt";
    args.rank = -1;
    args.world_size = -1;
    args.local_rank = -1;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--M" && i + 1 < argc) args.M = atoi(argv[++i]);
        else if (arg == "--N" && i + 1 < argc) args.N = atoi(argv[++i]);
        else if (arg == "--K" && i + 1 < argc) args.K = atoi(argv[++i]);
        else if (arg == "--B" && i + 1 < argc) args.num_batches = atoi(argv[++i]);
        else if (arg == "--chunk-batches" && i + 1 < argc) args.chunk_batches = atoi(argv[++i]);
        else if (arg == "--tp-rows" && i + 1 < argc) args.tp_rows = atoi(argv[++i]);
        else if (arg == "--tp-cols" && i + 1 < argc) args.tp_cols = atoi(argv[++i]);
        else if (arg == "--no-verify") args.verify = false;
        else if (arg == "--no-profile") args.profile = false;
        else if (arg == "--profile-runs" && i + 1 < argc) args.profile_runs = atoi(argv[++i]);
        else if (arg == "--walltime-file" && i + 1 < argc) args.walltime_file = argv[++i];
        else if (arg == "--rank" && i + 1 < argc) args.rank = atoi(argv[++i]);
        else if (arg == "--world-size" && i + 1 < argc) args.world_size = atoi(argv[++i]);
        else if (arg == "--local-rank" && i + 1 < argc) args.local_rank = atoi(argv[++i]);
        else if (arg == "--nccl-id-prefix" && i + 1 < argc) args.nccl_id_prefix = argv[++i];
    }
    return args;
}

bool validate_cli_args(const Args& args) {
    return args.M > 0 && args.N > 0 && args.K > 0 && args.num_batches > 0 && args.profile_runs > 0;
}
