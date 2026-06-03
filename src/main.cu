#include <stdio.h>

#include "src/app/cli.h"
#include "src/app/dist_context.h"
#include "src/app/logging.h"
#include "src/tp/summa_runner.h"
#include "src/verify/verify_runner.h"

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);
    if (!validate_cli_args(args)) {
        fprintf(stderr, "[error] invalid command-line arguments.\n");
        return 1;
    }

    const int num_ranks = args.tp_rows * args.tp_cols;
    const int num_gpus = query_num_gpus();
    DistContext dist = resolve_dist_context(args, num_ranks, num_gpus);

    print_run_header(args, dist.rank);

    if (args.verify) {
        print_verify_header(dist.rank);
        run_verify();
    }

    if (args.profile) {
        print_profile_header(args, dist.rank);
        ProfileStats stats = run_profile(args, dist);
        if (stats.avg_rank_ms <= 0.0f || stats.wall_ms <= 0.0f) {
            fprintf(stderr, "[profile] skipped: runtime validation failed or no successful launches.\n");
            return 1;
        }

        print_profile_summary(args, stats, dist.rank);
        if (dist.rank == 0) {
            append_walltime_log(args, stats);
            printf("[profile] wall-time log appended to: %s\n", args.walltime_file.c_str());
        }
    }

    return 0;
}
