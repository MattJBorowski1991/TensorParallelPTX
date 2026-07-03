#include "src/app/dist_context.h"

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "include/cuda_utils.h"

static int env_to_int(const char* name, int fallback) {
    const char* v = getenv(name);
    return (v && v[0] != '\0') ? atoi(v) : fallback;
}

static std::string detect_default_nccl_prefix() {
    const char* run_id = getenv("TORCHELASTIC_RUN_ID");
    if (run_id && run_id[0] != '\0') {
        return std::string("/tmp/tpx_nccl_id_") + run_id;
    }
    return "/tmp/tpx_nccl_id";
}

int query_num_gpus() {
    int n = 0;
    CHECK_CUDA(cudaGetDeviceCount(&n));
    return n;
}

void print_p2p_topology(int num_gpus) {
    printf("\n--- GPU topology (P2P access matrix) ---\n     ");
    for (int j = 0; j < num_gpus; ++j) printf("GPU%d ", j);
    printf("\n");
    for (int i = 0; i < num_gpus; ++i) {
        printf("GPU%d ", i);
        for (int j = 0; j < num_gpus; ++j) {
            if (i == j) { printf("   . "); continue; }
            int can = 0;
            CHECK_CUDA(cudaDeviceCanAccessPeer(&can, i, j));
            printf("   %c ", can ? 'Y' : '-');
        }
        printf("\n");
    }
    printf("'-' = no P2P: NCCL stages traffic through host/PCIe.\n");
    printf("Also try: nvidia-smi topo -m   and   NCCL_DEBUG=INFO <run> (shows ring construction)\n");
}

DistContext resolve_dist_context(const Args& args, int num_ranks, int num_gpus) {
    DistContext dc{};
    dc.enabled = false;
    dc.rank = (args.rank >= 0) ? args.rank : env_to_int("RANK", -1);
    dc.world_size = (args.world_size > 0) ? args.world_size : env_to_int("WORLD_SIZE", -1);
    dc.local_rank = (args.local_rank >= 0) ? args.local_rank : env_to_int("LOCAL_RANK", -1);
    dc.nccl_id_prefix = args.nccl_id_prefix.empty() ? detect_default_nccl_prefix() : args.nccl_id_prefix;

    if (dc.rank >= 0 || dc.world_size > 0 || dc.local_rank >= 0) {
        dc.enabled = true;
    }

    if (dc.enabled) {
        if (dc.rank < 0 || dc.world_size <= 0) {
            fprintf(stderr, "[error] distributed mode requires rank/world-size (args or env RANK/WORLD_SIZE).\n");
            exit(1);
        }
        if (dc.world_size != num_ranks) {
            fprintf(stderr, "[error] world_size (%d) must equal tp_rows*tp_cols (%d).\n", dc.world_size, num_ranks);
            exit(1);
        }
        if (dc.rank < 0 || dc.rank >= dc.world_size) {
            fprintf(stderr, "[error] rank (%d) must be in [0, %d).\n", dc.rank, dc.world_size);
            exit(1);
        }
        if (dc.local_rank < 0) dc.local_rank = dc.rank % num_gpus;
        if (dc.local_rank < 0 || dc.local_rank >= num_gpus) {
            fprintf(stderr, "[error] local_rank (%d) must be in [0, %d).\n", dc.local_rank, num_gpus);
            exit(1);
        }
        return dc;
    }

    dc.rank = 0;
    dc.world_size = 1;
    dc.local_rank = 0;
    if (num_ranks > 1) {
        fprintf(stderr,
                "[error] tp_rows*tp_cols=%d requires one process per GPU. Launch with torchrun/mpirun and set RANK/WORLD_SIZE/LOCAL_RANK.\n",
                num_ranks);
        exit(1);
    }
    return dc;
}
