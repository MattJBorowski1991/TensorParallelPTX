#include "src/comm/nccl_utils.h"

#ifdef USE_NCCL
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <chrono>

std::string make_id_path(const std::string& prefix, const std::string& tag) {
    return prefix + "_" + tag + ".bin";
}

void write_nccl_id_file(const std::string& path, const ncclUniqueId& id) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[error] failed to write NCCL id file: %s\n", path.c_str());
        exit(1);
    }
    size_t nw = fwrite(&id, 1, sizeof(ncclUniqueId), f);
    fclose(f);
    if (nw != sizeof(ncclUniqueId)) {
        fprintf(stderr, "[error] short write on NCCL id file: %s\n", path.c_str());
        exit(1);
    }
}

void read_nccl_id_file_retry(const std::string& path, ncclUniqueId* id_out) {
    constexpr int kMaxRetries = 10000;
    for (int i = 0; i < kMaxRetries; ++i) {
        FILE* f = fopen(path.c_str(), "rb");
        if (f) {
            size_t nr = fread(id_out, 1, sizeof(ncclUniqueId), f);
            fclose(f);
            if (nr == sizeof(ncclUniqueId)) return;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    fprintf(stderr, "[error] timed out waiting for NCCL id file: %s\n", path.c_str());
    exit(1);
}
#endif
