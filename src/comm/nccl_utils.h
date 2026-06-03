#pragma once

#include <string>

#ifdef USE_NCCL
#include <nccl.h>

#define CHECK_NCCL(call) \
    do { \
        ncclResult_t r_ = (call); \
        if (r_ != ncclSuccess) { \
            fprintf(stderr, "NCCL Error at %s:%d: %s\\n", __FILE__, __LINE__, ncclGetErrorString(r_)); \
            exit(1); \
        } \
    } while (0)

std::string make_id_path(const std::string& prefix, const std::string& tag);
void write_nccl_id_file(const std::string& path, const ncclUniqueId& id);
void read_nccl_id_file_retry(const std::string& path, ncclUniqueId* id_out);
#endif
