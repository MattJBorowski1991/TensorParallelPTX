#include "src/verify/verify_runner.h"

#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "include/config.h"
#include "include/cuda_utils.h"
#include "src/solver.h"
#include "src/data.h"

namespace {

constexpr int kVerifyM = 1024;
constexpr int kVerifyN = 1024;
constexpr int kVerifyK = 1024;
constexpr int kVerifyBatches = 1;

constexpr uint32_t kCacheMagic = 0x58505456; // "VTPX"
constexpr uint32_t kCacheVersion = 1;

enum class VerifyKind : uint32_t {
    Fp16 = 1,
    Int8 = 2,
    Int4 = 3,
};

std::string kernel_variant_name() {
#ifdef TP_KERNEL_VARIANT
    return TP_KERNEL_VARIANT;
#else
    return "unknown";
#endif
}

VerifyKind detect_verify_kind() {
    const std::string v = kernel_variant_name();
    if (v.find("int4") != std::string::npos) return VerifyKind::Int4;
    if (v.find("int8") != std::string::npos) return VerifyKind::Int8;
    return VerifyKind::Fp16;
}

std::string cache_path_for(VerifyKind kind) {
    switch (kind) {
        case VerifyKind::Fp16:
            return "prof/cache/verify_fp16_m1024_n1024_k1024.bin";
        case VerifyKind::Int8:
            return "prof/cache/verify_int8_m1024_n1024_k1024.bin";
        case VerifyKind::Int4:
            return "prof/cache/verify_int4_m1024_n1024_k1024.bin";
    }
    return "prof/cache/verify_unknown.bin";
}

struct CacheHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t kind;
    int m;
    int n;
    int k;
    int b;
};

bool load_header(std::ifstream& in, VerifyKind expected) {
    CacheHeader h{};
    in.read(reinterpret_cast<char*>(&h), sizeof(h));
    if (!in) return false;
    return h.magic == kCacheMagic &&
           h.version == kCacheVersion &&
           h.kind == static_cast<uint32_t>(expected) &&
           h.m == kVerifyM && h.n == kVerifyN && h.k == kVerifyK && h.b == kVerifyBatches;
}

void write_header(std::ofstream& out, VerifyKind kind) {
    CacheHeader h{};
    h.magic = kCacheMagic;
    h.version = kCacheVersion;
    h.kind = static_cast<uint32_t>(kind);
    h.m = kVerifyM;
    h.n = kVerifyN;
    h.k = kVerifyK;
    h.b = kVerifyBatches;
    out.write(reinterpret_cast<const char*>(&h), sizeof(h));
}

void build_int8_inputs(std::vector<int8_t>& A,
                       std::vector<int8_t>& BT,
                       std::vector<int8_t>& B,
                       std::vector<int32_t>& C_ref) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-7, 7);

    A.resize((size_t)kVerifyM * kVerifyK);
    B.resize((size_t)kVerifyK * kVerifyN);
    BT.resize((size_t)kVerifyN * kVerifyK);
    C_ref.assign((size_t)kVerifyM * kVerifyN, 0);

    for (size_t i = 0; i < A.size(); ++i) A[i] = static_cast<int8_t>(dist(rng));
    for (size_t i = 0; i < B.size(); ++i) B[i] = static_cast<int8_t>(dist(rng));

    for (int k = 0; k < kVerifyK; ++k) {
        for (int n = 0; n < kVerifyN; ++n) {
            BT[(size_t)n * kVerifyK + k] = B[(size_t)k * kVerifyN + n];
        }
    }

    for (int m = 0; m < kVerifyM; ++m) {
        for (int n = 0; n < kVerifyN; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < kVerifyK; ++k) {
                acc += (int32_t)A[(size_t)m * kVerifyK + k] * (int32_t)B[(size_t)k * kVerifyN + n];
            }
            C_ref[(size_t)m * kVerifyN + n] = acc;
        }
    }
}

inline int8_t clamp_int4(int x) {
    if (x < -8) return -8;
    if (x > 7) return 7;
    return (int8_t)x;
}

uint8_t pack_int4_pair(int8_t low, int8_t high) {
    uint8_t lo = (uint8_t)(low & 0x0F);
    uint8_t hi = (uint8_t)(high & 0x0F);
    return (uint8_t)(lo | (uint8_t)(hi << 4));
}

void build_int4_inputs(std::vector<int8_t>& A_packed,
                       std::vector<int8_t>& BT_packed,
                       std::vector<int8_t>& A_unpacked,
                       std::vector<int8_t>& B_unpacked,
                       std::vector<int32_t>& C_ref) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-8, 7);

    A_unpacked.resize((size_t)kVerifyM * kVerifyK);
    B_unpacked.resize((size_t)kVerifyK * kVerifyN);
    C_ref.assign((size_t)kVerifyM * kVerifyN, 0);

    for (size_t i = 0; i < A_unpacked.size(); ++i) A_unpacked[i] = clamp_int4(dist(rng));
    for (size_t i = 0; i < B_unpacked.size(); ++i) B_unpacked[i] = clamp_int4(dist(rng));

    const size_t kBytes = (size_t)kVerifyK / 2;
    A_packed.resize((size_t)kVerifyM * kBytes);
    BT_packed.resize((size_t)kVerifyN * kBytes);

    for (int m = 0; m < kVerifyM; ++m) {
        for (int kb = 0; kb < kVerifyK; kb += 2) {
            int8_t v0 = A_unpacked[(size_t)m * kVerifyK + kb];
            int8_t v1 = A_unpacked[(size_t)m * kVerifyK + kb + 1];
            A_packed[(size_t)m * kBytes + kb / 2] = (int8_t)pack_int4_pair(v0, v1);
        }
    }

    for (int n = 0; n < kVerifyN; ++n) {
        for (int kb = 0; kb < kVerifyK; kb += 2) {
            int8_t v0 = B_unpacked[(size_t)kb * kVerifyN + n];
            int8_t v1 = B_unpacked[(size_t)(kb + 1) * kVerifyN + n];
            BT_packed[(size_t)n * kBytes + kb / 2] = (int8_t)pack_int4_pair(v0, v1);
        }
    }

    for (int m = 0; m < kVerifyM; ++m) {
        for (int n = 0; n < kVerifyN; ++n) {
            int32_t acc = 0;
            for (int k = 0; k < kVerifyK; ++k) {
                acc += (int32_t)A_unpacked[(size_t)m * kVerifyK + k] *
                       (int32_t)B_unpacked[(size_t)k * kVerifyN + n];
            }
            C_ref[(size_t)m * kVerifyN + n] = acc;
        }
    }
}

bool compare_int32_exact(const std::vector<int32_t>& ref,
                         const std::vector<int32_t>& out,
                         int32_t* max_abs,
                         double* rel_pct) {
    int32_t max_err = 0;
    double rel_sum = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        int32_t e = ref[i] - out[i];
        int32_t ae = e >= 0 ? e : -e;
        if (ae > max_err) max_err = ae;
        double denom = (double)(ref[i] >= 0 ? ref[i] : -ref[i]);
        if (denom < 1.0) denom = 1.0;
        rel_sum += ((double)ae / denom) * 100.0;
    }
    *max_abs = max_err;
    *rel_pct = rel_sum / (double)ref.size();
    return max_err == 0;
}

bool load_verify_cache_fp16(const std::string& path,
                            std::vector<half>& h_A,
                            std::vector<half>& h_B,
                            std::vector<float>& h_C_ref) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    if (!load_header(in, VerifyKind::Fp16)) return false;

    in.read(reinterpret_cast<char*>(h_A.data()), h_A.size() * sizeof(half));
    in.read(reinterpret_cast<char*>(h_B.data()), h_B.size() * sizeof(half));
    in.read(reinterpret_cast<char*>(h_C_ref.data()), h_C_ref.size() * sizeof(float));
    return in.good();
}

void save_verify_cache_fp16(const std::string& path,
                            const std::vector<half>& h_A,
                            const std::vector<half>& h_B,
                            const std::vector<float>& h_C_ref) {
    std::error_code ec;
    std::filesystem::create_directories("prof/cache", ec);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "[warn] failed to open verify cache for write: %s\n", path.c_str());
        return;
    }

    write_header(out, VerifyKind::Fp16);
    out.write(reinterpret_cast<const char*>(h_A.data()), h_A.size() * sizeof(half));
    out.write(reinterpret_cast<const char*>(h_B.data()), h_B.size() * sizeof(half));
    out.write(reinterpret_cast<const char*>(h_C_ref.data()), h_C_ref.size() * sizeof(float));
    if (!out.good()) {
        fprintf(stderr, "[warn] failed while writing verify cache: %s\n", path.c_str());
    }
}

bool load_verify_cache_int(const std::string& path,
                           VerifyKind kind,
                           std::vector<int8_t>& A,
                           std::vector<int8_t>& BT,
                           std::vector<int32_t>& C_ref) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    if (!load_header(in, kind)) return false;
    in.read(reinterpret_cast<char*>(A.data()), A.size() * sizeof(int8_t));
    in.read(reinterpret_cast<char*>(BT.data()), BT.size() * sizeof(int8_t));
    in.read(reinterpret_cast<char*>(C_ref.data()), C_ref.size() * sizeof(int32_t));
    return in.good();
}

void save_verify_cache_int(const std::string& path,
                           VerifyKind kind,
                           const std::vector<int8_t>& A,
                           const std::vector<int8_t>& BT,
                           const std::vector<int32_t>& C_ref) {
    std::error_code ec;
    std::filesystem::create_directories("prof/cache", ec);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        fprintf(stderr, "[warn] failed to open verify cache for write: %s\n", path.c_str());
        return;
    }
    write_header(out, kind);
    out.write(reinterpret_cast<const char*>(A.data()), A.size() * sizeof(int8_t));
    out.write(reinterpret_cast<const char*>(BT.data()), BT.size() * sizeof(int8_t));
    out.write(reinterpret_cast<const char*>(C_ref.data()), C_ref.size() * sizeof(int32_t));
    if (!out.good()) {
        fprintf(stderr, "[warn] failed while writing verify cache: %s\n", path.c_str());
    }
}

} // namespace

void run_verify() {
    const VerifyKind kind = detect_verify_kind();
    const std::string variant = kernel_variant_name();
    const std::string cache_path = cache_path_for(kind);

    GemmConfig vcfg{};
    vcfg.M = kVerifyM;
    vcfg.N = kVerifyN;
    vcfg.K = kVerifyK;
    vcfg.num_batches = kVerifyBatches;
    vcfg.warmups = 0;
    vcfg.runs = 1;
    vcfg.tp_rows = 1;
    vcfg.tp_cols = 1;
    vcfg.gpu_rank = 0;

    Solver solver;
    solver.configure(vcfg);

    if (kind == VerifyKind::Fp16) {
        std::vector<half> h_A((size_t)kVerifyBatches * kVerifyM * kVerifyK);
        std::vector<half> h_B((size_t)kVerifyBatches * kVerifyK * kVerifyN);
        std::vector<float> h_C_ref((size_t)kVerifyM * kVerifyN, 0.f);
        std::vector<float> h_C_out((size_t)kVerifyM * kVerifyN, 0.f);

        const bool cache_hit = load_verify_cache_fp16(cache_path, h_A, h_B, h_C_ref);
        if (!cache_hit) {
            generate_fp16(h_A.data(), h_B.data(), kVerifyM, kVerifyN, kVerifyK, kVerifyBatches);
            cpu_gemm_fp16(h_A.data(), h_B.data(), h_C_ref.data(), kVerifyM, kVerifyN, kVerifyK);
            save_verify_cache_fp16(cache_path, h_A, h_B, h_C_ref);
        }

        DeviceBuffer<half> d_A((size_t)kVerifyM * kVerifyK);
        DeviceBuffer<half> d_B((size_t)kVerifyK * kVerifyN);
        DeviceBuffer<float> d_C((size_t)kVerifyM * kVerifyN);
        CHECK_CUDA(cudaMemcpy(d_A.get(), h_A.data(), (size_t)kVerifyM * kVerifyK * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_B.get(), h_B.data(), (size_t)kVerifyK * kVerifyN * sizeof(half), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C.get(), 0, (size_t)kVerifyM * kVerifyN * sizeof(float)));

        solver.run(d_A.get(), d_B.get(), d_C.get());
        CHECK_CUDA(cudaMemcpy(h_C_out.data(), d_C.get(), (size_t)kVerifyM * kVerifyN * sizeof(float), cudaMemcpyDeviceToHost));
        AccuracyResult acc = measure_accuracy(h_C_ref.data(), h_C_out.data(), kVerifyM, kVerifyN);
        printf("[verify] variant=%s M=%d N=%d K=%d cache=%s  %s  max_abs=%.4e  rmse=%.4e  rel=%.3f%%\n",
               variant.c_str(), kVerifyM, kVerifyN, kVerifyK, cache_hit ? "hit" : "miss",
               acc.pass ? "PASS" : "FAIL", acc.max_abs_err, acc.rmse, acc.real_err_pct);
        return;
    }

    if (kind == VerifyKind::Int8) {
        std::vector<int8_t> A;
        std::vector<int8_t> B;
        std::vector<int8_t> BT((size_t)kVerifyN * kVerifyK);
        std::vector<int32_t> C_ref((size_t)kVerifyM * kVerifyN, 0);
        std::vector<int32_t> C_out((size_t)kVerifyM * kVerifyN, 0);

        A.resize((size_t)kVerifyM * kVerifyK);
        const bool cache_hit = load_verify_cache_int(cache_path, VerifyKind::Int8, A, BT, C_ref);
        if (!cache_hit) {
            build_int8_inputs(A, BT, B, C_ref);
            save_verify_cache_int(cache_path, VerifyKind::Int8, A, BT, C_ref);
        }

        DeviceBuffer<int8_t> d_A((size_t)kVerifyM * kVerifyK);
        DeviceBuffer<int8_t> d_BT((size_t)kVerifyN * kVerifyK);
        DeviceBuffer<int32_t> d_C((size_t)kVerifyM * kVerifyN);
        CHECK_CUDA(cudaMemcpy(d_A.get(), A.data(), A.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_BT.get(), BT.data(), BT.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_C.get(), 0, (size_t)kVerifyM * kVerifyN * sizeof(int32_t)));

        solver.run(reinterpret_cast<const half*>(d_A.get()),
                   reinterpret_cast<const half*>(d_BT.get()),
                   reinterpret_cast<float*>(d_C.get()));

        CHECK_CUDA(cudaMemcpy(C_out.data(), d_C.get(), (size_t)kVerifyM * kVerifyN * sizeof(int32_t), cudaMemcpyDeviceToHost));
        int32_t max_abs = 0;
        double rel_pct = 0.0;
        const bool pass = compare_int32_exact(C_ref, C_out, &max_abs, &rel_pct);
        printf("[verify] variant=%s M=%d N=%d K=%d cache=%s  %s  max_abs=%d  rel=%.3f%%\n",
               variant.c_str(), kVerifyM, kVerifyN, kVerifyK, cache_hit ? "hit" : "miss",
               pass ? "PASS" : "FAIL", (int)max_abs, rel_pct);
        return;
    }

    std::vector<int8_t> A_packed((size_t)kVerifyM * (kVerifyK / 2));
    std::vector<int8_t> BT_packed((size_t)kVerifyN * (kVerifyK / 2));
    std::vector<int8_t> A_unpacked;
    std::vector<int8_t> B_unpacked;
    std::vector<int32_t> C_ref((size_t)kVerifyM * kVerifyN, 0);
    std::vector<int32_t> C_out((size_t)kVerifyM * kVerifyN, 0);

    const bool cache_hit = load_verify_cache_int(cache_path, VerifyKind::Int4, A_packed, BT_packed, C_ref);
    if (!cache_hit) {
        build_int4_inputs(A_packed, BT_packed, A_unpacked, B_unpacked, C_ref);
        save_verify_cache_int(cache_path, VerifyKind::Int4, A_packed, BT_packed, C_ref);
    }

    DeviceBuffer<int8_t> d_A((size_t)kVerifyM * (kVerifyK / 2));
    DeviceBuffer<int8_t> d_BT((size_t)kVerifyN * (kVerifyK / 2));
    DeviceBuffer<int32_t> d_C((size_t)kVerifyM * kVerifyN);
    CHECK_CUDA(cudaMemcpy(d_A.get(), A_packed.data(), A_packed.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_BT.get(), BT_packed.data(), BT_packed.size() * sizeof(int8_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C.get(), 0, (size_t)kVerifyM * kVerifyN * sizeof(int32_t)));

    solver.run(reinterpret_cast<const half*>(d_A.get()),
               reinterpret_cast<const half*>(d_BT.get()),
               reinterpret_cast<float*>(d_C.get()));

    CHECK_CUDA(cudaMemcpy(C_out.data(), d_C.get(), (size_t)kVerifyM * kVerifyN * sizeof(int32_t), cudaMemcpyDeviceToHost));
    int32_t max_abs = 0;
    double rel_pct = 0.0;
    const bool pass = compare_int32_exact(C_ref, C_out, &max_abs, &rel_pct);
    printf("[verify] variant=%s M=%d N=%d K=%d cache=%s  %s  max_abs=%d  rel=%.3f%%\n",
           variant.c_str(), kVerifyM, kVerifyN, kVerifyK, cache_hit ? "hit" : "miss",
           pass ? "PASS" : "FAIL", (int)max_abs, rel_pct);
}
