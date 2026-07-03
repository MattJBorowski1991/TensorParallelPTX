#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// No trailing ';' after while(0) — it would break `if (...) CHECK_CUDA(x); else ...`
#define CHECK_CUDA(call)\
    do {\
        cudaError_t err = call;\
        if(err != cudaSuccess){\
            fprintf(stderr, "CUDA Error at %s - %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(1);\
        } \
    } while(0)

template<typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(size_t n){ CHECK_CUDA(cudaMalloc(&ptr, n * sizeof(T))); } // explicit = block implicit conversion; fail loudly on OOM
    ~DeviceBuffer() {cudaFree(ptr); }
    DeviceBuffer(const DeviceBuffer&) = delete; // disable the copy constructor (eg DeviceBuffer<float> buf2 = buf1;) = force unique memory ownership
    DeviceBuffer& operator=(const DeviceBuffer&) = delete; // disable copy assignment operator (eg bufA = bufB;)
    DeviceBuffer(DeviceBuffer&& other) noexcept : ptr(other.ptr) {other.ptr = nullptr; }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if(this != &other) { cudaFree(ptr); ptr = other.ptr; other.ptr = nullptr; }
        return *this;
    }
    T* get() const {return ptr; } // allow to use .get() also on const objects
};

template<typename T>
struct PinnedBuffer {
    T* ptr = nullptr;
    explicit PinnedBuffer(size_t n) { CHECK_CUDA(cudaMallocHost(&ptr, n * sizeof(T))); }
    ~PinnedBuffer() { if (ptr) CHECK_CUDA(cudaFreeHost(ptr)); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
    PinnedBuffer(PinnedBuffer&& other) noexcept : ptr(other.ptr) { other.ptr = nullptr; }
    PinnedBuffer& operator=(PinnedBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) CHECK_CUDA(cudaFreeHost(ptr));
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    T* get() const { return ptr; }
};

struct CudaEvent{
    cudaEvent_t ev;
    CudaEvent() { cudaEventCreate(&ev); }
    ~CudaEvent() { cudaEventDestroy(ev); }
    operator cudaEvent_t() const { return ev; }
};

struct CudaStream {
    cudaStream_t s;
    CudaStream() {CHECK_CUDA(cudaStreamCreate(&s)); }
    ~CudaStream() { CHECK_CUDA(cudaStreamDestroy(s)); }
    operator cudaStream_t() const {return s; }
};  