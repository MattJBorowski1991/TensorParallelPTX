cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(TensorParallelPTX LANGUAGES CXX CUDA)

# ── CUDA Configuration ────────────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target sm_89 (L4 GPU)
set(CMAKE_CUDA_ARCHITECTURES 89)

# Verbose PTX generation for profiling + optimization feedback
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")

# ── C++ Configuration ────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(CUDA REQUIRED)

# Optional: NCCL for multi-GPU (will be used for tensor parallelism later)
# Uncomment and configure when adding multi-GPU support:
# find_package(NCCL REQUIRED)

# ── Include directories ──────────────────────────────────────────────────────
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# ── Source Files ────────────────────────────────────────────────────────────
# Host-side implementations
set(HOST_SOURCES
    src/main.cu
    src/solver.cu
    src/data.cu
)

# Baseline: fp16 WMMA (double-buffered)
set(BASELINE_KERNEL_SOURCES
    kernels/fp16_wmma.cu
)

# ── Build Target: TensorParallelPTX (baseline, single GPU) ──────────────────
add_executable(tensor_parallel_ptx ${HOST_SOURCES} ${BASELINE_KERNEL_SOURCES})

# Link CUDA libraries
target_link_libraries(tensor_parallel_ptx
    PRIVATE
    cudart
    cublas
)

# Optional: Link NCCL when multi-GPU support is added
# target_link_libraries(tensor_parallel_ptx PRIVATE nccl)

# Set output directory
set_target_properties(tensor_parallel_ptx PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── Build Information ──────────────────────────────────────────────────────
message(STATUS "========== TensorParallelPTX Build Configuration ==========")
message(STATUS "CUDA Architecture: sm_89 (NVIDIA L4)")
message(STATUS "CUDA Compute Capability: 8.9")
message(STATUS "C++ Standard: C++17")
message(STATUS "CUDA Standard: CUDA 17")
message(STATUS "Kernel Variant: Baseline (fp16_wmma, double-buffered)")
message(STATUS "===========================================================")
