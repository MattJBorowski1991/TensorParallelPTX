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

# Optional NCCL for K-split correctness (tp_cols > 1)
find_path(NCCL_INCLUDE_DIR nccl.h)
find_library(NCCL_LIBRARY nccl)
if (NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
    message(STATUS "NCCL found: ${NCCL_LIBRARY}")
    set(HAVE_NCCL ON)
else()
    message(STATUS "NCCL not found: building without USE_NCCL")
    set(HAVE_NCCL OFF)
endif()

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

# ── Build Target: TensorParallelPTX (baseline with all kernel variants) ──────
add_executable(tensor_parallel_ptx ${HOST_SOURCES} ${BASELINE_KERNEL_SOURCES})

# Link CUDA libraries
foreach(target tensor_parallel_ptx)
    target_link_libraries(${target}
        PRIVATE
        cudart
        cublas
    )
    
    if (HAVE_NCCL)
        target_include_directories(${target} PRIVATE ${NCCL_INCLUDE_DIR})
        target_link_libraries(${target} PRIVATE ${NCCL_LIBRARY})
        target_compile_definitions(${target} PRIVATE USE_NCCL=1)
    endif()
    
    # Set output directory
    set_target_properties(${target} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
endforeach()

# ── Build Information ──────────────────────────────────────────────────────
message(STATUS "========== TensorParallelPTX Build Configuration ==========")
message(STATUS "CUDA Architecture: sm_89 (NVIDIA L4)")
message(STATUS "CUDA Compute Capability: 8.9")
message(STATUS "C++ Standard: C++17")
message(STATUS "CUDA Standard: CUDA 17")
message(STATUS "Kernel Variant: Baseline (fp16_wmma, double-buffered)")
message(STATUS "===========================================================")
