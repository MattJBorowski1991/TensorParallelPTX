cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(TensorParallelPTX LANGUAGES CXX CUDA)

# ── Kernel Variant Selection ──────────────────────────────────────────────────
# Usage: cmake -DKERNEL_VARIANT=int8_wmma -S . -B build
# Options: fp16_wmma (default), int8_wmma, int8_ptx_mma_k32, int4_wmma, int4_ptx_mma_k64_x4_x2nontrans_ca
set(KERNEL_VARIANT "fp16_wmma" CACHE STRING "Kernel variant to compile")
set_property(CACHE KERNEL_VARIANT PROPERTY STRINGS 
    "fp16_wmma" "int8_wmma" "int8_ptx_mma_k32" "int4_wmma" "int4_ptx_mma_k64_x4_x2nontrans_ca")

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
# All sources are .cu, so nvcc provides the CUDA toolkit headers itself.
include_directories(${CMAKE_SOURCE_DIR})

# ── Source Files ────────────────────────────────────────────────────────────
# Host-side implementations
set(HOST_SOURCES
    src/main.cu
    src/app/cli.cu
    src/app/dist_context.cu
    src/app/logging.cu
    src/comm/nccl_utils.cu
    src/tp/sharding.cu
    src/tp/summa_runner.cu
    src/tp/oned_runner.cu
    src/verify/verify_runner.cu
    src/data.cu
)

# Kernel variant selection
if(KERNEL_VARIANT STREQUAL "fp16_wmma")
    set(SELECTED_KERNEL kernels/fp16_wmma.cu)
    set(KERNEL_DISPLAY_NAME "FP16 WMMA (m16n16k16)")
elseif(KERNEL_VARIANT STREQUAL "int8_wmma")
    set(SELECTED_KERNEL kernels/int8_wmma.cu)
    set(KERNEL_DISPLAY_NAME "INT8 WMMA (m16n16k16)")
elseif(KERNEL_VARIANT STREQUAL "int8_ptx_mma_k32")
    set(SELECTED_KERNEL kernels/int8_ptx_mma_k32.cu)
    set(KERNEL_DISPLAY_NAME "INT8 PTX (m16n8k32)")
elseif(KERNEL_VARIANT STREQUAL "int4_wmma")
    set(SELECTED_KERNEL kernels/int4_wmma.cu)
    set(KERNEL_DISPLAY_NAME "INT4 WMMA (m8n8k32, packed)")
elseif(KERNEL_VARIANT STREQUAL "int4_ptx_mma_k64_x4_x2nontrans_ca")
    set(SELECTED_KERNEL kernels/int4_ptx_mma_k64_x4_x2nontrans_ca.cu)
    set(KERNEL_DISPLAY_NAME "INT4 PTX (m16n8k64, packed)")
else()
    message(FATAL_ERROR "Unknown KERNEL_VARIANT: ${KERNEL_VARIANT}")
endif()

# ── Build Target ──────────────────────────────────────────────────────────────
add_executable(tensor_parallel_ptx ${HOST_SOURCES} ${SELECTED_KERNEL})

target_compile_definitions(tensor_parallel_ptx PRIVATE TP_KERNEL_VARIANT="${KERNEL_VARIANT}")
target_link_libraries(tensor_parallel_ptx PRIVATE cudart cublas)

if (HAVE_NCCL)
    target_include_directories(tensor_parallel_ptx PRIVATE ${NCCL_INCLUDE_DIR})
    target_link_libraries(tensor_parallel_ptx PRIVATE ${NCCL_LIBRARY})
    target_compile_definitions(tensor_parallel_ptx PRIVATE USE_NCCL=1)
endif()

set_target_properties(tensor_parallel_ptx PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# ── Build Information ──────────────────────────────────────────────────────
message(STATUS "========== TensorParallelPTX Build Configuration ==========")
message(STATUS "CUDA Architecture: sm_89 (NVIDIA L4)")
message(STATUS "CUDA Compute Capability: 8.9")
message(STATUS "C++ Standard: C++17")
message(STATUS "CUDA Standard: CUDA 17")
message(STATUS "Kernel Variant: ${KERNEL_DISPLAY_NAME}")
message(STATUS "Build flag: -DKERNEL_VARIANT=${KERNEL_VARIANT}")
message(STATUS "===========================================================")
