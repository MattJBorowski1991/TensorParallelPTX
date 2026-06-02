cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
project(TensorParallelPTX LANGUAGES CXX CUDA)

# ── Kernel Variant Selection ──────────────────────────────────────────────────
# Usage: cmake -DKERNEL_VARIANT=int8_wmma -S . -B build
# Options: fp16_wmma (default), int8_wmma, int8_ptx, int4_wmma, int4_ptx
set(KERNEL_VARIANT "fp16_wmma" CACHE STRING "Kernel variant to compile")
set_property(CACHE KERNEL_VARIANT PROPERTY STRINGS 
    "fp16_wmma" "int8_wmma" "int8_ptx" "int4_wmma" "int4_ptx")

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

# Kernel variant selection
if(KERNEL_VARIANT STREQUAL "fp16_wmma")
    set(SELECTED_KERNEL kernels/fp16_wmma.cu)
    set(KERNEL_DISPLAY_NAME "FP16 WMMA (m16n16k16)")
elseif(KERNEL_VARIANT STREQUAL "int8_wmma")
    set(SELECTED_KERNEL kernels/int8_wmma.cu)
    set(KERNEL_DISPLAY_NAME "INT8 WMMA (m16n16k16)")
elseif(KERNEL_VARIANT STREQUAL "int8_ptx")
    set(SELECTED_KERNEL kernels/int8_ptx_mma_k32.cu)
    set(KERNEL_DISPLAY_NAME "INT8 PTX (m16n8k32)")
elseif(KERNEL_VARIANT STREQUAL "int4_wmma")
    set(SELECTED_KERNEL kernels/int4_wmma.cu)
    set(KERNEL_DISPLAY_NAME "INT4 WMMA (m8n8k32, packed)")
elseif(KERNEL_VARIANT STREQUAL "int4_ptx")
    set(SELECTED_KERNEL kernels/int4_ptx_mma_k64_x4_x2nontrans_ca.cu)
    set(KERNEL_DISPLAY_NAME "INT4 PTX (m16n8k64, packed)")
else()
    message(FATAL_ERROR "Unknown KERNEL_VARIANT: ${KERNEL_VARIANT}")
endif()

set(BASELINE_KERNEL_SOURCES
    ${SELECTED_KERNEL}
)

# ── Build Target: TensorParallelPTX (baseline with all kernel variants) ──────
add_executable(tensor_parallel_ptx ${HOST_SOURCES} ${BASELINE_KERNEL_SOURCES})

# Link CUDA libraries
foreach(target tensor_parallel_ptx)
    target_compile_definitions(${target}
        PRIVATE
        TP_KERNEL_VARIANT="${KERNEL_VARIANT}"
    )

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
message(STATUS "Kernel Variant: ${KERNEL_DISPLAY_NAME}")
message(STATUS "Build flag: -DKERNEL_VARIANT=${KERNEL_VARIANT}")
message(STATUS "===========================================================")
