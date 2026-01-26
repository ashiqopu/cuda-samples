cmake_minimum_required(VERSION 3.20)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules")

project(ncclAllGatherTest LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CUDA_ARCHITECTURES 75 80 86 87 89 90)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")        # enable cuda-gdb (may significantly affect performance on some targets)
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo") # add line information to all builds for debug tools (exclusive to -G option)
endif()

# Include directories and libraries
include_directories(../../../Common)

# Find NCCL library
find_path(NCCL_INCLUDE_DIR nccl.h
    HINTS ${NCCL_ROOT}/include /usr/include /usr/local/include)

find_library(NCCL_LIBRARY nccl
    HINTS ${NCCL_ROOT}/lib ${NCCL_ROOT}/lib64 /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu)

if(NOT NCCL_INCLUDE_DIR OR NOT NCCL_LIBRARY)
    message(STATUS "NCCL not found. Skipping ncclAllGatherTest.")
    return()
endif()

message(STATUS "Found NCCL: ${NCCL_LIBRARY}")
message(STATUS "NCCL Include: ${NCCL_INCLUDE_DIR}")

# Add target for ncclAllGatherTest
add_executable(ncclAllGatherTest ncclAllGatherTest.cu)

target_compile_options(ncclAllGatherTest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)

target_compile_features(ncclAllGatherTest PRIVATE cxx_std_17 cuda_std_17)

set_target_properties(ncclAllGatherTest PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(ncclAllGatherTest PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${NCCL_INCLUDE_DIR}
)

target_link_libraries(ncclAllGatherTest PUBLIC
    CUDA::cudart
    ${NCCL_LIBRARY}
)
