cmake_minimum_required(VERSION 3.0)
project(sgm)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_FALGS CUDA_NVCC_FLAGS_DEBUG)
set(CMAKE_CXX_FLAGS "-std=c++11")

list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

set(DEVICE_COMPILER "nvcc")
set(CUDA_COMPILER "${DEVICE_COMPILER}" CACHE STRING "Select the device compiler")

set(CUDA_ARCH "20;30;35;37;50;52;60;61;62" CACHE STRING "Set GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=20;35;60')")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(LIBS ${LIBS} ${CUDA_LIBRARIES})

cuda_add_executable(
        sgm
        main.cu disparity.cu debug.cu costs.cu hamming_cost.cu)

target_link_libraries(sgm ${OpenCV_LIBS})
