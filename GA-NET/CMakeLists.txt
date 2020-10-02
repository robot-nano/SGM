cmake_minimum_required(VERSION 3.0)
project(GACPP)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_PREFIX_PATH "$ENV{CUDA_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "$ENV{CMAKE_PREFIX_PATH}")

set(DEVICE_COMPILER "nvcc")
set(CUDA_COMPILER "${DEVICE_COMPILER}" CACHE STRING "Select the device compiler")

set(CUDA_ARCH "20;30;35;37;50;52;60;61;62" CACHE STRING "Sett GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=20;35;60)")
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
set(LIBS ${LIBS} ${CUDA_LIBRARIES})

# torch
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_subdirectory(libs)