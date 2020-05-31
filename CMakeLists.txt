cmake_minimum_required(VERSION 3.0)
project(SGM)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)


if (USEGPU)
    set(DEVICE_COMPILER "nvcc")
    set(CUDA_COMPILER "${DEVICE_COMPILER}" CACHE STRING "Select the device compiler")
    set(CUDA_ARCH "20;30;35;37;50;52;60;61;62" CACHE STRING "Set GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=20;35;60')")

    find_package(CUDA 10.2 REQUIRED)

    CUDA_ADD_EXECUTABLE(sgm_cpu src/census_transform.cpp)
else()
    add_executable(sgm_cpu
            src/census_transform.cpp)
endif ()

target_link_libraries(sgm_cpu ${OpenCV_LIBS})