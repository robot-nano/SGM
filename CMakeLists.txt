cmake_minimum_required(VERSION 3.0)
project(SGM)

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/include)


if (USEGPU)
    set(DEVICE_COMPILER "nvcc")
    set(CUDA_COMPILER "${DEVICE_COMPILER}" CACHE STRING "Select the device compiler")
    set(CUDA_ARCH "20;30;35;37;50;52;60;61;62" CACHE STRING "Set GPU architecture (semicolon separated list, e.g. '-DCUDA_ARCH=20;35;60')")

    find_package(CUDA 10.2 REQUIRED)
    include_directories(${OpenCV_INCLUDE_DIRS})

    CUDA_ADD_LIBRARY(sgm_lib SHARED
            src/census_transform.cu
            src/compute_cost.cu
            src/compute_disp.cu
            src/sgm.cu
            src/cost_aggregate.cu
            )
    target_link_libraries(sgm_lib ${CUDA_LIBRARIES})
    CUDA_ADD_EXECUTABLE(sgm main.cu)
    target_link_libraries(sgm ${CUDA_LIBRARIES} ${OpenCV_LIBS} sgm_lib)
else()
    add_executable(sgm
            main.cu
            src/census_transform.cu)
    target_link_libraries(sgm ${OpenCV_LIBS})
endif ()

CUDA_ADD_EXECUTABLE(temp temp.cu)