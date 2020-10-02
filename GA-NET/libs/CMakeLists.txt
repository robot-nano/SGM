cuda_add_library(
        GANet
        GANet/src/GANet_cuda.cpp
        GANet/src/GANet_kernel.cu
)
target_link_libraries(GANet ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

cuda_add_executable(
        main
        GANet/src/main.cu)
target_link_libraries(main GANet)

#add_library(sync_bn_cpu
#        sync_bn/src/cpu/operator.cpp
#        sync_bn/src/cpu/sync_bn.cpp)
#target_link_libraries(sync_bn_cpu ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

cuda_add_library(
        sync_bn_gpu
        sync_bn/src/gpu/operator.cpp
        sync_bn/src/gpu/sync_bn_cuda.cu
)

target_link_libraries(sync_bn_gpu
        ${TORCH_LIBRARIES} ${PYTHON_LIBRARIES})

add_executable(sync_bn_exe
        sync_bn/src/sync_bn_exe.cpp)
target_link_libraries(sync_bn_exe
        sync_bn_gpu)