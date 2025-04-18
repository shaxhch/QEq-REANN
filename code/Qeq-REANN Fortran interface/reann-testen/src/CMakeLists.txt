add_subdirectory(interfaces)

add_executable(test test.f90)

if(BUILD_CUDA)
  SET(CMAKE_PREFIX_PATH     "/home/sxh/libtorchGPU-1.12")   # add to link the libtorch
  set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda-11.3/")
  set(CUDNN_LIBRARY "/usr/local/cuda-11.3/lib64/")
  set(CUDNN_INCLUDE_DIR "/usr/local/cuda-11.3/include/")
  message(STATUS "Building with libtorch ON GPU
    -- libtorch        :    ${CMAKE_PREFIX_PATH}
    -- CUDA  ROOT      :    ${CUDA_TOOLKIT_ROOT_DIR}
    -- CUDNN libraries :    ${CUDNN_LIBRARY}
    -- CUDNN includes  :    ${CUDNN_INCLUDE_DIR};
  ")
else()
  SET(CMAKE_PREFIX_PATH     "/home/sxh/libtorch-1.12")   # add to link the libtorch
  message(STATUS "Building with libtorch ON CPU
    -- Libtorch        :    ${CMAKE_PREFIX_PATH}
  ")
endif()

find_package(Torch REQUIRED)   #to find the torch in the path

target_link_libraries(test
  PRIVATE
    reann-fortran-wrap
    ${TORCH_LIBRARIES}
  )
