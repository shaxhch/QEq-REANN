add_subdirectory(interfaces)

add_executable(test test.f90)

if(BUILD_CUDA)
  SET(CMAKE_PREFIX_PATH     "/home/shaxh/apps/libtorchGPU-1.12")   # add to link the libtorch
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
  SET(CMAKE_PREFIX_PATH     "/home/shaxh/apps/libtorch2.0.1")   # add to link the libtorch
  message(STATUS "Building with libtorch ON CPU
    -- Libtorch        :    ${CMAKE_PREFIX_PATH}
  ")
endif()

find_package(Torch REQUIRED)   #to find the torch in the path
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_link_libraries(test
  PRIVATE
    Chreann-fortran-wrap
    ${TORCH_LIBRARIES}
    OpenMP::OpenMP_CXX
  )
