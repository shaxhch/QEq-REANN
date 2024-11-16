enable_language(C)
include(FortranCInterface)
FortranCInterface_VERIFY(CXX)

add_library(reann-fortran-wrap SHARED "")
target_sources(reann-fortran-wrap
  PRIVATE
    pes_cpp.cpp
    pes_c.cpp
    pes_f.f90
  )

if(BUILD_CUDA)
  SET(CMAKE_PREFIX_PATH     "/home/shaxh/apps/libtorchGPU-1.12")   # add to link the libtorch
  set(CUDA_TOOLKIT_ROOT_DIR "/opt/cuda-11.3/")
  #set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/")
  set(CUDNN_LIBRARY "/usr/local/cuda-11.3/lib64/")
  set(CUDNN_INCLUDE_DIR "/usr/local/cuda-11.3/include/")
  message(STATUS "Building with libtorch ON GPU
    -- libtorch        :    ${CMAKE_PREFIX_PATH}
    -- CUDA  ROOT      :    ${CUDA_TOOLKIT_ROOT_DIR}
    -- CUDNN libraries :    ${CUDNN_LIBRARY}
    -- CUDNN includes  :    ${CUDNN_INCLUDE_DIR};
  ")
else()
  SET(CMAKE_PREFIX_PATH    "/home/sxh/libtorch-1.12")   # add to link the libtorch
  message(STATUS "Building with libtorch ON CPU
    -- Libtorch        :    ${CMAKE_PREFIX_PATH}
  ")
endif()
find_package(Torch REQUIRED)   #to find the torch in the path
target_link_libraries(reann-fortran-wrap
  PUBLIC
    ${TORCH_LIBRARIES}
  )
