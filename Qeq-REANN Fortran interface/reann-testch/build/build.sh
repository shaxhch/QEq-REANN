cmake \
 -D BUILD_CUDA=OFF \
 -D CMAKE_Fortran_COMPILER=`which gfortran` \
 -D CMAKE_CXX_COMPILER=`which g++` \
 -D CMAKE_C_COMPILER=`which gcc` \
 ../cmake
