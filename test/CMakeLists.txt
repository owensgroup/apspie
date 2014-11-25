# CMakeLists.txt for unittest

# required cmakeversion                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

# packages                                                                                                                                                                                                 
find_package(CUDA)

set( PROJ_NAME      "test" )
set( PROJ_PATH      ${CMAKE_SOURCE_DIR})
set( PROJ_OUT_PATH  ${CMAKE_BINARY_DIR})
set( PROJ_SOURCES   "main.cpp" )
set( PROJ_HEADERS   "" )
set( PROJ_LIBRARIES "" )
set( PROJ_INCLUDES  "../inc" )

set( CUDA_CURAND_LIBRARY "$ENV{CUDA_HOME}/lib64/libcurand.so" )
set( CUDA_CUBLAS_LIBRARY "$ENV{CUDA_HOME}/lib64/libcublas.so" )

#FILE( GLOB_RECURSE PROJ_SOURCES ../src/*.cpp )
#FILE( GLOB_RECURSE PROJ_LIBRARIES ../ext/cublas1.1/*.cu )
#FILE( GLOB_RECURSE PROJ_HEADERS ../inc/*.h )

# nvcc flags                                                                                                                                                                                               
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_35,code=sm_35)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_20,code=sm_21)

# needed for cudamalloc
set(CMAKE_CXX_FLAGS "-fpermissive")
set(CMAKE_EXE_LINKER_FLAGS "-lpthread")
#set(CMAKE_EXE_LINKER_FLAGS "-Xlinker --verbose")

include_directories( ${PROJ_INCLUDES} )
cuda_add_library( lib ${PROJ_LIBRARIES} ${PROJ_HEADERS} )
#cuda_compile( ${PROJ_NAME} ${PROJ_SOURCES} )
cuda_add_executable( ${PROJ_NAME} ${PROJ_SOURCES})
target_link_libraries( ${PROJ_NAME} lib ${CUDA_CURAND_LIBRARY} ${CUDA_CUBLAS_LIBRARY} )
