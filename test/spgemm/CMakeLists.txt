# CMakeLists.txt for unittest
 
# required cmakeversion                                                                                                                                                                                   
cmake_minimum_required(VERSION 2.8)

# packages                                                                                                                                                                                                 
find_package(CUDA)

set( PROJ_NAME      "test" )
set( PROJ_PATH      ${CMAKE_SOURCE_DIR})
set( PROJ_OUT_PATH  ${CMAKE_BINARY_DIR})
set( PROJ_SOURCES   "main.cu" )
set( PROJ_HEADERS   "" )
set( PROJ_LIBRARIES "" )
set( PROJ_INCLUDES  "../../src" "../../ext/moderngpu/include" "../../ext/cub" "../../ext/cudpp/include" )
#set( PROJ_INCLUDES  "../../src" "../../ext/moderngpu/include" "../../ext/cub" "/home/ctcyang/Benchmark_SpGEMM_using_CSR/SpGEMM_cuda" "/home/ctcyang/samples/common/inc" "../../ext/cudpp/include" )
set( mgpu_SRC_DIR "../../ext/moderngpu/src")
set( mgpu_SRC_FILES ${mgpu_SRC_DIR}/mgpucontext.cu ${mgpu_SRC_DIR}/mgpuutil.cpp)
#set( bhsparse_SRC_DIR "/home/ctcyang/Benchmark_SpGEMM_using_CSR/SpGEMM_cuda" )
#set( bhsparse_SRC_FILES ${bhsparse_SRC_DIR}/bhsparse_cuda.cu ${bhsparse_SRC_DIR}/bhsparse.cpp )
set( mt_SRC_FILES "../../src/mt19937ar.cpp" )

#set( CUDA_CURAND_LIBRARY "$ENV{CUDA_HOME}/lib64/libcurand.so" )
#set( CUDA_CUBLAS_LIBRARY "$ENV{CUDA_HOME}/lib64/libcublas.so" )
set( CUDA_CUSPARSE_LIBRARY "$ENV{CUDA_HOME}/lib64/libcusparse.so" )
set( CUDA_CUDPP_LIBRARY "/home/ctcyang/apspie/ext/cudpp/lib/libcudpp.a" )

#FILE( GLOB_RECURSE PROJ_SOURCES ../../src/mt19937ar.cpp ../../src/*.cpp )
#FILE( GLOB_RECURSE PROJ_LIBRARIES ../../ext/cublas1.1/*.cu )
FILE( GLOB_RECURSE PROJ_HEADERS ../../src/*.cuh ../../src/*.h )

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35;-lineinfo;-Xptxas=-v,-dlcm=ca)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-arch=sm_35;-lineinfo;-Xptxas=-dlcm=ca)

# needed for cudamalloc
set(CMAKE_CXX_FLAGS "-fpermissive")
set(CMAKE_EXE_LINKER_FLAGS "-lpthread")
#set(CMAKE_EXE_LINKER_FLAGS "-Xlinker --verbose")

# ptx generation
#include_directories( ${PROJ_INCLUDES} ${PROJ_HEADERS} )
#cuda_compile_ptx( cuda_ptx_files main.cu )
#add_custom_target( ptx ALL DEPENDS ${cuda_ptx_files} ${PROJ_SOURCES} ${mgpu_SRC_FILES} )

# standard compilation
include_directories( ${PROJ_INCLUDES} )
cuda_add_library( lib ${PROJ_LIBRARIES} ${PROJ_HEADERS} )
cuda_add_executable( ${PROJ_NAME} ${PROJ_SOURCES} ${mgpu_SRC_FILES} ${mt_SRC_FILES} )
target_link_libraries( ${PROJ_NAME} lib ${CUDA_CUSPARSE_LIBRARY} ${CUDA_CUBLAS_LIBRARY} ${CUDA_CUDPP_LIBRARY} )
