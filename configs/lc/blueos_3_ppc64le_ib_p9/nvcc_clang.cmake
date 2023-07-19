set(COMPILER_BASE "/usr/tce/packages/clang/clang-ibm-14.0.5" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/clang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/clang++" CACHE PATH "")

set(GCC_HOME "/usr/tce/packages/gcc/gcc-8.3.1" CACHE PATH "")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")

set(BLT_CMAKE_IMPLICIT_LINK_DIRECTORIES_EXCLUDE "/usr/tce/packages/gcc/gcc-4.9.3/lib64/gcc/powerpc64le-unknown-linux-gnu/4.9.3;/usr/tce/packages/gcc/gcc-4.9.3/lib64" CACHE STRING "")

set(BLT_EXPORT_THIRDPARTY OFF CACHE BOOL "")

set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA")
set(CMAKE_CUDA_ARCHITECTURES "70" CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-Xcompiler=--gcc-toolchain=${GCC_HOME}" CACHE STRING "")
#TODO: Remove CUDA_ARCH and CUDA_CODE
set(CUDA_ARCH "sm_70" CACHE STRING "Set the CUDA virtual architecture")
set(CUDA_CODE "compute_70" CACHE STRING "Set the CUDA actual architecture")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-11.7.0" CACHE PATH "Path to CUDA")
set(NVTOOLSEXT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "Path to NVTOOLSEXT")

