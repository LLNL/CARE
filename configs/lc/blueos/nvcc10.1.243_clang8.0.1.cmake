set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-10.1.243" CACHE PATH "Path to CUDA")
set(CUDA_ARCH "sm_70" CACHE STRING "Set the CUDA virtual architecture")
set(CUDA_CODE "compute_70" CACHE STRING "Set the CUDA actual architecture")

set(NVTOOLSEXT_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "Path to NVTOOLSEXT")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-ibm-11.0.1/bin/clang++" CACHE FILEPATH "Path to clang++")
