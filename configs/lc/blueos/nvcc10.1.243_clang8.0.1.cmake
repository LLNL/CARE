set(ENABLE_CUDA ON CACHE BOOL "Enable CUDA")
set(CUDA_TOOLKIT_ROOT_DIR "/usr/tce/packages/cuda/cuda-10.1.243" CACHE PATH "Path to CUDA")
set(NVTX_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "Path to NVTX")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-8.0.1/bin/clang++" CACHE FILEPATH "Path to clang++")
