# Things to do before building:
#module load opt
#module load rocm
#setenv HCC_AMDGPU_TARGET gfx900
#setenv HIP_CLANG_PATH /opt/rocm/llvm/bin

set(ENABLE_HIP ON CACHE BOOL "Enable HIP")
set(HIP_CLANG_PATH "/opt/rocm/llvm/bin" CACHE PATH "Path to HIP CLANG")
set(HCC_AMDGPU_TARGET "gfx900" CACHE STRING "Set the AMD actual architecture")

set(CMAKE_CXX_COMPILER "/opt/rocm/llvm/bin/clang++" CACHE FILEPATH "Path to clang++")
set(CMAKE_C_COMPILER "/opt/rocm/llvm/bin/clang" CACHE FILEPATH "Path to clang++")

