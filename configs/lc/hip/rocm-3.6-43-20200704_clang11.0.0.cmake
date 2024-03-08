##############################################################################
# Copyright (c) 2020-24, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

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

# Virtual functions are not supported in HIP device code
set(CARE_ENABLE_MANAGED_PTR OFF CACHE BOOL "Enable aliases, tests, and reproducer for managed_ptr")
