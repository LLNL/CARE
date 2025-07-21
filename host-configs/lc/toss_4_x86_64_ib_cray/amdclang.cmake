##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(ROCM_VER "6.4.1" CACHE STRING "")
set(ROCM_PATH "/usr/tce/packages/rocmcc/rocmcc-${ROCM_VER}-magic" CACHE PATH "")
set(COMPILER_BASE "/usr/tce/packages/rocmcc/rocmcc-${ROCM_VER}-magic" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/amdclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/amdclang++" CACHE PATH "")

set(GCC_VER "13.3.1" CACHE STRING "")
set(GCC_PATH "/usr/tce/packages/gcc/gcc-${GCC_VER}" CACHE PATH "")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=${GCC_PATH}" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_PATH}" CACHE STRING "")

set(ENABLE_HIP ON CACHE BOOL "Enable Hip")
set(CMAKE_HIP_ARCHITECTURES "gfx942" CACHE STRING "")
set(AMDGPU_TARGETS "gfx942" CACHE STRING "")
set(GPU_TARGETS "gfx942" CACHE STRING "")

# Used by the DeviceASAN example
set(CLANG_VER "19" CACHE STRING "")
set(CARE_ASAN_RPATH_FLAG "-Wl,-rpath,/opt/rocm-${ROCM_VER}/lib/asan/:/opt/rocm-${ROCM_VER}/llvm/lib/asan:/opt/rocm-${ROCM_VER}/lib/llvm/lib/clang/${CLANG_VER}/lib/linux" CACHE STRING "")
