##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(ROCM_VER "6.4.0" CACHE STRING "")
set(MPI_VER "8.1.32" CACHE STRING "")

set(COMPILER_BASE "/usr/tce/packages/cray-mpich/cray-mpich-${MPI_VER}-rocmcc-${ROCM_VER}-magic" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/mpiamdclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/mpiamdclang++" CACHE PATH "")

#set(COMPILER_BASE "/usr/tce/packages/rocmcc/rocmcc-${ROCM_VER}-magic" CACHE PATH "")
#set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/amdclang" CACHE PATH "")
#set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/amdclang++" CACHE PATH "")

set(ENABLE_HIP ON CACHE BOOL "Enable Hip")
set(ROCM_PATH "/usr/tce/packages/rocmcc/rocmcc-${ROCM_VER}-magic" CACHE PATH "")
set(CMAKE_HIP_ARCHITECTURES "gfx942:xnack+" CACHE STRING "")
set(AMDGPU_TARGETS "gfx942:xnack+" CACHE STRING "")
set(GPU_TARGETS "gfx942:xnack+" CACHE STRING "")

# Used by the DeviceASAN example
set(CARE_ASAN_RPATH_FLAG "-Wl,-rpath,/opt/rocm-${ROCM_VER}/lib/asan/:/opt/rocm-${ROCM_VER}/llvm/lib/asan:/opt/rocm-${ROCM_VER}/lib/llvm/lib/clang/19/lib/linux" CACHE STRING "")

set(ENABLE_EXAMPLES ON CACHE BOOL "")
set(CARE_ENABLE_EXAMPLES ON CACHE BOOL "")

#set(ENABLE_TEST OFF CACHE BOOL "")
#set(ENABLE_FRUIT OFF CACHE BOOL "")
