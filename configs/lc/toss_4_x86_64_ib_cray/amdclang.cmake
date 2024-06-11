##############################################################################
# Copyright (c) 2020-24, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(COMPILER_BASE "/usr/tce/packages/rocmcc/rocmcc-6.1.0-magic" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/amdclang" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/amdclang++" CACHE PATH "")

set(ENABLE_HIP ON CACHE BOOL "Enable Hip")
set(ROCM_PATH "/usr/tce/packages/rocmcc/rocmcc-6.1.0-magic" CACHE PATH "")
set(CMAKE_HIP_ARCHITECTURES "gfx942:xnack+" CACHE STRING "")
set(AMDGPU_TARGETS "gfx942:xnack+" CACHE STRING "")

set(CARE_ENABLE_IMPLICIT_CONVERSIONS OFF CACHE BOOL "Enable implicit conversions")
