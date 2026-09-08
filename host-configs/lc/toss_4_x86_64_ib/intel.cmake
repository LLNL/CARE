##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CARE
# contributors. See the CARE LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(COMPILER_BASE "/usr/tce/packages/intel/intel-2025.2.0-magic" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/bin/icx" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/bin/icpx" CACHE PATH "")

set(GCC_HOME "/usr/tce/packages/gcc/gcc-13.3.1-magic" CACHE PATH "")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "")

set(BLT_EXPORT_THIRDPARTY OFF CACHE BOOL "")

