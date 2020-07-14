######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

cmake_minimum_required(VERSION 3.4)

project(care LANGUAGES CXX)

################################
# Build options
################################

# Configuration options
option(ENABLE_MPI "Build MPI support" OFF)
option(ENABLE_CUDA "Build CUDA support" OFF)
option(ENABLE_OPENMP "Build OpenMP support" OFF)
option(ENABLE_HIP "Build HIP support" OFF)

# Advanced configuration options
# TODO: If these are disabled, the code will not compile or run correctly.
#       Fix those issues so that these options are actually configurable.
option(ENABLE_PICK "Enable pick and set methods on ManagedArrays" ON)
option(ENABLE_PINNED "Enable pinned memory space (used for scans)" ON)

# Extra components
option(ENABLE_TESTS "Build tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_DOCUMENTATION "Build documentation" ON)
option(ENABLE_EXAMPLES "Build examples" ON)

# TODO: Not currently used, but allows ENABLE_EXERCISES to be set properly
#       for submodules. Investigate alternatives.
option(ENABLE_EXERCISES "Build exercises" OFF)

# Extra submodule components
option(ENABLE_SUBMODULE_TESTS "Build submodule tests" OFF)
option(ENABLE_SUBMODULE_BENCHMARKS "Build submodule benchmarks" OFF)
option(ENABLE_SUBMODULE_DOCUMENTATION "Build submodule documentation" OFF)
option(ENABLE_SUBMODULE_EXAMPLES "Build submodule examples" OFF)
option(ENABLE_SUBMODULE_EXERCISES "Build submodule exercises" OFF)

# Required dependencies (by default uses submodules)
# The upper case variables simply point to the root install location.
# The lower case variables point to the cmake config file (if applicable).
# Whichever form is most convenient can be used to specify the dependencies.
set(BLT_SOURCE_DIR "" CACHE PATH "Path to external BLT")

set(CAMP_DIR "" CACHE PATH "Path to installed CAMP")
set(camp_DIR "" CACHE PATH "Path to campConfig.cmake")

set(UMPIRE_DIR "" CACHE PATH "Path to installed Umpire")
set(umpire_DIR "" CACHE PATH "Path to umpire-config.cmake")

set(RAJA_DIR "" CACHE PATH "Path to installed RAJA")
set(raja_DIR "" CACHE PATH "Path to external rajaConfig.cmake or raja-config.cmake")

set(CHAI_DIR "" CACHE PATH "Path to installed CHAI")
set(chai_DIR "" CACHE PATH "Path to chai-config.cmake")

# Optional dependencies without submodules (not open source)
set(LLNL_GLOBALID_DIR "" CACHE PATH "Path to external LLNL_GlobalID")

# CUDA specific options
if (ENABLE_CUDA)
   # CUDA configuration
   set(CUDA_TOOLKIT_ROOT_DIR "" CACHE PATH "Set the CUDA toolkit path")
   set(CUDA_ARCH "sm_35" CACHE STRING "Set the CUDA virtual architecture")
   set(CUDA_CODE "compute_35" CACHE STRING "Set the CUDA actual architecture")
   option(CUDA_SEPARABLE_COMPILATION "Enable CUDA separable compilation" ON)
   option(CUDA_LINK_WITH_NVCC "Enable linking with NVCC" OFF)

   # CUDA required dependencies
   # TODO: Fix RAJA to accept a CUB cmake target
   set(CUB_DIR "" CACHE PATH "Path to external CUB (top level of the install or source directory)")

   # CUDA optional dependencies without submodules (not open source)
   set(NVTX_DIR "" CACHE PATH "Path to external NVTX")
   set(BASIL_DIR "" CACHE PATH "Path to external BASIL")
endif ()

################################
# BLT
################################

# TODO: Setting BLT_DOCS_TARGET_NAME will not be necessary after RAJA is
#       updated past v0.11.0. BLT and RAJA both define a "docs" target,
#       which cmake does not allow. The name of the BLT "docs" target is
#       changed here to "blt-docs". The next version of RAJA renames its
#       "docs" target to "raja-docs". A future update of BLT may also change
#       the default name in BLT to something like "blt-docs", in which case
#       this would no longer be necessary, either.
if (ENABLE_SUBMODULE_DOCUMENTATION)
    set(BLT_DOCS_TARGET_NAME "blt-docs" CACHE STRING "Name of the master documentation generation target")
endif ()

set(BLT_CXX_STD "c++11" CACHE STRING "Set the c++ standard to use")

if (BLT_SOURCE_DIR)
    message(STATUS "Using external BLT")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
    message(STATUS "Using BLT submodule")

    set(BLT_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/blt")

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
       message(FATAL_ERROR "BLT submodule is not initialized. Run `git submodule update --init` in git repository or set BLT_SOURCE_DIR to external BLT.")
    endif()
endif()

include(${BLT_SOURCE_DIR}/SetupBLT.cmake)

################################
# Set up compiler and flags
################################
if (ENABLE_CUDA)
   set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
   set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "")

   # nvcc dies if compiler flags are duplicated, and RAJA adds all these compiler flags
   set(RAJA_CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -arch ${CUDA_CODE} -code ${CUDA_ARCH}")
   set(CARE_CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()

### needed if we use hip_runtime as a dependency instead of just hip
### this is kind of a hack
#if (ENABLE_HIP)
#   set(HIP_ROOT_DIR "/opt/rocm-3.5.0/hip" CACHE PATH "")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --hip-device-lib-path=/opt/rocm-3.5.0/lib  -I/opt/rocm-3.5.0/llvm/lib/clang/11.0.0 -x hip")
#   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --hip-device-lib-path=/opt/rocm-3.5.0/lib -I/opt/rocm-3.5.0/llvm/lib/clang/11.0.0 -x hip")
#endif()

################################
# Set up dependencies
################################
add_subdirectory(tpl)
include(cmake/SetupLibraries.cmake)

################################
# Build CARE library
################################
add_subdirectory(src)

################################
# Build CARE extras
################################
if (ENABLE_TESTS)
   add_subdirectory(test)
endif()

if (ENABLE_BENCHMARKS)
   add_subdirectory(benchmarks)
endif()

if (ENABLE_DOCUMENTATION)
   add_subdirectory(docs)
endif ()

if (ENABLE_EXAMPLES)
   add_subdirectory(examples)
endif ()

