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

# Extra components
option(ENABLE_TESTS "Build tests" ON)
option(ENABLE_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_DOCUMENTATION "Build documentation" ON)
option(ENABLE_EXAMPLES "Build examples" ON)

# Extra submodule components
option(ENABLE_SUBMODULE_TESTS "Build submodule tests" OFF)
option(ENABLE_SUBMODULE_BENCHMARKS "Build submodule benchmarks" OFF)
option(ENABLE_SUBMODULE_DOCUMENTATION "Build submodule documentation" OFF)
option(ENABLE_SUBMODULE_EXAMPLES "Build submodule examples" OFF)

# Dependencies
set(BLT_SOURCE_DIR "" CACHE PATH "Path to external BLT")

# TODO: When RAJA is updated past v0.11.0, camp_DIR should point to the
#       directory containing campConfig.cmake or camp-config.cmake.
#       Update the variable documentation to reflect that change.
set(camp_DIR "" CACHE PATH "Path to external CAMP (top level of the install or source directory)")

set(umpire_DIR "" CACHE PATH "Path to external umpireConfig.cmake or umpire-config.cmake")
set(chai_DIR "" CACHE PATH "Path to external chaiConfig.cmake or chai-config.cmake")
set(raja_DIR "" CACHE PATH "Path to external rajaConfig.cmake or raja-config.cmake")

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
# Set up compiler flags
################################
if (ENABLE_CUDA)
   set(CMAKE_CUDA_COMPILER "${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc" CACHE PATH "")
   set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE PATH "")
   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda -arch=sm_35")
endif()

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

