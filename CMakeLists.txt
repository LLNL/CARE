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

# Dependencies
set(BLT_SOURCE_DIR "" CACHE PATH "Path to external BLT")
set(camp_DIR "" CACHE PATH "Path to external campConfig.cmake or camp-config.cmake")
set(umpire_DIR "" CACHE PATH "Path to external umpireConfig.cmake or umpire-config.cmake")
set(chai_DIR "" CACHE PATH "Path to external chaiConfig.cmake or chai-config.cmake")

################################
# BLT
################################
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

