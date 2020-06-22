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
set(ENABLE_MPI OFF CACHE BOOL "Build MPI support")
set(ENABLE_CUDA OFF CACHE BOOL "Build CUDA support")
set(ENABLE_HIP Off CACHE BOOL "Enable HIP")
set(ENABLE_OPENMP OFF CACHE BOOL "Build OpenMP support")
set(ENABLE_TESTS ON CACHE BOOL "Build tests")
set(ENABLE_BENCHMARKS ON CACHE BOOL "Build benchmarks")
set(ENABLE_DOCUMENTATION ON CACHE BOOL "Build documentation")
set(ENABLE_EXAMPLES ON CACHE BOOL "Build examples")

################################
# BLT
################################
set(BLT_CXX_STD "c++11" CACHE STRING "Set the c++ standard to use")

if (DEFINED BLT_SOURCE_DIR)
    # Support having a shared BLT outside of the repository if given a BLT_SOURCE_DIR

    if (NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
        message(FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
    endif()
else()
   message(FATAL_ERROR "BLT_SOURCE_DIR not defined")
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

