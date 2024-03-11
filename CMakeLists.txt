######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

cmake_minimum_required(VERSION 3.4)

project(care)

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

include(cmake/SetupLibraries.cmake)

add_subdirectory(src)

if (ENABLE_TESTS)
   add_subdirectory(test)
endif()

if (ENABLE_BENCHMARKS)
   add_subdirectory(benchmarks)
endif()

if (ENABLE_DOCUMENTATION)
   add_subdirectory(docs)
endif ()

