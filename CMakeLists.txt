##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CARE
# contributors. See the CARE LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

##########################################################################
# Set up CMake
##########################################################################

cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0096 OLD) # Required when building Umpire submodule

##########################################################################
# Set up project
##########################################################################

project(CARE
        LANGUAGES C CXX
        VERSION 2026.07.0)

include(${PROJECT_SOURCE_DIR}/cmake/Setup.cmake)

##########################################################################
# Set up library
##########################################################################

add_subdirectory(src)

##########################################################################
# Set up tests
##########################################################################

if(CARE_ENABLE_TESTS)
   add_subdirectory(test)
endif()

##########################################################################
# Set up benchmarks
##########################################################################

if(CARE_ENABLE_BENCHMARKS)
   add_subdirectory(benchmarks)
endif()

##########################################################################
# Set up documentation
##########################################################################

if(CARE_ENABLE_DOCS)
   add_subdirectory(docs)
endif()

##########################################################################
# Set up examples
##########################################################################

if(CARE_ENABLE_EXAMPLES)
   add_subdirectory(examples)
endif()

##########################################################################
# Set up reproducers
##########################################################################

if(CARE_ENABLE_REPRODUCERS)
   add_subdirectory(reproducers)
endif()
