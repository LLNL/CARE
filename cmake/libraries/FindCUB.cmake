######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

###############################################################################
#
# Setup CUB
# This file defines:
#  CUB_FOUND - If CUB was found
#  CUB_INCLUDE_DIRS - The CUB include directories

# first Check for CUB_DIR

if(NOT CUB_DIR)
    MESSAGE(FATAL_ERROR "Could not find CUB. CUB support needs explicit CUB_DIR")
endif()

#find includes
find_path( CUB_INCLUDE_DIRS cub/cub.cuh
           PATHS  ${CUB_DIR}/include/
           NO_DEFAULT_PATH
           NO_CMAKE_ENVIRONMENT_PATH
           NO_CMAKE_PATH
           NO_SYSTEM_ENVIRONMENT_PATH
           NO_CMAKE_SYSTEM_PATH)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set CUB_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(CUB  DEFAULT_MSG
                                  CUB_INCLUDE_DIRS )
