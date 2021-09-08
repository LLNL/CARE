######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

###############################################################################
#
# Setup CUB
#
# This file defines:
#    CUB_FOUND - If CUB was found
#    CUB_INCLUDE_DIRS - The CUB include directories
#    cub - An imported target
#
###############################################################################

find_path(CUB_INCLUDE_DIR
          NAMES cub/cub.cuh
          PATHS ${CUB_PATHS}
          NO_DEFAULT_PATH
          NO_PACKAGE_ROOT_PATH
          NO_CMAKE_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(CUB
                                  FOUND_VAR CUB_FOUND
                                  CUB_INCLUDE_DIR)

if(CUB_FOUND)
   set(CUB_INCLUDE_DIRS ${CUB_INCLUDE_DIR})

   if(NOT TARGET CUB::CUB)
      blt_import_library(NAME CUB::CUB
                         DEPENDS_ON cuda
                         INCLUDES ${CUB_INCLUDE_DIR}
                         TREAT_INCLUDES_AS_SYSTEM ON)
   endif()
endif()

mark_as_advanced(CUB_INCLUDE_DIR)

