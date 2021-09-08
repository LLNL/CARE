######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

###############################################################################
#
# Setup NVTX
#
# This file defines:
#    NVTX_FOUND - If NVTX was found
#    NVTX_LIBRARIES - The NVTX library
#    NVTX_INCLUDE_DIRS - The NVTX include directories
#    NVTX - The NVTX imported target
#
###############################################################################

find_path(NVTX_INCLUDE_DIR
          NAMES nvToolsExt.h
          PATHS ${NVTX_DIR}/include
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

find_library(NVTX_LIBRARY
             NAMES nvToolsExt libnvToolsExt
             PATHS ${NVTX_DIR}/lib
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(NVTX
                                  FOUND_VAR NVTX_FOUND
                                  REQUIRED_VARS
                                     NVTX_LIBRARY
                                     NVTX_INCLUDE_DIR)

if(NVTX_FOUND)
   set(NVTX_LIBRARIES ${NVTX_LIBRARY})
   set(NVTX_INCLUDE_DIRS ${NVTX_INCLUDE_DIR})

   if(NOT TARGET NVTX::NVTX)
      blt_import_library(NAME NVTX::NVTX
                         DEPENDS_ON cuda
                         LIBRARIES ${NVTX_LIBRARY}
                         INCLUDES ${NVTX_INCLUDE_DIR}
                         TREAT_INCLUDES_AS_SYSTEM ON)
   endif()
endif()

mark_as_advanced(NVTX_INCLUDE_DIR
                 NVTX_LIBRARY)

