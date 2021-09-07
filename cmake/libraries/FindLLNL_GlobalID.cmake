######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

###############################################################################
#
# Setup LLNL_GlobalID
#
# This file defines:
#    LLNL_GLOBALID_FOUND - If LLNL_GlobalID was found
#    LLNL_GLOBALID_INCLUDE_DIRS - The LLNL_GlobalID include directories
#    llnl_globalid - An imported target
#
###############################################################################

if (NOT TARGET llnl_globalid)
   find_path(LLNL_GLOBALID_INCLUDE_DIRS LLNL_GlobalID.h
             PATHS ${LLNL_GLOBALID_DIR}/include/ ${LLNL_GLOBALID_DIR}
             NO_DEFAULT_PATH
             NO_CMAKE_ENVIRONMENT_PATH
             NO_CMAKE_PATH
             NO_SYSTEM_ENVIRONMENT_PATH
             NO_CMAKE_SYSTEM_PATH)

   include(FindPackageHandleStandardArgs)

   find_package_handle_standard_args(LLNL_GLOBALID DEFAULT_MSG
                                     LLNL_GLOBALID_INCLUDE_DIRS)

   if (LLNL_GLOBALID_FOUND)
      blt_import_library(NAME llnl_globalid
                         TREAT_INCLUDES_AS_SYSTEM ON
                         INCLUDES ${LLNL_GLOBALID_INCLUDE_DIRS})
   else ()
      message(FATAL_ERROR "CARE: Unable to find LLNL_GlobalID with given path: ${LLNL_GLOBALID_DIR}!")
   endif ()
endif ()

