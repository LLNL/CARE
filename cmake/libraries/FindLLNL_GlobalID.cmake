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
#    LLNL_GlobalID_FOUND - If LLNL_GlobalID was found
#    LLNL_GlobalID_INCLUDE_DIRS - The LLNL_GlobalID include directories
#    LLNL_GlobalID::LLNL_GlobalID - The LLNL_GlobalID imported target
#
###############################################################################

find_path(LLNL_GlobalID_INCLUDE_DIR
          NAMES LLNL_GlobalID.h
          PATHS ${LLNL_GLOBALID_DIR}/include/ ${LLNL_GLOBALID_DIR}
          NO_DEFAULT_PATH
          NO_CMAKE_ENVIRONMENT_PATH
          NO_CMAKE_PATH
          NO_SYSTEM_ENVIRONMENT_PATH
          NO_CMAKE_SYSTEM_PATH)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(LLNL_GlobalID
                                  FOUND_VAR LLNL_GlobalID_FOUND
                                  REQUIRED_VARS
                                    LLNL_GlobalID_LIBRARY
                                    LLNL_GlobalID_INCLUDE_DIR)

if (LLNL_GlobalID_FOUND)
   set(LLNL_GlobalID_INCLUDE_DIRS ${LLNL_GlobalID_INCLUDE_DIR})

   if (NOT TARGET LLNL_GlobalID::LLNL_GlobalID)
      blt_import_library(NAME LLNL_GlobalID::LLNL_GlobalID
                         INCLUDES ${LLNL_GLOBALID_INCLUDE_DIR}
                         TREAT_INCLUDES_AS_SYSTEM ON)
   endif()
endif()

