######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

###############################################################################
#
# Setup BASIL
# This file defines:
#  BASIL_FOUND - If BASIL was found
#  BASIL_INCLUDE_DIRS - The BASIL include directories
#  BASIL_LIBRARY - The BASIL library

# first Check for BASIL_DIR

if(NOT BASIL_DIR)
    MESSAGE(FATAL_ERROR "Could not find BASIL. BASIL support needs explicit BASIL_DIR")
endif()

set(BASIL_LIBRARY basil)

include (${BASIL_DIR}/share/basil/cmake/basil.cmake)

set (BASIL_INCLUDE_DIR ${BASIL_DIR}/include)

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set BASIL_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(BASIL  DEFAULT_MSG
                                  BASIL_INCLUDE_DIR
                                  BASIL_LIBRARY )
