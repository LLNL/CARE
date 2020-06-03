######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

###############################################################################
#
# Setup UMPIRE
# This file defines:
#  UMPIRE_FOUND - If UMPIRE was found
#  UMPIRE_INCLUDE_DIRS - The UMPIRE include directories
#  UMPIRE_LIBRARY - The UMPIRE library

# first Check for UMPIRE_DIR

if(NOT UMPIRE_DIR)
    MESSAGE(FATAL_ERROR "Could not find UMPIRE. UMPIRE support needs explicit UMPIRE_DIR")
endif()

# umpire's installed cmake target is lower case
set(umpire_DIR ${UMPIRE_DIR})
list(APPEND CMAKE_PREFIX_PATH ${umpire_DIR})
find_package(camp REQUIRED)
find_package(umpire REQUIRED)

set (UMPIRE_FOUND ${umpire_FOUND} CACHE STRING "")
set (UMPIRE_LIBRARY umpire)

