######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

###############################################################################
#
# Setup CHAI
# This file defines:
#  CHAI_FOUND - If CHAI was found
#  CHAI_INCLUDE_DIRS - The CHAI include directories
#  CHAI_LIBRARY - The CHAI library

# first Check for CHAI_DIR

if(NOT CHAI_DIR)
    MESSAGE(FATAL_ERROR "Could not find CHAI. CHAI support needs explicit CHAI_DIR")
endif()

set (chai_DIR ${CHAI_DIR})
list(APPEND CMAKE_PREFIX_PATH ${chai_DIR})
find_package(chai REQUIRED)

set(CHAI_LIBRARY chai)
set(CHAI_FOUND ${chai_FOUND})
