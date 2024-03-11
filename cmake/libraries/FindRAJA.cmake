######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

###############################################################################
#
# Setup RAJA
# This file defines:
#  RAJA_FOUND - If RAJA was found
#  RAJA_INCLUDE_DIRS - The RAJA include directories
#  RAJA_LIBRARY - The RAJA library

# first Check for RAJA_DIR

if(NOT RAJA_DIR)
    MESSAGE(FATAL_ERROR "Could not find RAJA. RAJA support needs explicit RAJA_DIR")
endif()

# give cmake the install prefix to RAJA cmake target
list(APPEND CMAKE_PREFIX_PATH ${RAJA_DIR})
find_package(RAJA REQUIRED)

set(RAJA_LIBRARY RAJA)

