##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

#[=======================================================================[.rst:

FindCUB
-------

Finds the CUB package.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported target, if found:

``CUB::CUB``
  The CUB package

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variable:

``CUB_FOUND``
  True if the CUB package is found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variable may also be set:

``CUB_INCLUDE_DIR``
  The directory containing CUB headers.

#]=======================================================================]

include(FindPackageHandleStandardArgs)

find_path(CUB_INCLUDE_DIR
          NAMES cub/cub.cuh
          PATHS
             ${CUB_DIR}/
             ${CUB_DIR}/include/
          NO_DEFAULT_PATH)

mark_as_advanced(CUB_INCLUDE_DIR)

find_package_handle_standard_args(CUB
                                  REQUIRED_VARS
                                  CUB_INCLUDE_DIR)

if(CUB_FOUND AND NOT TARGET CUB::CUB)
   add_library(CUB::CUB INTERFACE IMPORTED)
   set_target_properties(CUB::CUB PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUB_INCLUDE_DIR}")
endif()
