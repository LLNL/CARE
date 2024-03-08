######################################################################################
# Copyright (c) 2024-24, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

#[=======================================================================[.rst:

Findcub
-------

Finds the CUB package.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following imported target, if found:

``cub::cub``
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
             ${CUB_DIR}/include/
             ${CUDA_TOOLKIT_ROOT_DIR}
             ${PROJECT_SOURCE_DIR}/tpl/cub
          NO_DEFAULT_PATH)

mark_as_advanced(CUB_INCLUDE_DIR)

find_package_handle_standard_args(cub
                                  REQUIRED_VARS
                                  CUB_INCLUDE_DIR)

if(CUB_FOUND AND NOT TARGET cub::cub)
   add_library(cub::cub INTERFACE IMPORTED)
   set_target_properties(cub::cub PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${CUB_INCLUDE_DIR}")
endif()
