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
#    CUB_DEPENDS - The CUB dependencies
#    cub - An imported target
#
###############################################################################

if (NOT TARGET cub)
   if (CUB_DIR)
      message(STATUS "CARE: Using external CUB")

      set(CUB_PATHS ${CUB_DIR} ${CUB_DIR}/include)
   else ()
      message(STATUS "CARE: Using CUB submodule")

      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/cub/cub/cub.cuh)
         message(FATAL_ERROR "CARE: CUB submodule not initialized. Run 'git submodule update --init' in the git repository or set CUB_DIR to use an external build of CUB.")
      else ()
         set(CUB_PATHS ${PROJECT_SOURCE_DIR}/tpl/cub)
      endif ()
   endif ()

   find_path(CUB_INCLUDE_DIRS
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
                                     DEFAULT_MSG
                                     CUB_INCLUDE_DIRS)

   if (CUB_FOUND)
      set(CUB_DEPENDS cuda)

      blt_import_library(NAME cub
                         INCLUDES ${CUB_INCLUDE_DIRS}
                         DEPENDS_ON ${CUB_DEPENDS}
                         TREAT_INCLUDES_AS_SYSTEM ON)

      message(STATUS "CARE: CUB found at ${CUB_INCLUDE_DIRS}")
   else ()
      message(FATAL_ERROR "CARE: CUB not found. Run 'git submodule update --init' in the git repository or set CUB_DIR to use an external build of CUB.")
   endif ()
endif ()
