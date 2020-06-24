######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

################################
# CAMP (required)
################################
if (NOT TARGET camp)
   if (camp_DIR)
      message(STATUS "Using external CAMP")

      list(APPEND CMAKE_PREFIX_PATH ${camp_DIR})
      find_package(camp CONFIG REQUIRED)
   else ()
      message(STATUS "Using CAMP submodule")

      if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/camp/CMakeLists.txt)
         message(FATAL_ERROR "CAMP submodule not initialized. Run 'git submodule update --init' in the git repository or set camp_DIR to use an external build of CAMP.")
      else ()
         add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/camp)
      endif ()
   endif ()
endif ()

################################
# Umpire (required)
################################
if (NOT TARGET umpire)
   if (umpire_DIR)
      message(STATUS "Using external Umpire")

      list(APPEND CMAKE_PREFIX_PATH ${umpire_DIR})
      find_package(umpire CONFIG REQUIRED)
   else ()
      message(STATUS "Using Umpire submodule")

      if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/umpire/CMakeLists.txt)
         message(FATAL_ERROR "Umpire submodule not initialized. Run 'git submodule update --init' in the git repository or set umpire_DIR to use an external build of Umpire.")
      else ()
         add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/umpire)
      endif ()
   endif ()
endif ()
