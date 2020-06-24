######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

################################
# Extra submodule components
################################
set(ENABLE_TESTS "${ENABLE_SUBMODULE_TESTS}")
set(ENABLE_BENCHMARKS "${ENABLE_SUBMODULE_BENCHMARKS}")
set(ENABLE_DOCUMENTATION "${ENABLE_SUBMODULE_DOCUMENTATION}")
set(ENABLE_EXAMPLES "${ENABLE_SUBMODULE_EXAMPLES}")
set(ENABLE_EXERCISES "${ENABLE_SUBMODULE_EXERCISES}")

################################
# CAMP (required)
################################
if (NOT TARGET camp)
   if (camp_DIR)
      message(STATUS "Using external CAMP")

      # TODO: When RAJA is updated past v0.11.0, re-enable this code
      #       since RAJA will be able to use the "camp" cmake target
      #       directly.
      # list(APPEND CMAKE_PREFIX_PATH ${camp_DIR})
      # find_package(camp CONFIG REQUIRED)
   else ()
      message(STATUS "Using CAMP submodule")

      if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/camp/CMakeLists.txt)
         message(FATAL_ERROR "CAMP submodule not initialized. Run 'git submodule update --init' in the git repository or set camp_DIR to use an external build of CAMP.")
      else ()
         # TODO: When RAJA is updated past v0.11.0, re-enable this code
         #       and remove the enabled code since RAJA will be able to
         #       use the "camp" cmake target directly.
         # add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/camp)
         set (camp_DIR "${CMAKE_CURRENT_LIST_DIR}/camp")
      endif ()
   endif ()
endif ()

################################
# RAJA (required)
################################
if (NOT TARGET raja)
   if (raja_DIR)
      message(STATUS "Using external RAJA")

      list(APPEND CMAKE_PREFIX_PATH ${raja_DIR})
      find_package(raja CONFIG REQUIRED)
   else ()
      message(STATUS "Using RAJA submodule")

      if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/raja/CMakeLists.txt)
         message(FATAL_ERROR "RAJA submodule not initialized. Run 'git submodule update --init' in the git repository or set raja_DIR to use an external build of RAJA.")
      else ()
         # TODO: When RAJA is updated past v0.11.0, EXTERNAL_CAMP_SOURCE_DIR
         #       does not need to be set since RAJA will be able to use
         #       the "camp" cmake target directly.
         set(EXTERNAL_CAMP_SOURCE_DIR "${camp_DIR}" CACHE PATH "Build with a specific external camp source repository")
         add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/raja)
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

################################
# CHAI (required)
################################
if (NOT TARGET chai)
   if (chai_DIR)
      message(STATUS "Using external CHAI")

      list(APPEND CMAKE_PREFIX_PATH ${chai_DIR})
      find_package(chai CONFIG REQUIRED)
   else ()
      message(STATUS "Using CHAI submodule")

      if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/chai/CMakeLists.txt)
         message(FATAL_ERROR "CHAI submodule not initialized. Run 'git submodule update --init' in the git repository or set chai_DIR to use an external build of CHAI.")
      else ()
         add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/chai)
      endif ()
   endif ()
endif ()
