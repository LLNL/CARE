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
   if (CAMP_DIR)
      message(STATUS "Using external CAMP")
      list(APPEND CMAKE_PREFIX_PATH ${CAMP_DIR})
      find_package(camp CONFIG REQUIRED)
   elseif (camp_DIR)
      message(STATUS "Using external CAMP")
      find_package(camp CONFIG REQUIRED)
   else ()
      message(STATUS "Using CAMP submodule")

      if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/camp/CMakeLists.txt)
         message(FATAL_ERROR "CAMP submodule not initialized. Run 'git submodule update --init' in the git repository or set CAMP_DIR/camp_DIR to use an external build of CAMP.")
      else ()
         add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/camp)
      endif ()
   endif ()
endif ()

################################
# CUB (CUDA required)
################################
if (ENABLE_CUDA)
   if (NOT TARGET CUB)
      if (CUB_DIR)
         message(STATUS "Using external CUB")

         # TODO: Update RAJA to accept a CUB cmake target and write a
         #       FindCUB.cmake find module.
      else ()
         message(STATUS "Using CUB submodule")

         if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/cub/cub/cub.cuh)
            message(FATAL_ERROR "CUB submodule not initialized. Run 'git submodule update --init' in the git repository or set CUB_DIR to use an external build of CUB.")
         else ()
            # TODO: Update RAJA to accept a CUB cmake target and insert
            #       a call to add_library here.
            set (CUB_DIR "${CMAKE_CURRENT_LIST_DIR}/cub" CACHE PATH "Path to CUB" FORCE)
         endif ()
      endif ()
   endif ()
endif ()


################################
# RAJA (required)
################################
if (NOT TARGET raja)
   if (RAJA_DIR)
      message(STATUS "Using external RAJA")
      list(APPEND CMAKE_PREFIX_PATH ${RAJA_DIR})
      find_package(raja CONFIG REQUIRED)
   elseif (raja_DIR)
      message(STATUS "Using external RAJA")
      find_package(raja CONFIG REQUIRED)
   else ()
      message(STATUS "Using RAJA submodule")

      if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/raja/CMakeLists.txt)
         message(FATAL_ERROR "RAJA submodule not initialized. Run 'git submodule update --init' in the git repository or set RAJA_DIR/raja_DIR to use an external build of RAJA.")
      else ()
         # TODO: Remove when these fixes are in RAJA (after v0.11.0).
         # The current patch includes fixes for integrating CAMP and CUB
         # as neighbor submodules.
         file(COPY ${CMAKE_CURRENT_LIST_DIR}/patches/raja/CMakeLists.txt
              DESTINATION ${CMAKE_CURRENT_LIST_DIR}/raja)

         if (ENABLE_CUDA)
            # nvcc dies if compiler flags are duplicated, and RAJA adds duplicates
            set(CMAKE_CUDA_FLAGS "${RAJA_CMAKE_CUDA_FLAGS}")
            set(ENABLE_EXTERNAL_CUB ON CACHE BOOL "Enable use of external CUB in RAJA")
            # CUB_DIR is already set above
         endif ()

         add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/raja)

         if (ENABLE_CUDA)
            # Reset CMAKE_CUDA_FLAGS
            set(CMAKE_CUDA_FLAGS "${CARE_CMAKE_CUDA_FLAGS}")
         endif ()
      endif ()
   endif ()
endif ()

################################
# Umpire (required)
################################
if (NOT TARGET umpire)
   if (UMPIRE_DIR)
      message(STATUS "Using external Umpire")
      list(APPEND CMAKE_PREFIX_PATH ${UMPIRE_DIR})
      find_package(umpire CONFIG REQUIRED)
   elseif (umpire_DIR)
      message(STATUS "Using external Umpire")
      find_package(umpire CONFIG REQUIRED)
   else ()
      message(STATUS "Using Umpire submodule")

      if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/umpire/CMakeLists.txt)
         message(FATAL_ERROR "Umpire submodule not initialized. Run 'git submodule update --init' in the git repository or set UMPIRE_DIR/umpire_DIR to use an external build of Umpire.")
      else ()
         add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/umpire)
      endif ()
   endif ()
endif ()

################################
# CHAI (required)
################################
if (NOT TARGET chai)
   if (CHAI_DIR)
      message(STATUS "Using external CHAI")
      list(APPEND CMAKE_PREFIX_PATH ${CHAI_DIR})
      find_package(chai CONFIG REQUIRED)
   elseif (chai_DIR)
      message(STATUS "Using external CHAI")
      find_package(chai CONFIG REQUIRED)
   else ()
      message(STATUS "Using CHAI submodule")

      if (NOT EXISTS ${CMAKE_CURRENT_LIST_DIR}/chai/CMakeLists.txt)
         message(FATAL_ERROR "CHAI submodule not initialized. Run 'git submodule update --init' in the git repository or set CHAI_DIR/chai_DIR to use an external build of CHAI.")
      else ()
         add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/chai)
      endif ()
   endif ()
endif ()
