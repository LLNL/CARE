##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

list(PREPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")

################################
# CUB (required for CUDA build)
################################
if(ENABLE_CUDA AND NOT TARGET CUB::CUB)
   if(NOT CUB_DIR)
      set(CUB_DIR "${CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "")
   endif()

   find_package(CUB MODULE)

   if(NOT CUB_FOUND)
      message(FATAL_ERROR "CARE: CUB not found. Please set CUB_DIR to point to an installation of CUB (defaults to CUDA_TOOLKIT_ROOT_DIR).")
   endif()
endif()

################################
# Umpire (required)
################################
if(NOT TARGET umpire)
   care_find_package(NAME umpire TARGETS umpire)

   if(UMPIRE_FOUND)
      message(STATUS "CARE: Using external Umpire")
   else()
      message(STATUS "CARE: Using Umpire submodule")

      if(NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/umpire/CMakeLists.txt)
         message(FATAL_ERROR "CARE: Umpire submodule not initialized. Run 'git submodule update --init' in the git repository or set UMPIRE_DIR to use an external build of Umpire.")
      else()
         set(UMPIRE_ENABLE_PINNED ${ENABLE_PINNED} CACHE BOOL "Enable pinned memory")

         set(UMPIRE_ENABLE_TESTS ${CARE_ENABLE_SUBMODULE_TESTS} CACHE BOOL "Enable Umpire tests")
         set(UMPIRE_ENABLE_EXAMPLES ${CARE_ENABLE_SUBMODULE_EXAMPLES} CACHE BOOL "Enable Umpire examples")
         set(UMPIRE_ENABLE_DOCS ${CARE_ENABLE_SUBMODULE_DOCS} CACHE BOOL "Enable Umpire documentation")

         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/umpire)
      endif()
   endif()
endif()

################################
# RAJA (required)
################################
if(NOT TARGET RAJA)
   care_find_package(NAME raja TARGETS RAJA)

   if(RAJA_FOUND)
      message(STATUS "CARE: Using external RAJA")
   else()
      message(STATUS "CARE: Using RAJA submodule")

      if(NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/raja/CMakeLists.txt)
         message(FATAL_ERROR "CARE: RAJA submodule not initialized. Run 'git submodule update --init' in the git repository or set RAJA_DIR to use an external build of RAJA.")
      else()
         set(RAJA_ENABLE_TESTS ${CARE_ENABLE_SUBMODULE_TESTS} CACHE BOOL "Enable RAJA tests")
         set(RAJA_ENABLE_BENCHMARKS ${CARE_ENABLE_SUBMODULE_BENCHMARKS} CACHE BOOL "Enable RAJA benchmarks")
         set(RAJA_ENABLE_EXAMPLES ${CARE_ENABLE_SUBMODULE_EXAMPLES} CACHE BOOL "Enable RAJA examples")
         set(RAJA_ENABLE_EXERCISES ${CARE_ENABLE_SUBMODULE_EXERCISES} CACHE BOOL "Enable RAJA exercises")

         if(ENABLE_CUDA)
            # nvcc dies if compiler flags are duplicated, and RAJA adds duplicates
            set(CMAKE_CUDA_FLAGS "${RAJA_CMAKE_CUDA_FLAGS}")

            # Use external CUB
            set(RAJA_ENABLE_EXTERNAL_CUB ON CACHE BOOL "Use external CUB in RAJA")
         endif()

         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/raja)

         if(ENABLE_CUDA)
            # Reset CMAKE_CUDA_FLAGS
            set(CMAKE_CUDA_FLAGS "${CARE_CMAKE_CUDA_FLAGS}")
         endif()
      endif()
   endif()
endif()

################################
# CHAI (required)
################################
if(NOT TARGET chai)
   care_find_package(NAME chai TARGETS chai)

   if(CHAI_FOUND)
      message(STATUS "CARE: Using external CHAI")
   else()
      message(STATUS "CARE: Using CHAI submodule")

      if(NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/chai/CMakeLists.txt)
         message(FATAL_ERROR "CARE: CHAI submodule not initialized. Run 'git submodule update --init' in the git repository or set CHAI_DIR to use an external build of CHAI.")
      else()
         set(CHAI_ENABLE_PINNED ${ENABLE_PINNED} CACHE BOOL "Enable pinned memory support in CHAI")

         set(CHAI_ENABLE_TESTS ${CARE_ENABLE_SUBMODULE_TESTS} CACHE BOOL "Enable CHAI tests")
         set(CHAI_ENABLE_BENCHMARKS ${CARE_ENABLE_SUBMODULE_BENCHMARKS} CACHE BOOL "Enable CHAI benchmarks")
         set(CHAI_ENABLE_EXAMPLES ${CARE_ENABLE_SUBMODULE_EXAMPLES} CACHE BOOL "Enable CHAI examples")
         set(CHAI_ENABLE_DOCS ${CARE_ENABLE_SUBMODULE_DOCS} CACHE BOOL "Enable CHAI documentation")

         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/chai)
      endif()
   endif()
endif()

################################
# LLNL_GlobalID
################################
if(NOT TARGET llnl_globalid::llnl_globalid)
   care_find_package(NAME llnl_globalid TARGETS llnl_globalid::llnl_globalid)
   set(CARE_HAVE_LLNL_GLOBALID ${LLNL_GLOBALID_FOUND} CACHE BOOL "")

   if(NOT LLNL_GLOBALID_FOUND)
      message(STATUS "CARE: LLNL_GlobalID disabled")
   endif()
endif()

################################
# NVTOOLSEXT
################################
if(ENABLE_CUDA AND NOT CUDAToolkit_FOUND)
   set(CUDAToolkit_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
   find_package(CUDAToolkit)
endif()
