######################################################################################
# Copyright 2024 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

################################
# CAMP (required)
################################
if(NOT TARGET camp)
   care_find_package(NAME camp TARGETS camp REQUIRED)

   if(CAMP_FOUND)
      message(STATUS "CARE: Using external CAMP")
   else()
      message(STATUS "CARE: Using CAMP submodule")

      if(NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/camp/CMakeLists.txt)
         message(FATAL_ERROR "CARE: CAMP submodule not initialized. Run 'git submodule update --init' in the git repository or set CAMP_DIR to use an external build of CAMP.")
      else()
         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/camp)
      endif()
   endif()
endif()

################################
# Umpire (required)
################################
if(NOT TARGET umpire)
   care_find_package(NAME umpire TARGETS umpire REQUIRED)

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
   care_find_package(NAME raja TARGETS RAJA REQUIRED)

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
   care_find_package(NAME chai TARGETS chai REQUIRED)

   if(CHAI_FOUND)
      message(STATUS "CARE: Using external CHAI")
   else()
      message(STATUS "CARE: Using CHAI submodule")

      if(NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/chai/CMakeLists.txt)
         message(FATAL_ERROR "CARE: CHAI submodule not initialized. Run 'git submodule update --init' in the git repository or set CHAI_DIR to use an external build of CHAI.")
      else()
         set(CHAI_ENABLE_PICK ${ENABLE_PICK} CACHE BOOL "Enable picks/sets in chai::ManagedArray")
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
# CUB (required for CUDA build)
################################
if(ENABLE_CUDA AND NOT TARGET cub::cub)
   find_package(cub MODULE)

   if(NOT CUB_FOUND)
      message(FATAL_ERROR "CARE: CUB not found. Run 'git submodule update --init' in the git repository or set CUB_DIR to use an external build of CUB or use CUDA 11 or newer.")
   endif()
endif()

################################
# NVTOOLSEXT
################################
if(ENABLE_CUDA AND NOT TARGET CUDA::nvToolsExt)
   set(CUDAToolkit_ROOT ${CUDA_TOOLKIT_ROOT_DIR})
   find_package(CUDAToolkit)
endif()
