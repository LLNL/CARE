######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

################################
# CUB (required for CUDA build)
################################
if (ENABLE_CUDA)
   if (CUB_DIR)
      message(STATUS "CARE: Using external CUB")
   else ()
      message(STATUS "CARE: Using CUB submodule")

      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/cub/cub/cub.cuh)
         message(FATAL_ERROR "CARE: CUB submodule not initialized. Run 'git submodule update --init' in the git repository or set CUB_DIR to use an external build of CUB.")
      else ()
         set(CUB_DIR ${PROJECT_SOURCE_DIR}/tpl/cub)
      endif ()
   endif ()

   find_package(CUB REQUIRED)
endif ()

################################
# CAMP (required)
################################
if (NOT TARGET camp)
   if (CAMP_DIR)
      set(camp_DIR ${CAMP_DIR}/lib/cmake/camp)
   endif ()

   find_package(camp QUIET)

   if (camp_FOUND)
      message(STATUS "CARE: Using external CAMP")
      set_target_properties(camp PROPERTIES IMPORTED_GLOBAL TRUE)
   else ()
      message(STATUS "CARE: Using CAMP submodule")

      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/camp/CMakeLists.txt)
         message(FATAL_ERROR "CARE: CAMP submodule not initialized. Run 'git submodule update --init' in the git repository or set camp_DIR or CAMP_DIR to use an external build of CAMP.")
      else ()
         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/camp)
      endif ()
   endif ()

   # Manually set includes as system includes
   get_target_property(_dirs camp INTERFACE_INCLUDE_DIRECTORIES)
   set_property(TARGET camp
                APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                "${_dirs}")

   if (ENABLE_CUDA)
      blt_add_target_definitions(TO camp
                                 SCOPE INTERFACE
                                 TARGET_DEFINITIONS CAMP_HAVE_CUDA)
   endif ()
endif ()

################################
# Umpire (required)
################################
if (NOT TARGET umpire)
   if (UMPIRE_DIR)
      set(umpire_DIR ${UMPIRE_DIR}/share/umpire/cmake)
   endif ()

   find_package(umpire QUIET)

   if (umpire_FOUND)
      message(STATUS "CARE: Using external Umpire")
   else ()
      message(STATUS "CARE: Using Umpire submodule")

      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/umpire/CMakeLists.txt)
         message(FATAL_ERROR "CARE: Umpire submodule not initialized. Run 'git submodule update --init' in the git repository or set umpire_DIR or UMPIRE_DIR to use an external build of Umpire.")
      else ()
         set(UMPIRE_ENABLE_TESTS ${CARE_ENABLE_SUBMODULE_TESTS} CACHE BOOL "Enable Umpire tests")
         set(UMPIRE_ENABLE_EXAMPLES ${CARE_ENABLE_SUBMODULE_EXAMPLES} CACHE BOOL "Enable Umpire examples")
         set(UMPIRE_ENABLE_DOCS ${CARE_ENABLE_SUBMODULE_DOCS} CACHE BOOL "Enable Umpire documentation")

         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/umpire)
      endif ()
   endif ()

   # Manually set includes as system includes
   get_target_property(_dirs umpire INTERFACE_INCLUDE_DIRECTORIES)
   set_property(TARGET umpire
                APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                "${_dirs}")
endif ()

################################
# RAJA (required)
################################
if (NOT TARGET RAJA)
   if (RAJA_DIR)
      set(raja_DIR ${RAJA_DIR}/share/raja/cmake)
   endif ()

   find_package(raja QUIET)

   if (raja_FOUND)
      message(STATUS "CARE: Using external RAJA")
   else ()
      message(STATUS "CARE: Using RAJA submodule")

      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/raja/CMakeLists.txt)
         message(FATAL_ERROR "CARE: RAJA submodule not initialized. Run 'git submodule update --init' in the git repository or set raja_DIR or RAJA_DIR to use an external build of RAJA.")
      else ()
         # TODO: Remove when these fixes are in RAJA
         # The current patch includes fixes for integrating CUB as a neighbor submodule.
         file(COPY ${PROJECT_SOURCE_DIR}/tpl/patches/raja/SetupPackages.cmake
              DESTINATION ${PROJECT_SOURCE_DIR}/tpl/raja/cmake)

         set(RAJA_ENABLE_TESTS ${CARE_ENABLE_SUBMODULE_TESTS} CACHE BOOL "Enable RAJA tests")
         set(RAJA_ENABLE_BENCHMARKS ${CARE_ENABLE_SUBMODULE_BENCHMARKS} CACHE BOOL "Enable RAJA benchmarks")
         set(RAJA_ENABLE_EXAMPLES ${CARE_ENABLE_SUBMODULE_EXAMPLES} CACHE BOOL "Enable RAJA examples")
         set(RAJA_ENABLE_EXERCISES ${CARE_ENABLE_SUBMODULE_EXERCISES} CACHE BOOL "Enable RAJA exercises")

         if (ENABLE_CUDA)
            # nvcc dies if compiler flags are duplicated, and RAJA adds duplicates
            set(CMAKE_CUDA_FLAGS "${RAJA_CMAKE_CUDA_FLAGS}")

            # Use external CUB
            set(ENABLE_EXTERNAL_CUB ON CACHE BOOL "Use external CUB in RAJA")
         endif ()

         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/raja)

         if (ENABLE_CUDA)
            # Reset CMAKE_CUDA_FLAGS
            set(CMAKE_CUDA_FLAGS "${CARE_CMAKE_CUDA_FLAGS}")
         endif ()
      endif ()
   endif ()

   # Manually set includes as system includes
   get_target_property(_dirs RAJA INTERFACE_INCLUDE_DIRECTORIES)
   set_property(TARGET RAJA
                APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                "${_dirs}")
endif ()

################################
# CHAI (required)
################################
if (NOT TARGET chai)
   if (CHAI_DIR)
      set(chai_DIR ${CHAI_DIR}/share/chai/cmake)
   endif ()

   find_package(chai QUIET)

   if (chai_FOUND)
      message(STATUS "CARE: Using external CHAI")
   else ()
      message(STATUS "CARE: Using CHAI submodule")

      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/chai/CMakeLists.txt)
         message(FATAL_ERROR "CARE: CHAI submodule not initialized. Run 'git submodule update --init' in the git repository or set chai_DIR or CHAI_DIR to use an external build of CHAI.")
      else ()
         # TODO: Put these changes back into umpire
         file(COPY ${PROJECT_SOURCE_DIR}/tpl/patches/chai/CMakeLists.txt
              DESTINATION ${PROJECT_SOURCE_DIR}/tpl/chai)

         set(CHAI_ENABLE_TESTS ${CARE_ENABLE_SUBMODULE_TESTS} CACHE BOOL "Enable CHAI tests")
         set(CHAI_ENABLE_BENCHMARKS ${CARE_ENABLE_SUBMODULE_BENCHMARKS} CACHE BOOL "Enable CHAI benchmarks")
         set(CHAI_ENABLE_EXAMPLES ${CARE_ENABLE_SUBMODULE_EXAMPLES} CACHE BOOL "Enable CHAI examples")
         set(CHAI_ENABLE_DOCS ${CARE_ENABLE_SUBMODULE_DOCS} CACHE BOOL "Enable CHAI documentation")

         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/chai)
      endif ()
   endif ()

   # Manually set includes as system includes
   get_target_property(_dirs chai INTERFACE_INCLUDE_DIRECTORIES)
   set_property(TARGET chai
                APPEND PROPERTY INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
                "${_dirs}")
endif ()

################################
# NVTX (optional)
################################
if (ENABLE_CUDA)
   if (NVTX_DIR)
      find_package(NVTX QUIET)

      if (NVTX_FOUND)
         message(STATUS "CARE: NVTX support enabled.")
         set(CARE_HAVE_NVTX "1" CACHE STRING "")
      else ()
         message(WARNING "CARE: Unable to find NVTX. NVTX support disabled.")
         set(CARE_HAVE_NVTX "0" CACHE STRING "")
      endif ()
   else ()
      message(STATUS "CARE: NVTX support disabled.")
      set(CARE_HAVE_NVTX "0" CACHE STRING "")
   endif ()
endif ()

################################
# LLNL_GlobalID (optional)
################################
if (LLNL_GLOBALID_DIR)
   find_package(LLNL_GlobalID QUIET)

   if (LLNL_GlobalID_FOUND)
      message(STATUS "CARE: LLNL_GlobalID support enabled.")
      set(CARE_HAVE_LLNL_GLOBALID "1" CACHE STRING "")
   else ()
      message(WARNING "CARE: Unable to find LLNL_GlobalID. LLNL_GlobalID support disabled.")
      set(CARE_HAVE_LLNL_GLOBALID "0" CACHE STRING "")
   endif ()
else ()
   message(STATUS "CARE: LLNL_GlobalID support disabled.")
   set(CARE_HAVE_LLNL_GLOBALID "0" CACHE STRING "")
endif ()

