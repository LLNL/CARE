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
   include(cmake/libraries/FindCUB.cmake)
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

      set(CAMP_INCLUDE_DIRS ${camp_INSTALL_PREFIX}/include)
      set(CAMP_DEPENDS )
      blt_list_append(TO CAMP_DEPENDS ELEMENTS cuda IF ENABLE_CUDA)

      blt_register_library(NAME camp
                           TREAT_INCLUDES_AS_SYSTEM ON
                           INCLUDES ${CAMP_INCLUDE_DIRS}
                           DEPENDS_ON ${CAMP_DEPENDS})
   else ()
      message(STATUS "CARE: Using CAMP submodule")

      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/camp/CMakeLists.txt)
         message(FATAL_ERROR "CARE: CAMP submodule not initialized. Run 'git submodule update --init' in the git repository or set camp_DIR or CAMP_DIR to use an external build of CAMP.")
      else ()
         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/camp)
      endif ()
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

      set(UMPIRE_LIBRARIES umpire)
      set(UMPIRE_DEPENDS camp)
      blt_list_append(TO UMPIRE_DEPENDS ELEMENTS mpi IF ENABLE_MPI)
      blt_list_append(TO UMPIRE_DEPENDS ELEMENTS cuda IF ENABLE_CUDA)

      blt_register_library(NAME umpire
                           TREAT_INCLUDES_AS_SYSTEM ON
                           INCLUDES ${UMPIRE_INCLUDE_DIRS}
                           LIBRARIES ${UMPIRE_LIBRARIES}
                           DEPENDS_ON ${UMPIRE_DEPENDS})
   else ()
      message(STATUS "CARE: Using Umpire submodule")

      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/umpire/CMakeLists.txt)
         message(FATAL_ERROR "CARE: Umpire submodule not initialized. Run 'git submodule update --init' in the git repository or set umpire_DIR or UMPIRE_DIR to use an external build of Umpire.")
      else ()
         # TODO: Put these changes back into umpire
         file(COPY ${PROJECT_SOURCE_DIR}/tpl/patches/umpire/CMakeLists.txt
              DESTINATION ${PROJECT_SOURCE_DIR}/tpl/umpire)

         set(UMPIRE_ENABLE_TESTS ${CARE_ENABLE_SUBMODULE_TESTS} CACHE BOOL "Enable Umpire tests")
         set(UMPIRE_ENABLE_BENCHMARKS ${CARE_ENABLE_SUBMODULE_BENCHMARKS} CACHE BOOL "Enable Umpire benchmarks")
         set(UMPIRE_ENABLE_EXAMPLES ${CARE_ENABLE_SUBMODULE_EXAMPLES} CACHE BOOL "Enable Umpire examples")
         set(UMPIRE_ENABLE_DOCS ${CARE_ENABLE_SUBMODULE_DOCS} CACHE BOOL "Enable Umpire documentation")
         set(UMPIRE_ENABLE_TOOLS ${CARE_ENABLE_SUBMODULE_TOOLS} CACHE BOOL "Enable Umpire tools")

         add_subdirectory(${PROJECT_SOURCE_DIR}/tpl/umpire)
      endif ()
   endif ()
endif ()

################################
# RAJA (required)
################################
if (NOT TARGET raja)
   if (RAJA_DIR)
      set(raja_DIR ${RAJA_DIR}/share/raja/cmake)
   endif ()

   find_package(raja QUIET)

   if (raja_FOUND)
      message(STATUS "CARE: Using external RAJA")

      get_target_property(RAJA_INCLUDE_DIRS RAJA INTERFACE_INCLUDE_DIRECTORIES)
      set(RAJA_LIBRARIES RAJA)
      set(RAJA_DEPENDS camp)
      blt_list_append(TO RAJA_DEPENDS ELEMENTS cuda IF ENABLE_CUDA)
      blt_list_append(TO RAJA_DEPENDS ELEMENTS openmp IF ENABLE_OPENMP)

      blt_register_library(NAME RAJA
                           TREAT_INCLUDES_AS_SYSTEM ON
                           INCLUDES ${RAJA_INCLUDE_DIRS}
                           LIBRARIES ${RAJA_LIBRARIES}
                           DEPENDS_ON ${RAJA_DEPENDS})
   else ()
      message(STATUS "CARE: Using RAJA submodule")

      if (NOT EXISTS ${PROJECT_SOURCE_DIR}/tpl/raja/CMakeLists.txt)
         message(FATAL_ERROR "CARE: RAJA submodule not initialized. Run 'git submodule update --init' in the git repository or set raja_DIR or RAJA_DIR to use an external build of RAJA.")
      else ()
         # TODO: Remove when these fixes are in RAJA (after v0.11.0).
         # The current patch includes fixes for integrating CAMP and CUB
         # as neighbor submodules.
         file(COPY ${PROJECT_SOURCE_DIR}/tpl/patches/raja/CMakeLists.txt
              DESTINATION ${PROJECT_SOURCE_DIR}/tpl/raja)

         set(RAJA_ENABLE_TESTS ${CARE_ENABLE_SUBMODULE_TESTS} CACHE BOOL "Enable RAJA tests")
         set(RAJA_ENABLE_BENCHMARKS ${CARE_ENABLE_SUBMODULE_BENCHMARKS} CACHE BOOL "Enable RAJA benchmarks")
         set(RAJA_ENABLE_EXAMPLES ${CARE_ENABLE_SUBMODULE_EXAMPLES} CACHE BOOL "Enable RAJA examples")
         set(RAJA_ENABLE_DOCS ${CARE_ENABLE_SUBMODULE_DOCS} CACHE BOOL "Enable RAJA documentation")
         set(RAJA_ENABLE_REPRODUCERS ${CARE_ENABLE_SUBMODULE_REPRODUCERS} CACHE BOOL "Enable RAJA reproducers")
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

      set(CHAI_LIBRARIES chai)
      set(CHAI_DEPENDS umpire camp)
      blt_list_append(TO CHAI_DEPENDS ELEMENTS mpi IF ENABLE_MPI)
      blt_list_append(TO CHAI_DEPENDS ELEMENTS cuda IF ENABLE_CUDA)

      blt_register_library(NAME chai
                           TREAT_INCLUDES_AS_SYSTEM ON
                           INCLUDES ${CHAI_INCLUDE_DIRS}
                           LIBRARIES ${CHAI_LIBRARIES}
                           DEPENDS_ON ${CHAI_DEPENDS})
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
endif ()

################################
# BASIL
################################
if (BASIL_DIR)
    include(cmake/libraries/FindBasil.cmake)
    if (BASIL_FOUND)
        set(BASIL_DEPENDS )
        blt_list_append(TO BASIL_DEPENDS ELEMENTS cuda IF ${ENABLE_CUDA})
        blt_register_library( NAME       basil
                                TREAT_INCLUDES_AS_SYSTEM ON
                                INCLUDES   ${BASIL_INCLUDE_DIR}
                                LIBRARIES  ${BASIL_LIBRARY}
                                DEPENDS_ON ${BASIL_DEPENDS})

        set(CARE_HAVE_BASIL "1" CACHE STRING "")
    else()
        message(FATAL_ERROR "CARE: Unable to find BASIL with given path: ${BASIL_DIR}")
    endif()
else()
    message(STATUS "CARE: BASIL disabled")
    set(CARE_HAVE_BASIL "0" CACHE STRING "")
endif()

################################
# NVTOOLSEXT
################################
if (NVTOOLSEXT_DIR)
    if (ENABLE_CUDA)
       include(cmake/libraries/FindNvToolsExt.cmake)
       if (NVTOOLSEXT_FOUND)
           blt_register_library( NAME       nvtoolsext
                                   TREAT_INCLUDES_AS_SYSTEM ON
                                   INCLUDES   ${NVTOOLSEXT_INCLUDE_DIRS}
                                   LIBRARIES  ${NVTOOLSEXT_LIBRARY}
                                   )

           set(CARE_HAVE_NVTOOLSEXT "0" CACHE STRING "")
       else()
           message(FATAL_ERROR "CARE: Unable to find NVTOOLSEXT with given path: ${NVTOOLSEXT_DIR}")
       endif()
    endif()
else()
    message(STATUS "CARE: NVTX disabled")
    set(CARE_HAVE_NVTOOLSEXT "0" CACHE STRING "")
endif()

################################
# LLNL_GlobalID
################################
if (LLNL_GLOBALID_DIR)
    include(cmake/libraries/FindLLNL_GlobalID.cmake)

    if (LLNL_GLOBALID_FOUND)
        blt_register_library( NAME       llnl_globalid
                                TREAT_INCLUDES_AS_SYSTEM ON
                                INCLUDES   ${LLNL_GLOBALID_INCLUDE_DIRS})
        set(CARE_HAVE_LLNL_GLOBALID "1" CACHE STRING "")
    else()
        message(FATAL_ERROR "CARE: Unable to find LLNL_GlobalID with given path: ${LLNL_GLOBALID_DIR}")
    endif()
else()
    message(STATUS "CARE: LLNL_GlobalID disabled")
    set(CARE_HAVE_LLNL_GLOBALID "0" CACHE STRING "")
endif()

