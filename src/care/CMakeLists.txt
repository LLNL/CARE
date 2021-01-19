######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

if (WIN32)
  if (NOT BUILD_SHARED_LIBS)
     add_definitions(-DCARESTATICLIB)
  endif ()
endif ()

configure_file(
    ${PROJECT_SOURCE_DIR}/src/care/config.h.in
    ${PROJECT_BINARY_DIR}/include/care/config.h)

set(care_headers
    ${PROJECT_BINARY_DIR}/include/care/config.h
    algorithm.h
    algorithm_decl.h
    array.h
    atomic.h
    CHAICallback.h
    CHAIDataGetter.h
    GPUWatchpoint.h
    Debug.h
    DefaultMacros.h
    device_ptr.h
    ExecutionSpace.h
    forall.h
    FOREACHMACRO.h
    GPUMacros.h
    host_device_ptr.h
    host_ptr.h
    KeyValueSorter.h
    KeyValueSorter_decl.h
    KeyValueSorter_inst.h
    local_host_device_ptr.h
    local_ptr.h
    LoopFuser.h
    managed_ptr.h
    openmp.h
    SortFuser.h
    numeric.h
    PointerTypes.h
    policies.h
    care.h
    RAJAPlugin.h
    scan.h
    scan_impl.h
    Setup.h
    single_access_ptr.h
    util.h
 )

set(care_sources
    care.cpp
    CHAICallback.cpp
    LoopFuser.cpp
    RAJAPlugin.cpp
    scan.cpp
    )

if(CARE_ENABLE_EXTERN_INSTANTIATE)
   list(APPEND care_headers care_inst.h)
   list(APPEND care_sources care_inst.cpp)
endif()


set(care_depends chai RAJA umpire camp)

if (ENABLE_MPI)
   list(APPEND care_depends mpi)
endif ()

if (LLNL_GLOBALID_FOUND)
   list (APPEND care_depends llnl_globalid)
endif()

if (ENABLE_CUDA)
   # separable compilation, delay all device linking to executable linking time
   # set (CUDA_SEPARABLE_COMPILATION ON BOOL "" )
   if (NVTOOLSEXT_FOUND)
      list(APPEND care_depends nvtoolsext)
   endif()

   list(APPEND care_depends cub cuda)
endif()

if (ENABLE_OPENMP)
   list(APPEND care_depends openmp)
endif()

if (ENABLE_HIP)
  list(APPEND care_depends hip)
endif ()

blt_add_library(NAME care
                SOURCES ${care_sources}
                HEADERS ${care_headers}
                DEPENDS_ON ${care_depends}
                )

target_include_directories(care PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(care PRIVATE ${PROJECT_BINARY_DIR}/include)

install(TARGETS care
        EXPORT  ${PROJECT_NAME}-targets 
        DESTINATION lib
        INCLUDES DESTINATION include/care
        )

install(FILES       ${care_headers}
        DESTINATION include/care
        )

install(EXPORT ${PROJECT_NAME}-targets DESTINATION share/care/cmake)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/care/care-config.cmake.in
  ${PROJECT_BINARY_DIR}/share/care/cmake/care-config.cmake)

install(
  FILES ${PROJECT_BINARY_DIR}/share/care/cmake/care-config.cmake
  DESTINATION share/care/cmake/)

