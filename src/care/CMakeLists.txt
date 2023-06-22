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
    Accessor.h
    algorithm.h
    algorithm_decl.h
    algorithm_impl.h
    array.h
    ArrayView.h
    atomic.h
    care_inst.h
    CHAICallback.h
    CHAIDataGetter.h
    GPUWatchpoint.h
    Debug.h
    DefaultMacros.h
    device_ptr.h
    host_device_map.h
    ExecutionSpace.h
    forall.h
    FOREACHMACRO.h
    GPUMacros.h
    host_device_ptr.h
    host_ptr.h
    KeyValueSorter.h
    KeyValueSorter_decl.h
    KeyValueSorter_impl.h
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
    DebugPlugin.h
    ProfilePlugin.h
    scan.h
    scan_impl.h
    Setup.h
    single_access_ptr.h
    util.h
 )

set(care_sources
    care.cpp
    CHAICallback.cpp
    ExecutionSpace.cpp
    LoopFuser.cpp
    RAJAPlugin.cpp
    DebugPlugin.cpp
    ProfilePlugin.cpp
    scan.cpp
    )

if(CARE_ENABLE_EXTERN_INSTANTIATE)
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
  list(APPEND care_depends blt::hip)
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

blt_export_tpl_targets(EXPORT ${PROJECT_NAME}-targets NAMESPACE care)

configure_file(
  ${PROJECT_SOURCE_DIR}/src/care/care-config.cmake.in
  ${PROJECT_BINARY_DIR}/share/care/cmake/care-config.cmake)

install(
  FILES ${PROJECT_BINARY_DIR}/share/care/cmake/care-config.cmake
  DESTINATION share/care/cmake/)

