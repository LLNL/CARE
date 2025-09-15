##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

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
    PluginData.h
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
    PluginData.cpp
    DebugPlugin.cpp
    ProfilePlugin.cpp
    scan.cpp
    )

if(CARE_ENABLE_EXTERN_INSTANTIATE)
   list(APPEND care_sources care_inst.cpp)
endif()

set(care_includes
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>)

set(care_depends chai RAJA umpire)

if(LLNL_GLOBALID_FOUND)
   list(APPEND care_depends llnl_globalid::llnl_globalid)
endif()

if(ENABLE_MPI)
   list(APPEND care_depends mpi)
endif()

if(ENABLE_OPENMP)
   list(APPEND care_depends openmp)
endif()

if(ENABLE_CUDA)
   # separable compilation, delay all device linking to executable linking time
   # set (CUDA_SEPARABLE_COMPILATION ON BOOL "" )

   if(TARGET CUDA::nvtx3)
      list(APPEND care_depends CUDA::nvtx3)
   endif()

   list(APPEND care_depends CUB::CUB cuda)
endif()


if(ENABLE_HIP)
  list(APPEND care_depends blt::hip)
endif ()

blt_add_library(NAME care
                SOURCES    ${care_sources}
                HEADERS    ${care_headers}
                INCLUDES   ${care_includes}
                DEPENDS_ON ${care_depends})

################################
# care package install
################################

install(FILES       ${care_headers}
        DESTINATION include/care)

install(TARGETS care
        EXPORT  care-targets
        DESTINATION lib)

install(EXPORT care-targets
        NAMESPACE care::
        DESTINATION lib/cmake/care)

blt_install_tpl_setups(DESTINATION lib/cmake/care)

include(CMakePackageConfigHelpers)

configure_package_config_file(
   ${PROJECT_SOURCE_DIR}/cmake/care-config.cmake.in
   ${PROJECT_BINARY_DIR}/cmake/care-config.cmake
   INSTALL_DESTINATION lib/cmake/care)

write_basic_package_version_file(
   ${PROJECT_BINARY_DIR}/cmake/care-config-version.cmake
   COMPATIBILITY SameMajorVersion)

install(
   FILES
      ${PROJECT_BINARY_DIR}/cmake/care-config.cmake
      ${PROJECT_BINARY_DIR}/cmake/care-config-version.cmake
      ${PROJECT_SOURCE_DIR}/cmake/modules/FindCUB.cmake
   DESTINATION
      lib/cmake/care)

