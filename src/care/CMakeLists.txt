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
    algorithm_impl.h
    array.h
    atomic.h
    care_inst.h
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

if (CARE_ENABLE_EXTERN_INSTANTIATE)
   list(APPEND care_sources care_inst.cpp)
endif()

set(care_depends chai umpire RAJA camp)

blt_list_append(TO care_depends ELEMENTS LLNL_GlobalID::LLNL_GlobalID IF ${LLNL_GlobalID_FOUND})
blt_list_append(TO care_depends ELEMENTS mpi IF ${ENABLE_MPI})

if (ENABLE_CUDA)
   # separable compilation, delay all device linking to executable linking time
   # set (CUDA_SEPARABLE_COMPILATION ON BOOL "" )
   blt_list_append(TO care_depends ELEMENTS NVTX::NVTX IF ${NVTX_FOUND})
   list(APPEND care_depends CUB::CUB cuda)
endif()

blt_list_append(TO care_depends ELEMENTS hip IF ${ENABLE_HIP})
blt_list_append(TO care_depends ELEMENTS openmp IF ${ENABLE_OPENMP})

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

