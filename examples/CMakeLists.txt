######################################################################################
# Copyright (c) 2020-24, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

set(care_depends
    care)

if (ENABLE_CUDA)
   list(APPEND care_depends cuda)
endif()

if (ENABLE_HIP)
  list(APPEND care_depends blt::hip)
endif ()

if (ENABLE_OPENMP)
   list(APPEND care_depends openmp)
endif()

blt_add_executable(NAME StreamLoopExample
                   SOURCES StreamLoop.cpp
                   DEPENDS_ON ${care_depends})

target_include_directories(StreamLoopExample PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(StreamLoopExample PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_executable(NAME ScanLoopExample
                   SOURCES ScanLoop.cpp
                   DEPENDS_ON ${care_depends})

target_include_directories(ScanLoopExample PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_include_directories(ScanLoopExample PRIVATE ${PROJECT_BINARY_DIR}/include)

