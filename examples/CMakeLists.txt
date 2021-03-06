######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

set(care_depends
    care)

if (ENABLE_CUDA)
   list(APPEND care_depends cuda)
endif()

if (ENABLE_HIP)
  list(APPEND care_depends hip)
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

