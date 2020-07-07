######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

set (care_test_dependencies gtest care)

if (ENABLE_CUDA)
   set (care_test_dependencies ${care_test_dependencies} cuda)
endif()

if (ENABLE_HIP)
  set (care_test_dependencies ${care_test_dependencies} hip)
endif ()

if (ENABLE_OPENMP)
   set (care_test_dependencies ${care_test_dependencies} openmp)
endif()

blt_add_executable( NAME TestLoopFuser
                    SOURCES LoopFuserTest.cxx
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestLoopFuser
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestLoopFuser
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestLoopFuser
              COMMAND TestLoopFuser )

blt_add_executable( NAME TestSortFuser
                    SOURCES TestSortFuser.cxx
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestSortFuser
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestSortFuser
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestSortFuser
              COMMAND TestSortFuser )

blt_add_executable( NAME TestArray
                    SOURCES TestArray.cpp
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestArray
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestArray
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestArray
              COMMAND TestArray )

blt_add_executable( NAME TestArrayUtils
                    SOURCES TestArrayUtils.cpp
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestArrayUtils
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestArrayUtils
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestArrayUtils
              COMMAND TestArrayUtils )

blt_add_executable( NAME TestNumeric
                    SOURCES TestNumeric.cpp
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestNumeric
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestNumeric
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestNumeric
              COMMAND TestNumeric )

blt_add_executable( NAME TestNestedMA
                    SOURCES TestNestedMA.cpp
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestNestedMA
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestNestedMA
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestNestedMA
              COMMAND TestNestedMA )

blt_add_executable( NAME TestScan
                    SOURCES TestScan.cpp
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestScan
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestScan
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestScan
              COMMAND TestScan )

blt_add_executable( NAME Benchmarks
                    SOURCES Benchmarks.cpp
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(Benchmarks
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(Benchmarks
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME Benchmarks
              COMMAND Benchmarks )
