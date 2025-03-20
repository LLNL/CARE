##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set (care_test_dependencies gtest care)

if (ENABLE_CUDA)
   set (care_test_dependencies ${care_test_dependencies} cuda)
endif()

if (ENABLE_HIP)
  set (care_test_dependencies ${care_test_dependencies} blt::hip)
endif ()

if (ENABLE_OPENMP)
   set (care_test_dependencies ${care_test_dependencies} openmp)
endif()

blt_add_executable( NAME TestForall
                    SOURCES TestForall.cpp
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestForall
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestForall
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestForall
              COMMAND TestForall )

blt_add_executable( NAME TestLoopFuser
                    SOURCES TestLoopFuser.cxx
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

blt_add_executable( NAME TestAlgorithm
                    SOURCES TestAlgorithm.cpp
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestAlgorithm
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestAlgorithm
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestAlgorithm
              COMMAND TestAlgorithm )

blt_add_executable( NAME TestKeyValueSorter
                    SOURCES TestKeyValueSorter.cpp
                    DEPENDS_ON ${care_test_dependencies} )

target_include_directories(TestKeyValueSorter
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(TestKeyValueSorter
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_test( NAME TestKeyValueSorter
              COMMAND TestKeyValueSorter )

if (CARE_ENABLE_MANAGED_PTR)
   blt_add_executable( NAME TestManagedPtr
                       SOURCES TestManagedPtr.cpp
                       DEPENDS_ON ${care_test_dependencies} )

   target_include_directories(TestManagedPtr
                              PRIVATE ${PROJECT_SOURCE_DIR}/src)

   target_include_directories(TestManagedPtr
                              PRIVATE ${PROJECT_BINARY_DIR}/include)

   blt_add_test( NAME TestManagedPtr
                 COMMAND TestManagedPtr )
endif()

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
