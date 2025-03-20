##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set (care_benchmark_depends care gbenchmark)
blt_list_append(TO care_benchmark_depends ELEMENTS openmp IF ${ENABLE_OPENMP})
blt_list_append(TO care_benchmark_depends ELEMENTS cuda IF ${ENABLE_CUDA})
blt_list_append(TO care_benchmark_depends ELEMENTS blt::hip IF ${ENABLE_HIP})

blt_add_executable(NAME BenchmarkArrayView
                   SOURCES BenchmarkArrayView.cpp
                   DEPENDS_ON ${care_benchmark_depends})

target_include_directories(BenchmarkArrayView
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(BenchmarkArrayView
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_benchmark(NAME BenchmarkArrayView
                  COMMAND BenchmarkArrayView)

blt_add_executable(NAME BenchmarkForall
                   SOURCES BenchmarkForall.cpp
                   DEPENDS_ON ${care_benchmark_depends})

target_include_directories(BenchmarkForall
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(BenchmarkForall
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_benchmark(NAME BenchmarkForall
                  COMMAND BenchmarkForall)

blt_add_executable(NAME BenchmarkNumeric
                   SOURCES BenchmarkNumeric.cpp
                   DEPENDS_ON ${care_benchmark_depends})

target_include_directories(BenchmarkNumeric
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(BenchmarkNumeric
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_benchmark(NAME BenchmarkNumeric
                  COMMAND BenchmarkNumeric)

blt_add_executable(NAME BenchmarkHostDeviceMap
                   SOURCES BenchmarkHostDeviceMap.cpp
                   DEPENDS_ON ${care_benchmark_depends})

target_include_directories(BenchmarkHostDeviceMap
                           PRIVATE ${PROJECT_SOURCE_DIR}/src)

target_include_directories(BenchmarkHostDeviceMap
                           PRIVATE ${PROJECT_BINARY_DIR}/include)

blt_add_benchmark(NAME BenchmarkHostDeviceMap
                  COMMAND BenchmarkHostDeviceMap)
