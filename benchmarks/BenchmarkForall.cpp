//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#define GPU_ACTIVE
#define OPENMP_ACTIVE

// CARE headers
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"

// Other library headers
#include <benchmark/benchmark.h>

// Std library headers
#include <climits>

static void benchmark_sequential_loop(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");

   for (auto _ : state) {
      CARE_SEQUENTIAL_LOOP(i, 0, size) {
         data[i] = i;
      } CARE_SEQUENTIAL_LOOP_END
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_sequential_loop)->Range(1, INT_MAX);

#if defined(_OPENMP)

static void benchmark_openmp_loop(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");

   for (auto _ : state) {
      CARE_OPENMP_LOOP(i, 0, size) {
         data[i] = i;
      } CARE_OPENMP_LOOP_END
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_openmp_loop)->Range(1, INT_MAX);

#endif

#if defined(CARE_GPUCC)

static void benchmark_gpu_loop(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");

   for (auto _ : state) {
      CARE_GPU_LOOP(i, 0, size) {
         data[i] = i;
      } CARE_GPU_LOOP_END
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop)->Range(1, INT_MAX);

#endif

// Run the benchmarks
BENCHMARK_MAIN();

