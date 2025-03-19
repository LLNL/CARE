//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/util.h"

// Other library headers
#include <benchmark/benchmark.h>

// Std library headers
#include <climits>

static void benchmark_sequential_loop(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");

   // For consistency with the GPU case, which requires a warm up kernel
   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      data[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   for (auto _ : state) {
      CARE_SEQUENTIAL_LOOP(i, 0, size) {
         data[i] += i;
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

   // For consistency with the GPU case, which requires a warm up kernel
   CARE_OPENMP_LOOP(i, 0, size) {
      data[i] = 0;
   } CARE_OPENMP_LOOP_END

   // TODO: Is a synchronize needed?

   for (auto _ : state) {
      CARE_OPENMP_LOOP(i, 0, size) {
         data[i] += i;
      } CARE_OPENMP_LOOP_END

      // TODO: Is a synchronize needed?
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

   // Warm up kernel
   CARE_GPU_LOOP(i, 0, size) {
      data[i] = 0;
   } CARE_GPU_LOOP_END

   care::gpuDeviceSynchronize(__FILE__, __LINE__);

   for (auto _ : state) {
      CARE_GPU_LOOP(i, 0, size) {
         data[i] += i;
      } CARE_GPU_LOOP_END

      // Timings are much more consistent with this synchronize
      care::gpuDeviceSynchronize(__FILE__, __LINE__);
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop)->Range(1, INT_MAX);

#endif

// Run the benchmarks
BENCHMARK_MAIN();

