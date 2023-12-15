//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////
//
//
// CARE headers
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/util.h"

// Other library headers
#include <benchmark/benchmark.h>

// Std library headers
#include <climits>

#if 0
static void benchmark_cpu_loop_to_view(benchmark::State& state) {
   const int size = state.range(0);

   care::host_device_ptr<int> a(size, "data");
   care::host_device_ptr<int> b(size, "data");
   care::host_device_ptr<int> c(size, "data");

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      a[i] = i;
      b[i] = i*2;
   } CARE_SEQUENTIAL_LOOP_END

   for (auto _ : state) {
      CARE_COMPUTE_LOOP(i, 0, size,
                        CARE_CAPTURE_AS_VIEW(c)
                        CARE_CAPTURE_AS_CONST_VIEW(a, b)) {
         c[i] = a[i] + b[i];
      } CARE_COMPUTE_LOOP_END
   }

   c.free();
   b.free();
   a.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_cpu_loop_to_view)->Range(1, INT_MAX);
#endif

static void benchmark_sequential_loop(benchmark::State& state) {
   const int size = state.range(0);

   care::host_device_ptr<int> a(size, "data");
   care::host_device_ptr<int> b(size, "data");
   care::host_device_ptr<int> c(size, "data");

   // For consistency with the GPU case, which requires a warm up kernel
   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      a[i] = i;
      b[i] = i*2;
   } CARE_SEQUENTIAL_LOOP_END

   for (auto _ : state) {
      CARE_SEQUENTIAL_LOOP(i, 0, size) {
         c[i] = a[i] + b[i];
      } CARE_SEQUENTIAL_LOOP_END
   }

   c.free();
   b.free();
   a.free();
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

static void benchmark_gpu_loop_to_view(benchmark::State& state) {
   const int size = state.range(0);

   care::host_device_ptr<int> a(size, "a");
   care::host_device_ptr<int> b(size, "b");
   care::host_device_ptr<int> c(size, "c");

   // Warm up kernel
   CARE_GPU_LOOP(i, 0, size) {
      a[i] = i;
      b[i] = i*2;
   } CARE_GPU_LOOP_END

   care::gpuDeviceSynchronize(__FILE__, __LINE__);

   for (auto _ : state) {
      CARE_COMPUTE_LOOP(i, 0, size,
                        CARE_CAPTURE_AS_VIEW(c)
                        CARE_CAPTURE_AS_CONST_VIEW(a, b)) {
         c[i] = a[i] + b[i];
      } CARE_COMPUTE_LOOP_END

      // Timings are much more consistent with this synchronize
      care::gpuDeviceSynchronize(__FILE__, __LINE__);
   }

   c.free();
   b.free();
   a.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop_to_view)->Range(1, INT_MAX/2);



static void benchmark_gpu_loop(benchmark::State& state) {
   const int size = state.range(0);

   care::host_device_ptr<int> a(size, "a");
   care::host_device_ptr<int> b(size, "b");
   care::host_device_ptr<int> c(size, "c");

   // Warm up kernel
   CARE_GPU_LOOP(i, 0, size) {
      a[i] = i;
      b[i] = i*2;
   } CARE_GPU_LOOP_END

   care::gpuDeviceSynchronize(__FILE__, __LINE__);

   for (auto _ : state) {
      CARE_PARALLEL_LOOP(i, 0, size) {
         c[i] = a[i] + b[i];
      } CARE_PARALLEL_LOOP_END

      // Timings are much more consistent with this synchronize
      care::gpuDeviceSynchronize(__FILE__, __LINE__);
   }

   c.free();
   b.free();
   a.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop)->Range(1, INT_MAX/2);












#endif

// Run the benchmarks
BENCHMARK_MAIN();

