//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/forall.h"
#include "care/policies.h"
#include "RAJA/RAJA.hpp"

// Other library headers
#include <benchmark/benchmark.h>

// Std library headers
#include <climits>
using namespace care;
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
//BENCHMARK(benchmark_sequential_loop)->Range(1, INT_MAX);

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
//BENCHMARK(benchmark_openmp_loop)->Range(1, INT_MAX);

#endif

#if defined(CARE_GPUCC)

static void benchmark_gpu_loop(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");

   for (auto _ : state) {
      care::forall(gpu{}, "test", 0, 0, size, [=] RAJA_HOST_DEVICE (int i) {
	      data[i] = i;
	   });
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop)->Range(1, INT_MAX/2);


static void benchmark_gpu_loop_stream_given(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");

   for (auto _ : state) {
      RAJA::resources::Cuda res;
      care::forall_given_stream(gpu{}, res, "test", 0, 0, size, [=] RAJA_HOST_DEVICE (int i) {
	      data[i] = i;
	   });
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop_stream_given)->Range(1, INT_MAX/2);

#endif

// Run the benchmarks
BENCHMARK_MAIN();

