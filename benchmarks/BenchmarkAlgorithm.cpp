//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#define GPU_ACTIVE

// CARE headers
#include "care/host_device_ptr.h"
#include "care/algorithm.h"

// Other library headers
#include <benchmark/benchmark.h>

// Std library headers
#include <climits>
#include <algorithm>

static void benchmark_vanilla_fill(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");
   int* host_data = data.data();

   for (auto _ : state) {
      for (int i = 0; i < size; ++i) {
         host_data[i] = 0;
      }
   }

   data.free();
}

static void benchmark_std_fill(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");
   int* host_data = data.data();

   for (auto _ : state) {
      std::fill(host_data, host_data + size, 0);
   }

   data.free();
}

static void benchmark_std_fill_n(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");
   int* host_data = data.data();

   for (auto _ : state) {
      std::fill_n(host_data, size, 0);
   }

   data.free();
}

static void benchmark_care_fill_n(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");

   for (auto _ : state) {
      care::fill_n(data, size, 0);
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_vanilla_fill)->Range(1, INT_MAX);
BENCHMARK(benchmark_std_fill)->Range(1, INT_MAX);
BENCHMARK(benchmark_std_fill_n)->Range(1, INT_MAX);
BENCHMARK(benchmark_care_fill_n)->Range(1, INT_MAX);

// Run the benchmark
BENCHMARK_MAIN();

