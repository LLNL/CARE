//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/host_device_ptr.h"
#include "care/numeric.h"

// Other library headers
#include <benchmark/benchmark.h>

// Std library headers
#include <climits>
#include <numeric>

static void benchmark_std_iota(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");
   int* host_data = data.data();

   for (auto _ : state) {
      std::iota(host_data, host_data + size, 0);
   }

   data.free();
}

static void benchmark_care_iota(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");

   for (auto _ : state) {
      care::iota(RAJA::seq_exec{}, data, size, 0);
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_std_iota)->Range(1, INT_MAX);
BENCHMARK(benchmark_care_iota)->Range(1, INT_MAX);

// Run the benchmark
BENCHMARK_MAIN();

