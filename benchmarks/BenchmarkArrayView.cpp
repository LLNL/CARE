//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/ArrayView.h"

// Other library headers
#include <benchmark/benchmark.h>

// Std library headers
#include <climits>

static void benchmark_array_view_2d(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(3*size, "data");

   for (auto _ : state) {
      care::ArrayView2D<int> view = care::makeArrayView2D(data, 3, size);

      CARE_STREAM_LOOP(i, 0, size) {
         for (int j = 0; j < 3; ++j) {
            view(j, i) = i*j;
         }
      } CARE_STREAM_LOOP_END
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_array_view_2d)->Range(1, INT_MAX);

// Run the benchmark
BENCHMARK_MAIN();

