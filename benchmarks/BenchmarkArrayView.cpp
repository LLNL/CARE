//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/ArrayView.h"

// Other library headers
#include <benchmark/benchmark.h>

// Std library headers
#include <climits>

static void benchmark_array_view_2d(benchmark::State& state) {
   const int extent1 = 2;
   const int extent2 = state.range(0);
   care::host_device_ptr<int> data(extent1*extent2, "data");

   // Make sure memory transfers are not considered in the timing
   auto temp = care::makeArrayView2D(data, extent1, extent2);
   (void) temp;

   for (auto _ : state) {
      auto view = care::makeArrayView2D(data, extent1, extent2);

      CARE_STREAM_LOOP(i, 0, extent2) {
         for (int j = 0; j < extent1; ++j) {
            view(j, i) = i*j;
         }
      } CARE_STREAM_LOOP_END
   }

   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_array_view_2d)->Range(32, INT_MAX/2);

// Run the benchmark
BENCHMARK_MAIN();

