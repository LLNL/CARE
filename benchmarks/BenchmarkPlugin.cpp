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
#include "care/RAJAPlugin.h"
#include "RAJA/RAJA.hpp"

// Other library headers
#include <benchmark/benchmark.h>
#include <omp.h>

// Std library headers
#include <climits>
#include <cmath>

using namespace care;

static void benchmark_sequential_loop(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");
	for(auto _ : state){
   	forall(sequential{}, "test", 0, 0, size, [=] RAJA_HOST_DEVICE (int i) {
							data[i] = i;
			});
	}
   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_sequential_loop)->Range(1, INT_MAX / 2);

static void benchmark_sequential_loop_plugin(benchmark::State& state) {
   const int size = state.range(0);
   care::host_device_ptr<int> data(size, "data");

	for(auto _ : state){
   	forall(sequential{}, "test", 0, 0, size, [=] RAJA_HOST_DEVICE (int i) {
							data[i] = i;
			});
	}
	
   data.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_sequential_loop_plugin)->Range(1, INT_MAX / 2);

// Run the benchmarks
BENCHMARK_MAIN();
