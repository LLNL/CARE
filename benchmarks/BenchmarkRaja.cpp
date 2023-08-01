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
#include <omp.h>

// Std library headers
#include <climits>
#include <cmath>

#define NUM_KERNELS 4

using namespace care;

#if defined(CARE_GPUCC)
//each kernel has a separate stream
static void benchmark_gpu_loop_separate_streams(benchmark::State& state) {
   int N = state.range(0);

   RAJA::resources::Cuda res_arr[NUM_KERNELS];
   RAJA::resources::Event event_arr[NUM_KERNELS];
   for(int i = 0; i < NUM_KERNELS; i++)
   {
      RAJA::resources::Cuda res;
      res_arr[i] = res;
      RAJA::resources::Event e = res.get_event();
      event_arr[i] = e;
   }
	
   care::host_device_ptr<int> arr(N, "arr");
   for (auto _ : state) {
      //run num kernels
      for(int j = 0; j < NUM_KERNELS; j++)
      {
         CARE_STREAMED_LOOP(res_arr[j], i, 0 , N) {
         arr[i] = i;
         } CARE_STREAMED_LOOP_END					
         if(j > 0) res_arr[j].wait_for(&event_arr[j - 1]);
      }
   }
   arr.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop_separate_streams)->Range(1, INT_MAX);

//all kernels on one stream
static void benchmark_gpu_loop_single_stream(benchmark::State& state) {
   int N = state.range(0);	

   RAJA::resources::Cuda res;   

   care::host_device_ptr<int> arr(N, "arr");	
   for (auto _ : state) {
      //run num kernels
      for(int j = 0; j < NUM_KERNELS; j++)
      {
         CARE_STREAMED_LOOP(res, i, 0, N) {
         arr[i] = i;
         }CARE_STREAMED_LOOP_END
         res.wait();
      }
   }
   arr.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop_single_stream)->Range(1, INT_MAX);

#endif

// Run the benchmarks
BENCHMARK_MAIN();
