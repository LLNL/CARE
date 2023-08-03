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

#define size 1000000

namespace care{

#if defined(CARE_GPUCC)
//each kernel has a separate stream
static void benchmark_gpu_loop_separate_streams(benchmark::State& state) {
   int N = state.range(0);
   RAJA::resources::Cuda res_arr[N];
   RAJA::resources::Event event_arr[N];
   care::host_device_ptr<int> arrays[16];
   for(int i = 0; i < N; i++)
   {
      RAJA::resources::Cuda res;
      res_arr[i] = res;
      RAJA::resources::Event e = res.get_event();
      event_arr[i] = e;
      care::host_device_ptr<int> arr(size, "arr");
      arrays[i] = arr;
   }

   //warmup kernel
   RAJA::resources::Cuda warmup_res;
   CARE_STREAMED_LOOP(warmup_res, i, 0 , size) {
      arrays[0][i] = 0;
   } CARE_STREAMED_LOOP_END					
	
   for (auto _ : state) {
      //run num kernels
      for(int j = 0; j < N; j++)
      {
         CARE_STREAMED_LOOP(res_arr[j], i, 0 , size) {
            arrays[j][i] = i;
         } CARE_STREAMED_LOOP_END					
      }
   }
   for(int i = 0; i < N; i++) {arrays[i].free();}
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop_separate_streams)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(12)->Arg(16);

//all kernels on one stream
static void benchmark_gpu_loop_single_stream(benchmark::State& state) {
   int N = state.range(0);	

   RAJA::resources::Cuda res;   

   care::host_device_ptr<int> arrays[16];
   for(int i = 0; i < N; i++)
   {
      care::host_device_ptr<int> arr(size, "arr");
      arrays[i] = arr;
   }

   //warmup kernel
   RAJA::resources::Cuda warmup_res;
   CARE_STREAMED_LOOP(warmup_res, i, 0, size) {
      arrays[0][i] = i;
   } CARE_STREAMED_LOOP_END

   for (auto _ : state) {
      //run num kernels
      for(int j = 0; j < N; j++)
      {
         CARE_STREAMED_LOOP(res, i, 0, size) {
            arrays[j][i] = i;
         } CARE_STREAMED_LOOP_END
         res.wait();
      }
   }
   for(int i = 0; i < N; i++) {arrays[i].free();}
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop_single_stream)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(12)->Arg(16);

#endif

} //namespace care

// Run the benchmarks
BENCHMARK_MAIN();
