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

#if defined(CARE_GPUCC)
//each kernel has a separate stream
static void benchmark_gpu_loop_separate_streams(benchmark::State& state) {
   int N = state.range(0);
   care::Resource res_arr[16];
   RAJA::resources::Event event_arr[16];
   care::host_device_ptr<int> arrays[16];
   for(int i = 0; i < N; i++)
   {
      res_arr[i] = care::Resource();
      event_arr[i] = res_arr[i].get_event();
      arrays[i] = care::host_device_ptr<int>(size, "arr");
   }

   //warmup kernel
   CARE_GPU_LOOP(i, 0 , size) {
      arrays[0][i] = 0;
   } CARE_GPU_LOOP_END	

   care::gpuDeviceSynchronize(__FILE__, __LINE__);   
	
   for (auto _ : state) {
      //run num kernels
      omp_set_num_threads(16);
      #pragma omp parallel for
      for(int j = 0; j < N; j++)
      {
            CARE_STREAMED_LOOP(res_arr[j], i, 0 , size) {
            arrays[j][i] = sqrtf(i) + cosf(j) * powf(i, j);
         } CARE_STREAMED_LOOP_END					
      }
      care::gpuDeviceSynchronize(__FILE__, __LINE__);
   }

   for(int i = 0; i < N; i++){
      arrays[i].free();
   }
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop_separate_streams)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(12)->Arg(16);

//all kernels on one stream
static void benchmark_gpu_loop_single_stream(benchmark::State& state) {
   int N = state.range(0);	

   care::host_device_ptr<int> arrays[16];
   for(int i = 0; i < N; i++)
   {
      arrays[i] = care::host_device_ptr<int>(size, "arr");
   }

   //warmup kernel
   CARE_GPU_LOOP(i, 0, size) {
      arrays[0][i] = 0;
   } CARE_GPU_LOOP_END

   care::gpuDeviceSynchronize(__FILE__, __LINE__);

   for (auto _ : state) {
      //run num kernels
      for(int j = 0; j < N; j++)
      {
         CARE_GPU_LOOP(i, 0, size) {
            arrays[j][i] = sqrtf(i) + cosf(j) * powf(i, j);
         } CARE_GPU_LOOP_END
      }
      care::gpuDeviceSynchronize(__FILE__, __LINE__);
   }

   for(int i = 0; i < N; i++){
      arrays[i].free();
   }
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_loop_single_stream)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(12)->Arg(16);

#endif

// Run the benchmarks
BENCHMARK_MAIN();
