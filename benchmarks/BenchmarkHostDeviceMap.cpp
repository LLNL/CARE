
//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#define GPU_ACTIVE
#define OPENMP_ACTIVE

// Other library headers
#include <benchmark/benchmark.h>

// CARE headers
#include "care/config.h"

#include "care/DefaultMacros.h"
#include "care/detail/test_utils.h"
#include "care/host_device_map.h"
#include "care/host_device_ptr.h"
#include "care/KeyValueSorter.h"


// Std library headers
#include <climits>


static void benchmark_init(benchmark::State& state ) {
   static bool initialized = 0;
   if (initialized == 0) {
      printf("Initializing\n");
      init_care_for_testing();
      printf("Initialized... \n");
      initialized = 1;
   }
   for (auto _ : state) {
   }
}
BENCHMARK(benchmark_init);


static void benchmark_kvs(benchmark::State& state) {
   for (auto _ : state) {
      const int length = state.range(0);
      care::host_device_ptr<int> data(length, "data");
      care::host_device_ptr<int> answer(length);
      PUSH_RANGE("createSorter")
      care::KeyValueSorter<size_t, int, RAJAExec> sorter(length);
      POP_RANGE
      PUSH_RANGE("insertions")
      CARE_STREAM_LOOP(i, 0, length) {
         sorter.setValue(i, length*10-2*i);
         sorter.setKey(i, i);
      } CARE_STREAM_LOOP_END
      POP_RANGE
      PUSH_RANGE("sort")
      sorter.sort();
      POP_RANGE
      PUSH_RANGE("lookups")
      CARE_STREAM_LOOP(i, 0, length) {
         answer[i] = sorter.keys()[care::BinarySearch<int>(sorter.values(),0,length,length*10-2*i)];
      } CARE_STREAM_LOOP_END
      POP_RANGE
      PUSH_RANGE("cleanup")
      answer.free();
      POP_RANGE
   }
}

// Register the function as a benchmark
BENCHMARK(benchmark_kvs)->Range(1, 1<<23);


template<typename Key, typename Value>
using care_std_map = care::host_device_map<Key, Value, RAJA::seq_exec>;

static void benchmark_seq_unordered_map(benchmark::State& state) {
   for (auto _ : state) {
      size_t length = state.range(0);
      PUSH_RANGE("createDeviceObject")
      care_std_map<int, int> data{length};
      POP_RANGE
      care::host_device_ptr<int> answer(length);
      PUSH_RANGE("insertions")
      CARE_SEQUENTIAL_LOOP(i,0,length) {
         data.emplace(length*10-2*i,i);
      } CARE_SEQUENTIAL_LOOP_END
      POP_RANGE
      PUSH_RANGE("sort");
      data.sort();
      POP_RANGE
      PUSH_RANGE("lookups")
      CARE_SEQUENTIAL_LOOP(i, 0, length) {
         answer[i] = data.at(length*10-2*i);
      } CARE_SEQUENTIAL_LOOP_END
      POP_RANGE
      PUSH_RANGE("cleanup")
      answer.free();
      data.free();
      cudaDeviceSynchronize();
      POP_RANGE
   }
}

// Register the function as a benchmark
BENCHMARK(benchmark_seq_unordered_map)->Range(1, 1<<23);

template<typename Key, typename Value>
using care_kv_map = care::host_device_map<Key, Value, RAJADeviceExec>;
static void benchmark_host_device_map(benchmark::State& state) {
   for (auto _ : state) {
      size_t length = state.range(0);
      PUSH_RANGE("createDeviceObject")
      care_kv_map<int, int> data{length};
      POP_RANGE
      care::host_device_ptr<int> answer(length);
      PUSH_RANGE("insertions")
      CARE_STREAM_LOOP(i,0,length) {
         data.emplace(length*10-2*i,i);
      } CARE_STREAM_LOOP_END
      POP_RANGE
      PUSH_RANGE("sort");
      data.sort();
      POP_RANGE
      PUSH_RANGE("lookups")
      CARE_STREAM_LOOP(i, 0, length) {
         answer[i] = data.at(length*10-2*i);
      } CARE_STREAM_LOOP_END
      POP_RANGE
      PUSH_RANGE("cleanup")
      answer.free();
      data.free();
      cudaDeviceSynchronize();
      POP_RANGE
   }
}
// Register the function as a benchmark
BENCHMARK(benchmark_host_device_map)->Range(1, 1<<23);

template<typename Key, typename Value>
using care_seq_kv_map = care::host_device_map<Key, Value, care::force_keyvaluesorter>;
static void benchmark_seq_force_kvs_unordered_map(benchmark::State& state) {

   for (auto _ : state) {
      size_t length = state.range(0);
      PUSH_RANGE("createDeviceObject")
      care_seq_kv_map<int, int> data{length};
      POP_RANGE
      care::host_device_ptr<int> answer(length);
      PUSH_RANGE("insertions")
      CARE_SEQUENTIAL_LOOP(i, 0, length) {
         data.emplace(length*10-2*i,i);
      } CARE_SEQUENTIAL_LOOP_END
      POP_RANGE
      PUSH_RANGE("sort");
      data.sort();
      POP_RANGE
      PUSH_RANGE("lookups")
      CARE_SEQUENTIAL_LOOP(i, 0, length) {
         answer[i] = data.at(length*10-2*i);
      } CARE_SEQUENTIAL_LOOP_END
      POP_RANGE
      PUSH_RANGE("cleanup")
      answer.free();
      data.free();
      cudaDeviceSynchronize();
      POP_RANGE
   }
}


// Register the function as a benchmark
BENCHMARK(benchmark_seq_force_kvs_unordered_map)->Range(1, 1<<23);


// Run the benchmarks
BENCHMARK_MAIN();
