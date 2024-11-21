//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-24, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/local_ptr.h"
#include "care/util.h"
#include "care/atomic.h"

// Other library headers
#include <benchmark/benchmark.h>

// Std library headers
#include <climits>

static void benchmark_sequential_no_atomic(benchmark::State& state) {
   const int elementsPerDim = (int) std::sqrt(state.range(0));
   const int numElements = elementsPerDim * elementsPerDim;

   const int nodesPerDim = elementsPerDim + 1;
   const int numNodes = nodesPerDim * nodesPerDim;

   const int nodesPerElement = 4;
   care::host_device_ptr<int> elementToNodeRelation(numElements * nodesPerElement);

   CARE_SEQUENTIAL_LOOP(i, 0, numElements) {
      care::local_ptr<int> elementNodes =
         elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

      const int row = i % elementsPerDim;
      const int col = i / elementsPerDim;
      const int startNode = col*nodesPerDim + row;

      elementNodes[0] = startNode;
      elementNodes[1] = startNode + 1;
      elementNodes[2] = startNode + nodesPerDim + 1;
      elementNodes[3] = startNode + nodesPerDim;
   } CARE_SEQUENTIAL_LOOP_END

   care::host_device_ptr<int> nodeTag(numNodes, "nodeTag");

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      nodeTag[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   for (auto _ : state) {
      CARE_SEQUENTIAL_LOOP(i, 0, numElements) {
         care::local_ptr<int> elementNodes =
            elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

         for (int j = 0; j < nodesPerElement; ++j) {
            nodeTag[elementNodes[j]] = 1;
         }
      } CARE_SEQUENTIAL_LOOP_END
   }

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      if (nodeTag[i] == 0) {
         std::cerr << "benchmark_sequential_no_atomic failed!\n";
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_sequential_no_atomic)->Range(1, INT_MAX/8);

static void benchmark_sequential_atomic_store(benchmark::State& state) {
   const int elementsPerDim = (int) std::sqrt(state.range(0));
   const int numElements = elementsPerDim * elementsPerDim;

   const int nodesPerDim = elementsPerDim + 1;
   const int numNodes = nodesPerDim * nodesPerDim;

   const int nodesPerElement = 4;
   care::host_device_ptr<int> elementToNodeRelation(numElements * nodesPerElement);

   CARE_SEQUENTIAL_LOOP(i, 0, numElements) {
      care::local_ptr<int> elementNodes =
         elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

      const int row = i % elementsPerDim;
      const int col = i / elementsPerDim;
      const int startNode = col*nodesPerDim + row;

      elementNodes[0] = startNode;
      elementNodes[1] = startNode + 1;
      elementNodes[2] = startNode + nodesPerDim + 1;
      elementNodes[3] = startNode + nodesPerDim;
   } CARE_SEQUENTIAL_LOOP_END

   care::host_device_ptr<int> nodeTag(numNodes, "nodeTag");

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      nodeTag[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   for (auto _ : state) {
      CARE_SEQUENTIAL_LOOP(i, 0, numElements) {
         care::local_ptr<int> elementNodes =
            elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

         for (int j = 0; j < nodesPerElement; ++j) {
            ATOMIC_STORE(nodeTag[elementNodes[j]], 1);
         }
      } CARE_SEQUENTIAL_LOOP_END
   }

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      if (nodeTag[i] == 0) {
         std::cerr << "benchmark_sequential_atomic_store failed!\n";
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_sequential_atomic_store)->Range(1, INT_MAX/8);

static void benchmark_sequential_atomic_cas(benchmark::State& state) {
   const int elementsPerDim = (int) std::sqrt(state.range(0));
   const int numElements = elementsPerDim * elementsPerDim;

   const int nodesPerDim = elementsPerDim + 1;
   const int numNodes = nodesPerDim * nodesPerDim;

   const int nodesPerElement = 4;
   care::host_device_ptr<int> elementToNodeRelation(numElements * nodesPerElement);

   CARE_SEQUENTIAL_LOOP(i, 0, numElements) {
      care::local_ptr<int> elementNodes =
         elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

      const int row = i % elementsPerDim;
      const int col = i / elementsPerDim;
      const int startNode = col*nodesPerDim + row;

      elementNodes[0] = startNode;
      elementNodes[1] = startNode + 1;
      elementNodes[2] = startNode + nodesPerDim + 1;
      elementNodes[3] = startNode + nodesPerDim;
   } CARE_SEQUENTIAL_LOOP_END

   care::host_device_ptr<int> nodeTag(numNodes, "nodeTag");

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      nodeTag[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   for (auto _ : state) {
      CARE_SEQUENTIAL_LOOP(i, 0, numElements) {
         care::local_ptr<int> elementNodes =
            elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

         for (int j = 0; j < nodesPerElement; ++j) {
            ATOMIC_CAS(nodeTag[elementNodes[j]], 0, 1);
         }
      } CARE_SEQUENTIAL_LOOP_END
   }

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      if (nodeTag[i] == 0) {
         std::cerr << "benchmark_sequential_atomic_cas failed!\n";
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_sequential_atomic_cas)->Range(1, INT_MAX/8);

#if defined(_OPENMP)

static void benchmark_openmp_no_atomic(benchmark::State& state) {
   const int elementsPerDim = (int) std::sqrt(state.range(0));
   const int numElements = elementsPerDim * elementsPerDim;

   const int nodesPerDim = elementsPerDim + 1;
   const int numNodes = nodesPerDim * nodesPerDim;

   const int nodesPerElement = 4;
   care::host_device_ptr<int> elementToNodeRelation(numElements * nodesPerElement);

   CARE_OPENMP_LOOP(i, 0, numElements) {
      care::local_ptr<int> elementNodes =
         elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

      const int row = i % elementsPerDim;
      const int col = i / elementsPerDim;
      const int startNode = col*nodesPerDim + row;

      elementNodes[0] = startNode;
      elementNodes[1] = startNode + 1;
      elementNodes[2] = startNode + nodesPerDim + 1;
      elementNodes[3] = startNode + nodesPerDim;
   } CARE_OPENMP_LOOP_END

   care::host_device_ptr<int> nodeTag(numNodes, "nodeTag");

   CARE_OPENMP_LOOP(i, 0, numNodes) {
      nodeTag[i] = 0;
   } CARE_OPENMP_LOOP_END

   // I believe the openmp loop synchronizes by default

   for (auto _ : state) {
      CARE_OPENMP_LOOP(i, 0, numElements) {
         care::local_ptr<int> elementNodes =
            elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

         for (int j = 0; j < nodesPerElement; ++j) {
            nodeTag[elementNodes[j]] = 1;
         }
      } CARE_OPENMP_LOOP_END

      // I believe the openmp loop synchronizes by default
   }

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      if (nodeTag[i] == 0) {
         std::abort();
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_openmp_no_atomic)->Range(1, INT_MAX/8);

static void benchmark_openmp_atomic_store(benchmark::State& state) {
   const int elementsPerDim = (int) std::sqrt(state.range(0));
   const int numElements = elementsPerDim * elementsPerDim;

   const int nodesPerDim = elementsPerDim + 1;
   const int numNodes = nodesPerDim * nodesPerDim;

   const int nodesPerElement = 4;
   care::host_device_ptr<int> elementToNodeRelation(numElements * nodesPerElement);

   CARE_OPENMP_LOOP(i, 0, numElements) {
      care::local_ptr<int> elementNodes =
         elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

      const int row = i % elementsPerDim;
      const int col = i / elementsPerDim;
      const int startNode = col*nodesPerDim + row;

      elementNodes[0] = startNode;
      elementNodes[1] = startNode + 1;
      elementNodes[2] = startNode + nodesPerDim + 1;
      elementNodes[3] = startNode + nodesPerDim;
   } CARE_OPENMP_LOOP_END

   care::host_device_ptr<int> nodeTag(numNodes, "nodeTag");

   CARE_OPENMP_LOOP(i, 0, numNodes) {
      nodeTag[i] = 0;
   } CARE_OPENMP_LOOP_END

   // I believe the openmp loop synchronizes by default

   for (auto _ : state) {
      CARE_OPENMP_LOOP(i, 0, numElements) {
         care::local_ptr<int> elementNodes =
            elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

         for (int j = 0; j < nodesPerElement; ++j) {
            ATOMIC_STORE(nodeTag[elementNodes[j]], 1);
         }
      } CARE_OPENMP_LOOP_END

      // I believe the openmp loop synchronizes by default
   }

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      if (nodeTag[i] == 0) {
         std::abort();
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_openmp_atomic_store)->Range(1, INT_MAX/8);

static void benchmark_openmp_atomic_cas(benchmark::State& state) {
   const int elementsPerDim = (int) std::sqrt(state.range(0));
   const int numElements = elementsPerDim * elementsPerDim;

   const int nodesPerDim = elementsPerDim + 1;
   const int numNodes = nodesPerDim * nodesPerDim;

   const int nodesPerElement = 4;
   care::host_device_ptr<int> elementToNodeRelation(numElements * nodesPerElement);

   CARE_OPENMP_LOOP(i, 0, numElements) {
      care::local_ptr<int> elementNodes =
         elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

      const int row = i % elementsPerDim;
      const int col = i / elementsPerDim;
      const int startNode = col*nodesPerDim + row;

      elementNodes[0] = startNode;
      elementNodes[1] = startNode + 1;
      elementNodes[2] = startNode + nodesPerDim + 1;
      elementNodes[3] = startNode + nodesPerDim;
   } CARE_OPENMP_LOOP_END

   care::host_device_ptr<int> nodeTag(numNodes, "nodeTag");

   CARE_OPENMP_LOOP(i, 0, numNodes) {
      nodeTag[i] = 0;
   } CARE_OPENMP_LOOP_END

   // I believe the openmp loop synchronizes by default

   for (auto _ : state) {
      CARE_OPENMP_LOOP(i, 0, numElements) {
         care::local_ptr<int> elementNodes =
            elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

         for (int j = 0; j < nodesPerElement; ++j) {
            ATOMIC_CAS(nodeTag[elementNodes[j]], 0, 1);
         }
      } CARE_OPENMP_LOOP_END

      // I believe the openmp loop synchronizes by default
   }

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      if (nodeTag[i] == 0) {
         std::abort();
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_openmp_atomic_cas)->Range(1, INT_MAX/8);

#endif

#if defined(CARE_GPUCC)

static void benchmark_gpu_no_atomic(benchmark::State& state) {
   const int elementsPerDim = (int) std::sqrt(state.range(0));
   const int numElements = elementsPerDim * elementsPerDim;

   const int nodesPerDim = elementsPerDim + 1;
   const int numNodes = nodesPerDim * nodesPerDim;

   const int nodesPerElement = 4;
   care::host_device_ptr<int> elementToNodeRelation(numElements * nodesPerElement);

   CARE_GPU_LOOP(i, 0, numElements) {
      care::local_ptr<int> elementNodes =
         elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

      const int row = i % elementsPerDim;
      const int col = i / elementsPerDim;
      const int startNode = col*nodesPerDim + row;

      elementNodes[0] = startNode;
      elementNodes[1] = startNode + 1;
      elementNodes[2] = startNode + nodesPerDim + 1;
      elementNodes[3] = startNode + nodesPerDim;
   } CARE_GPU_LOOP_END

   care::host_device_ptr<int> nodeTag(numNodes, "nodeTag");

   CARE_GPU_LOOP(i, 0, numNodes) {
      nodeTag[i] = 0;
   } CARE_GPU_LOOP_END

   care::gpuDeviceSynchronize(__FILE__, __LINE__);

   for (auto _ : state) {
      CARE_GPU_LOOP(i, 0, numElements) {
         care::local_ptr<int> elementNodes =
            elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

         for (int j = 0; j < nodesPerElement; ++j) {
            nodeTag[elementNodes[j]] = 1;
         }
      } CARE_GPU_LOOP_END

      care::gpuDeviceSynchronize(__FILE__, __LINE__);
   }

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      if (nodeTag[i] == 0) {
         std::abort();
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_no_atomic)->Range(1, INT_MAX/8);

static void benchmark_gpu_atomic_store(benchmark::State& state) {
   const int elementsPerDim = (int) std::sqrt(state.range(0));
   const int numElements = elementsPerDim * elementsPerDim;

   const int nodesPerDim = elementsPerDim + 1;
   const int numNodes = nodesPerDim * nodesPerDim;

   const int nodesPerElement = 4;
   care::host_device_ptr<int> elementToNodeRelation(numElements * nodesPerElement);

   CARE_GPU_LOOP(i, 0, numElements) {
      care::local_ptr<int> elementNodes =
         elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

      const int row = i % elementsPerDim;
      const int col = i / elementsPerDim;
      const int startNode = col*nodesPerDim + row;

      elementNodes[0] = startNode;
      elementNodes[1] = startNode + 1;
      elementNodes[2] = startNode + nodesPerDim + 1;
      elementNodes[3] = startNode + nodesPerDim;
   } CARE_GPU_LOOP_END

   care::host_device_ptr<int> nodeTag(numNodes, "nodeTag");

   CARE_GPU_LOOP(i, 0, numNodes) {
      nodeTag[i] = 0;
   } CARE_GPU_LOOP_END

   care::gpuDeviceSynchronize(__FILE__, __LINE__);

   for (auto _ : state) {
      CARE_GPU_LOOP(i, 0, numElements) {
         care::local_ptr<int> elementNodes =
            elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

         for (int j = 0; j < nodesPerElement; ++j) {
            ATOMIC_STORE(nodeTag[elementNodes[j]], 1);
         }
      } CARE_GPU_LOOP_END

      care::gpuDeviceSynchronize(__FILE__, __LINE__);
   }

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      if (nodeTag[i] == 0) {
         std::abort();
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_atomic_store)->Range(1, INT_MAX/8);

static void benchmark_gpu_atomic_cas(benchmark::State& state) {
   const int elementsPerDim = (int) std::sqrt(state.range(0));
   const int numElements = elementsPerDim * elementsPerDim;

   const int nodesPerDim = elementsPerDim + 1;
   const int numNodes = nodesPerDim * nodesPerDim;

   const int nodesPerElement = 4;
   care::host_device_ptr<int> elementToNodeRelation(numElements * nodesPerElement);

   CARE_GPU_LOOP(i, 0, numElements) {
      care::local_ptr<int> elementNodes =
         elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

      const int row = i % elementsPerDim;
      const int col = i / elementsPerDim;
      const int startNode = col*nodesPerDim + row;

      elementNodes[0] = startNode;
      elementNodes[1] = startNode + 1;
      elementNodes[2] = startNode + nodesPerDim + 1;
      elementNodes[3] = startNode + nodesPerDim;
   } CARE_GPU_LOOP_END

   care::host_device_ptr<int> nodeTag(numNodes, "nodeTag");

   CARE_GPU_LOOP(i, 0, numNodes) {
      nodeTag[i] = 0;
   } CARE_GPU_LOOP_END

   care::gpuDeviceSynchronize(__FILE__, __LINE__);

   for (auto _ : state) {
      CARE_GPU_LOOP(i, 0, numElements) {
         care::local_ptr<int> elementNodes =
            elementToNodeRelation.slice(i * nodesPerElement, nodesPerElement);

         for (int j = 0; j < nodesPerElement; ++j) {
            ATOMIC_CAS(nodeTag[elementNodes[j]], 0, 1);
         }
      } CARE_GPU_LOOP_END

      care::gpuDeviceSynchronize(__FILE__, __LINE__);
   }

   CARE_SEQUENTIAL_LOOP(i, 0, numNodes) {
      if (nodeTag[i] == 0) {
         std::abort();
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_gpu_atomic_cas)->Range(1, INT_MAX/8);

#endif

// Run the benchmarks
BENCHMARK_MAIN();

