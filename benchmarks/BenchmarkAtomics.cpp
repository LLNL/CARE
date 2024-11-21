//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-24, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/util.h"

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

      const int row = element % elementsPerDim;
      const int col = element / elementsPerDim;
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
         std::abort();
      }
   } CARE_SEQUENTIAL_LOOP_END

   nodeTag.free();
   elementToNodeRelation.free();
}

// Register the function as a benchmark
BENCHMARK(benchmark_sequential_no_atomic)->Range(1, INT_MAX);

// Run the benchmarks
BENCHMARK_MAIN();

