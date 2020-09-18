//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// This macro needs to be defined before including care/care.h,
// which allows you to port a file at a time. Without this define,
// CARE_STREAM_LOOP will run on the CPU. With this define, CARE_STREAM_LOOP
// will run on the GPU only if you built with CUDA enabled. Otherwise,
// if CUDA is disabled, it will default back to running on the host.
#define GPU_ACTIVE

// CARE library header
#include "care/care.h"
#include "care/algorithm.h"

// Std library headers
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

int main(int, char**) {
   ///////////////
   // Normal code
   ///////////////

   // Set the size
   int size = 1000;

   // Allocate memory on the host using a standard library container
   std::vector<int> data(size);

   // Fill the memory on the host using a for loop
   for (int i = 0; i < size; ++i) {
      data[i] = i * i;
   }

   // Use a standard library algorithm to find the max element and print it
   int max = *std::max_element(data.begin(), data.end());

   // Print out the max element
   std::cout << "CPU max: " << max << std::endl;

   ///////////////
   // Ported code
   ///////////////

   // Set the size
   int size2 = 1000;

   // Allocate memory on the host and device using a CARE wrapper around a CHAI container.
   // This object can be used on both the host and device, so it can be used in a
   // CARE_STREAM_LOOP. Raw arrays should not be used inside a CARE_STREAM_LOOP.
   care::host_device_ptr<int> data2(size2);

   // Fill the memory on the host or the device depending on whether the code
   // is compiled with nvcc and if GPU_ACTIVE is defined.
   CARE_STREAM_LOOP(i, 0, size2) {
      data2[i] = i * i;
   } CARE_STREAM_LOOP_END

   // data2 is readable and writeable, but data3 is read only, which can decrease
   // the number of copies between host and device.
   care::host_device_ptr<const int> data3 = data2; // Read only to prevent unnecessary transfers

   // Use a CARE algorithm to fine the max element
   int max2 = care::ArrayMax(data3, size2, std::numeric_limits<int>::lowest());

   // This code illustrates how to check if something ran on the host or the device.
   // In practice it is not necessary. It really just shows that this example compiles
   // and runs fine on the host or device without any changes.
#if defined(__CUDACC__) && defined(GPU_ACTIVE)
   std::cout << "GPU max: ";
#else
   std::cout << "Fallback to CPU max: ";
#endif

   // Print out the max element
   std::cout << max2 << std::endl;

   // Unlike standard library containers, care::host_device_ptr requires an explicit
   // call to free.
   data2.free();

   return 0;
}

