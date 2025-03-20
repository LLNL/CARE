//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

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
   // is compiled with nvcc.
   CARE_STREAM_LOOP(i, 0, size2) {
      data2[i] = i * i;
   } CARE_STREAM_LOOP_END

   // data2 is readable and writeable, but data3 is read only, which can decrease
   // the number of copies between host and device.
   care::host_device_ptr<const int> data3 = data2; // Read only to prevent unnecessary transfers

   // Use a CARE algorithm to fine the max element
   int max2 = care::ArrayMax(data3, size2, std::numeric_limits<int>::lowest());

   // This code illustrates how to write code specifically for the host or device.
   // This approach should be used sparingly.
#if defined(CARE_GPUCC)
   std::cout << "GPU max: ";
#else
   std::cout << "CPU max: ";
#endif

   // Print out the max element
   std::cout << max2 << std::endl;

   // Unlike standard library containers, care::host_device_ptr requires an explicit
   // call to free.
   data2.free();

   return 0;
}

