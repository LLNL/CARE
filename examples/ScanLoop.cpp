//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE library header
#include "care/care.h"
#include "care/scan.h"

// Std library headers
#include <iostream>
#include <vector>

int main(int, char**) {
   ///////////////
   // Normal code
   ///////////////

   // Set the size
   int size = 1000;

   // Instantiate a standard library container
   std::vector<int> data;

   // Set a conditional
   int threshold = 420;

   // Only fill the container if the computed value is less than the threshhold
   for (int i = 0; i < size; ++i) {
      int value = i * i;

      if (value < threshold) { 
         data.push_back(value);
      }
   }

   // Print out the number of computed values less than the threshhold
   std::cout << "CPU count: " << data.size() << std::endl;

   ///////////////
   // Ported code
   ///////////////

   // Set the size
   int size2 = 1000;

   // Allocate memory on the host and device using a CARE wrapper around a CHAI container.
   // This object can be used on both the host and device, so it can be used in a
   // LOOP_STREAM. Raw arrays should not be used inside a LOOP_STREAM.
   care::host_device_ptr<int> data2(size2);

   // Set a conditional
   int threshold2 = 420;

   // Set the starting point
   int count = 0;

   // Only fill the container if the computed value is less than the threshhold.
   // The equivalent code on the host is slightly different from using push_back,
   // but has the same effect:
   //
   // int pos = count;
   // for (int i = 0; i < size2; ++i) {
   //    if (i * i < threshhold2) {
   //       data2[pos++] = i * i;
   //    }
   // }
   // count = pos;
   SCAN_LOOP(i, 0, size2, pos, count, i * i < threshold2) {
      data2[pos] = i * i;
   } SCAN_LOOP_END(size2, pos, count)

   // Enough memory had to be preallocated in case all values were less than the
   // threshhold (often preallocation is required for parallel algorithms). Now
   // the array can be shrunk if needed.
   data2.reallocate(count);

   // This code illustrates how to write code specifically for the host or device.
   // This approach should be used sparingly.
#if defined(CARE_GPUCC)
   std::cout << "GPU count: ";
#else
   std::cout << "CPU count: ";
#endif

   // Print out the number of computed values less than the threshhold
   std::cout << count << std::endl;

   // Unlike standard library containers, care::host_device_ptr requires an explicit
   // call to free.
   data2.free();

   return 0;
}

