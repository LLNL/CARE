// This macro needs to be defined before including care/care.h,
// which allows you to port a file at a time. Without this define,
// LOOP_STREAM will run on the CPU. With this define, LOOP_STREAM
// will run on the GPU only if you built with CUDA enabled. Otherwise,
// if CUDA is disabled, it will default back to running on the host.
#define GPU_ACTIVE

// CARE library header
#include "care/care.h"
#include "care/array_utils.h"

// Std library headers
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

int main(int, char**) {
   // Normal code
   int size = 1000;
   std::vector<int> data(size);

   for (int i = 0; i < size; ++i) {
      data[i] = i * i;
   }

   int max = *std::max_element(data.begin(), data.end());
   std::cout << "CPU max: " << max << std::endl;

   // Ported code
   int size2 = 1000;
   care::host_device_ptr<int> data2(size2);

   LOOP_STREAM(i, 0, size2) {
      data2[i] = i * i;
   } LOOP_STREAM_END

   care::host_device_ptr<const int> data3 = data2; // Read only to prevent unnecessary transfers
   int max2 = care_utils::ArrayMax(data3, size2, std::numeric_limits<int>::lowest());

#if defined(__CUDACC__) && defined(GPU_ACTIVE)
   std::cout << "GPU max: ";
#else
   std::cout << "Fallback to CPU max: ";
#endif

   std::cout << max2 << std::endl;

   data2.free();

   return 0;
}

