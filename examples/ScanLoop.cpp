// This macro needs to be defined before including care/care.h,
// which allows you to port a file at a time. Without this define,
// LOOP_STREAM will run on the CPU. With this define, LOOP_STREAM
// will run on the GPU only if you built with CUDA enabled. Otherwise,
// if CUDA is disabled, it will default back to running on the host.
#define GPU_ACTIVE

// CARE library header
#include "care/care.h"
#include "care/scan.h"

// Std library headers
#include <iostream>
#include <vector>

int main(int, char**) {
   // Normal code
   int size = 1000;
   std::vector<int> data;

   int threshold = 420;

   for (int i = 0; i < size; ++i) {
      int value = i * i;

      if (value < threshold) { 
         data.push_back(value);
      }
   }

   std::cout << "CPU count: " << data.size() << std::endl;

   // Ported code
   int size2 = 1000;
   care::host_device_ptr<int> data2(size2);

   int count = 0;
   int threshold2 = 420;

   SCAN_LOOP(i, 0, size2, pos, count, i * i < threshold2) {
      data2[pos] = i * i;
   } SCAN_LOOP_END(size2, pos, count)

   data2.reallocate(count);

#if defined(__CUDACC__) && defined(GPU_ACTIVE)
   std::cout << "GPU count: ";
#else
   std::cout << "Fallback to CPU count: ";
#endif

   std::cout << count << std::endl;

   data2.free();

   return 0;
}

