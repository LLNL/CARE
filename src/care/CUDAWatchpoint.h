//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_CUDA_WATCHPOINT_H_
#define _CARE_CUDA_WATCHPOINT_H_

#ifdef CARE_DEBUG
#if __CUDACC__

// Other library headers
#include "cuda.h"

inline void watchpoint_gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
      fprintf(stderr, "[CARE] WATCHPOINT GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) {
         exit(code);
      }
   }
}

class CUDAWatchpoint {
   public:
      template <typename T>
      static void setOrCheckWatchpoint(T * deviceAddress = nullptr) {
         static T * wp = nullptr;
         static T oldVal = 0;
         // check
         if (!deviceAddress) {
            if (wp) {
               T currentVal;
               watchpoint_gpuAssert(cudaMemcpy(&currentVal, (void *) wp, sizeof(T), cudaMemcpyDeviceToHost), "CUDAWatchpoint", __LINE__, true);
               if (currentVal != oldVal) {
                  printf("[CARE] CUDA Watchpoint %p change detected!\n", wp);
                  oldVal = currentVal;
               }
            }
         }
         // set
         else {
            wp = deviceAddress;
            watchpoint_gpuAssert(cudaMemcpy(&oldVal, (void *) wp, sizeof(T), cudaMemcpyDeviceToHost), "CUDAWatchpoint", __LINE__, true);
         }
      }
};

#endif //__CUDACC__
#endif //CARE_DEBUG

#endif // !defined(_CARE_CUDA_WATCHPOINT_H_)
