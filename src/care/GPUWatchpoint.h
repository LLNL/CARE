//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_GPU_WATCHPOINT_H_
#define _CARE_GPU_WATCHPOINT_H_

#ifdef CARE_DEBUG
#if defined(CARE_GPUCC)

#if defined(__CUDACC__)
// Other library headers
#include "cuda.h"
#define care_watchpoint_err_t cudaError_t
#else
#include "hip/hip_runtime.h"
#define care_watchpoint_err_t hipError_t
#endif

inline void watchpoint_gpuAssert(care_watchpoint_err_t code, const char *file, int line, bool abort=true)
{
   if (code != gpuSuccess) {
#if defined(__CUDACC__)
      fprintf(stderr, "[CARE] WATCHPOINT GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
#elif defined(__HIPCC__)
      fprintf(stderr, "[CARE] WATCHPOINT GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
#endif
      if (abort) {
         exit(code);
      }
   }
}

class GPUWatchpoint {
   public:
      template <typename T>
      static void setOrCheckWatchpoint(T * deviceAddress = nullptr) {
         static T * wp = nullptr;
         static T oldVal = 0;
         // check
         if (!deviceAddress) {
            if (wp) {
               T currentVal;
#if defined(__CUDACC__)
               watchpoint_gpuAssert(cudaMemcpy(&currentVal, (void *) wp, sizeof(T), cudaMemcpyDeviceToHost), "GPUWatchpoint", __LINE__, true);
#elif defined(__HIPCC__)
               watchpoint_gpuAssert(hipMemcpy(&currentVal, (void *) wp, sizeof(T),  hipMemcpyDeviceToHost), "GPUWatchpoint", __LINE__, true);
#endif               
               if (currentVal != oldVal) {
                  printf("[CARE] GPU Watchpoint %p change detected!\n", wp);
                  oldVal = currentVal;
               }
            }
         }
         // set
         else {
            wp = deviceAddress;
#if defined(__CUDACC__)
            watchpoint_gpuAssert(cudaMemcpy(&oldVal, (void *) wp, sizeof(T), cudaMemcpyDeviceToHost), "GPUWatchpoint", __LINE__, true);
#elif defined(__HIPCC__)
            watchpoint_gpuAssert(hipMemcpy(&oldVal, (void *) wp, sizeof(T),  hipMemcpyDeviceToHost), "GPUWatchpoint", __LINE__, true);
#endif         
         }
      }
};

#endif //CARE_GPUCC
#endif //CARE_DEBUG

#endif // !defined(_CARE_GPU_WATCHPOINT_H_)
