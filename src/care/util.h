//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_UTIL_H_
#define _CARE_UTIL_H_

// CARE headers
#include "care/config.h"
#include "care/policies.h"

// Other library headers
#if defined(__CUDACC__)
#include "cuda.h"
#endif

#if defined(CARE_GPUCC)
#if defined(CARE_DEBUG)
#include "care/GPUWatchpoint.h"
#endif
#endif

// Std library headers
#include <cstddef>

/// Whether or not to force CUDA device synchronization after every call to forall
#ifndef FORCE_SYNC

#ifdef CARE_DEBUG
#define FORCE_SYNC 1
#else
#define FORCE_SYNC 0
#endif

#endif // !FORCE_SYNC

namespace care {
#if defined(__CUDACC__)

   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief Checks the return code from CUDA API calls for errors. Prints the
   ///        name of the file and the line number where the CUDA API call occurred
   ///        in the event of an error and exits if abort is true.
   ///
   /// @arg[in] code The return code from CUDA API calls
   /// @arg[in] file The file where the CUDA API call occurred
   /// @arg[in] line The line number where the CUDA API call occurred
   /// @arg[in] abort Whether or not to abort if an error occurred
   ///
   /////////////////////////////////////////////////////////////////////////////////
   inline void gpuAssert(cudaError_t code, const char *file, const int line,
                         const bool abort=true)
   {
      if (code != cudaSuccess) {
         fprintf(stderr, "[CARE] GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
         if (abort) {
            exit(code);
         }
      }
#if defined(CARE_DEBUG)
      GPUWatchpoint::setOrCheckWatchpoint<int>();
#endif
   }

#elif defined(__HIPCC__)

   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief Checks the return code from CUDA API calls for errors. Prints the
   ///        name of the file and the line number where the CUDA API call occurred
   ///        in the event of an error and exits if abort is true.
   ///
   /// @arg[in] code The return code from CUDA API calls
   /// @arg[in] file The file where the CUDA API call occurred
   /// @arg[in] line The line number where the CUDA API call occurred
   /// @arg[in] abort Whether or not to abort if an error occurred
   ///
   /////////////////////////////////////////////////////////////////////////////////
   inline void gpuAssert(hipError_t code, const char *file, const int line,
                         const bool abort=true)
   {
      if (code != hipSuccess) {
         fprintf(stderr, "[CARE] GPUassert: %s %s %d\n", hipGetErrorString(code), file, line);
         if (abort) {
            exit(code);
         }
      }
#if defined(CARE_DEBUG)
      GPUWatchpoint::setOrCheckWatchpoint<int>();
#endif
   }

#else // __CUDACC__

   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief This version of gpuAssert is for CPU builds and does nothing.
   ///
   /// @arg[in] code The return code from CUDA API calls
   /// @arg[in] file The file where the CUDA API call occurred
   /// @arg[in] line The line number where the CUDA API call occurred
   /// @arg[in] abort Whether or not to abort if an error occurred
   ///
   /////////////////////////////////////////////////////////////////////////////////
   inline void gpuAssert(int, const char *, int, bool) { return; }

#endif // __CUDACC__

   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief Convenience function for calling cudaDeviceSynchronize. This overload
   ///        does nothing if not in a CUDA context.
   ///
   /// @arg[in] ExecutionPolicy Used to choose this overload
   /// @arg[in] file The file where the cudaDeviceSynchronize call occurred
   /// @arg[in] line The line number where the cudaDeviceSynchronize call occurred
   /// @arg[in] abort Whether or not to abort if an error occurred
   ///
   /////////////////////////////////////////////////////////////////////////////////
   template <typename ExecutionPolicy>
   inline void cudaDeviceSynchronize(ExecutionPolicy, const char*,
                                     const int, const bool = true) {
   }

#if defined(__CUDACC__)

   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief Convenience function for calling cudaDeviceSynchronize. This overload
   ///        calls cudaDeviceSynchronize if in a CUDA context and FORCE_SYNC is true.
   ///
   /// @arg[in] ExecutionPolicy Used to choose this overload
   /// @arg[in] file The file where the cudaDeviceSynchronize call occurred
   /// @arg[in] line The line number where the cudaDeviceSynchronize call occurred
   /// @arg[in] abort Whether or not to abort if an error occurred
   ///
   /////////////////////////////////////////////////////////////////////////////////
   inline void cudaDeviceSynchronize(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>,
                                     const char* file, const int line,
                                     const bool abort=true) {
#if FORCE_SYNC
      gpuAssert(::cudaDeviceSynchronize(), file, line, abort);
#endif
   }

#elif __HIPCC__

   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief Convenience function for calling cudaDeviceSynchronize. This overload
   ///        calls cudaDeviceSynchronize if in a CUDA context and FORCE_SYNC is true.
   ///
   /// @arg[in] ExecutionPolicy Used to choose this overload
   /// @arg[in] file The file where the cudaDeviceSynchronize call occurred
   /// @arg[in] line The line number where the cudaDeviceSynchronize call occurred
   /// @arg[in] abort Whether or not to abort if an error occurred
   ///
   /////////////////////////////////////////////////////////////////////////////////
   inline void hipDeviceSynchronize(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>,
                                     const char* file, const int line,
                                     const bool abort=true) {
#if FORCE_SYNC
      gpuAssert(::hipDeviceSynchronize(), file, line, abort);
#endif
   }

#endif

/////////////////////////////////////////////////////////////////////////////////
///
/// @author Danny Taller
///
/// @brief Convenience function for calling cuda or hip DeviceSynchronize,
///        depending on the context.
////////////////////////////////////////////////////////////////////////////////
CARE_HOST inline void gpuDeviceSynchronize(const char *fileName, int lineNumber) {
#if defined(__CUDACC__)
   care::gpuAssert( ::cudaDeviceSynchronize(), fileName, lineNumber);
#elif defined(__HIPCC__)
   care::gpuAssert( ::hipDeviceSynchronize(), fileName, lineNumber); 
#else
   (void) fileName;
   (void) lineNumber;
#endif
}

// various GPU wrappers, only needed for GPU compiles
#if defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE

// wrapper for hip/cuda free
CARE_HOST inline void gpuFree(void* buffer) {
#if CARE_ENABLE_GPU_SIMULATION_MODE
   free(buffer);
#elif defined(__HIPCC__)
   gpuAssert(hipFree(buffer), "gpuFree", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaFree(buffer), "gpuFree", __LINE__);
#endif
}

// wrapper for hip/cuda free host
CARE_HOST inline void gpuFreeHost(void* buffer) {
#if CARE_ENABLE_GPU_SIMULATION_MODE
   free(buffer);
#elif defined(__HIPCC__)
   gpuAssert(hipHostFree(buffer), "gpuFreeHost", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaFreeHost(buffer), "gpuFreeHost", __LINE__);
#endif
}

// wrapper for hip/cuda mem copy
CARE_HOST inline void  gpuMemcpy(void* dst, const void* src, size_t count, gpuMemcpyKind kind) {
#if CARE_ENABLE_GPU_SIMULATION_MODE
   memcpy(dst, src, count);
#elif defined(__HIPCC__)
   gpuAssert(hipMemcpy(dst, src, count, kind), "gpuMemcpy", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaMemcpy(dst, src, count, kind), "gpuMemcpy", __LINE__);
#endif
}

// wrapper for hip/cuda malloc
CARE_HOST inline void gpuMalloc(void** devPtr, size_t size) {
#if CARE_ENABLE_GPU_SIMULATION_MODE
   *devPtr = (void*)malloc(size);
#elif defined(__HIPCC__)
   gpuAssert(hipMalloc(devPtr, size), "gpuMalloc", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaMalloc(devPtr, size), "gpuMalloc", __LINE__);
#endif
}

// wrapper for hip/cuda managed malloc
CARE_HOST inline void gpuMallocManaged(void** devPtr, size_t size) {
#if CARE_ENABLE_GPU_SIMULATION_MODE
   *devPtr = (void*)malloc(size);
#elif defined(__HIPCC__)
   gpuAssert(hipMallocManaged(devPtr, size), "gpuMallocManaged", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaMallocManaged(devPtr, size), "gpuMallocManaged", __LINE__);
#endif
}

// wrapper for hip/cuda host alloc
CARE_HOST inline void gpuHostAlloc(void** pHost, size_t size, unsigned int flags) {
#if CARE_ENABLE_GPU_SIMULATION_MODE
   *pHost = (void*)malloc(size);
#elif defined(__HIPCC__)
   gpuAssert(hipHostMalloc(pHost, size, flags), "gpuHostAlloc", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaHostAlloc(pHost, size, flags), "gpuHostAlloc", __LINE__);
#endif
}

#endif // defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE

#if defined(CARE_GPUCC)
// kernel launch
CARE_HOST inline void gpuLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, gpuStream_t stream) {
#if defined(__HIPCC__)
   gpuAssert(hipLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream), "gpuLaunchKernel", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream), "gpuLaunchKernel", __LINE__);
#endif
}

// wrapper for stream synchronize
CARE_HOST inline void gpuStreamSynchronize(gpuStream_t stream) {
#if defined(__HIPCC__)
   gpuAssert(hipStreamSynchronize(stream), "gpuStreamSynchronize", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaStreamSynchronize(stream), "gpuStreamSynchronize", __LINE__);
#endif
}

#endif // defined(CARE_GPUCC)

} // namespace care

#if defined(CARE_GPUCC) && defined(CARE_DEBUG)

/////////////////////////////////////////////////////////////////////////////////
///
/// @author Peter Robinson
///
/// @brief Macro for checking the return code from CUDA API calls for errors. Adds
///        the file and line number where the call occurred for improved debugging.
///
/// @arg[in] code The return code from CUDA API calls
///
/////////////////////////////////////////////////////////////////////////////////
#define care_gpuErrchk(code) { care::gpuAssert((code), __FILE__, __LINE__); }

#else

/////////////////////////////////////////////////////////////////////////////////
///
/// @author Peter Robinson
///
/// @brief In a release build, this version of the macro ensures we do not add
///        the extra overhead of error checking.
///
/// @arg[in] code The return code from CUDA API calls
///
/////////////////////////////////////////////////////////////////////////////////
#define care_gpuErrchk(code) code

#endif // defined(CARE_GPUCC) && defined(CARE_DEBUG)

#endif // !defined(_CARE_UTIL_H_)

