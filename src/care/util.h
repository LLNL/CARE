//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_UTIL_H_
#define _CARE_UTIL_H_

// CARE headers
#include "care/config.h"
#include "care/policies.h"

// Other library headers
#if defined(__CUDACC__)
#include "cuda.h"
#if defined(CARE_DEBUG)
#include "care/CUDAWatchpoint.h"
#endif
#endif

// Std library headers
#include <iostream>
#include <type_traits>

/// Whether or not to force CUDA device synchronization after every call to forall
#ifndef FORCE_SYNC

#ifdef CARE_DEBUG
#define FORCE_SYNC 1
#else
#define FORCE_SYNC 0
#endif

#endif // !FORCE_SYNC

namespace care {
   namespace detail {
      // c++17 feature
      // https://en.cppreference.com/w/cpp/types/void_t
      template< class... >
      using void_t = void;

      // Experimental c++ features

      // https://en.cppreference.com/w/cpp/experimental/nonesuch
      struct nonesuch {
         ~nonesuch() = delete;
         nonesuch(nonesuch const&) = delete;
         void operator=(nonesuch const&) = delete;
      };

      // https://en.cppreference.com/w/cpp/experimental/is_detected
      template <class Default, class AlwaysVoid,
                template<class...> class Op, class... Args>
      struct detector {
         using value_t = std::false_type;
         using type = Default;
      };
       
      template <class Default, template<class...> class Op, class... Args>
      struct detector<Default, void_t<Op<Args...>>, Op, Args...> {
         using value_t = std::true_type;
         using type = Op<Args...>;
      };

      template <template<class...> class Op, class... Args>
      using is_detected = typename detail::detector<nonesuch, void, Op, Args...>::value_t;
       
      template <template<class...> class Op, class... Args>
      using detected_t = typename detail::detector<nonesuch, void, Op, Args...>::type;
       
      template <class Default, template<class...> class Op, class... Args>
      using detected_or = detail::detector<Default, void, Op, Args...>;
   } // namespace detail

   // Taken and modified from https://nyorain.github.io/cpp-valid-expression.html
   // Also inspired by https://www.fluentcpp.com/2017/06/06/using-tostring-custom-types-cpp/
   template <typename T>
   using stream_insertion_t = decltype(std::cout << std::declval<T>());

   template <typename T,
             typename std::enable_if<detail::is_detected<stream_insertion_t, T>::value, int>::type = 0>
   inline void print(std::ostream& os, const T& obj)
   {
      os << obj;
   }

   template <typename T,
             typename std::enable_if<!detail::is_detected<stream_insertion_t, T>::value, int>::type = 0>
   inline void print(std::ostream& os, const T& obj)
   {
      os << "operator<< is not supported for type " << typeid(obj).name();
   }

   template <typename T,
             typename std::enable_if<detail::is_detected<stream_insertion_t, T>::value, int>::type = 0>
   inline void print(std::ostream& os, const T* array, size_t size)
   {
      // Write out size
      os << "[CARE] Size: " << size << std::endl;

      // Write out elements
      const int maxDigits = strlen(std::to_string(size).c_str());

      for (size_t i = 0; i < size; ++i) {
         const int digits = strlen(std::to_string(i).c_str());
         const int numSpaces = maxDigits - digits;

         // Write out index
         os << "[CARE] Index: ";

         for (int j = 0; j < numSpaces; ++j) {
            os << " ";
         }

         os << i;

         // Write out white space between index and value
         os << "    ";

         // Write out value
         os << "Value: " << array[i] << std::endl;
      }

      os << std::endl;
   }

   template <typename T,
             typename std::enable_if<!detail::is_detected<stream_insertion_t, T>::value, int>::type = 0>
   inline void print(std::ostream& os, const T* /* array */, size_t size)
   {
      // Write out size
      os << "[CARE] Size: " << size << std::endl;

      // TODO: Decide if we should write out the bytes
      os << "operator<< is not supported for type " << typeid(T).name() << std::endl;
   }

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
      CUDAWatchpoint::setOrCheckWatchpoint<int>();
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
      CUDAWatchpoint::setOrCheckWatchpoint<int>();
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

#ifdef __CUDACC__

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
#if defined(__CUDACC__)
   inline cudaError_t gpuDeviceSynchronize() {
      return ::cudaDeviceSynchronize();
   }
#elif defined(__HIPCC__)
   inline hipError_t gpuDeviceSynchronize() {
      return ::hipDeviceSynchronize();
   }
#else
   inline int gpuDeviceSynchronize() {
      return 0;
   }
#endif

// various GPU wrappers, only needed for GPU compiles
#if defined (CARE_GPUCC)

// wrapper for hip/cuda free
CARE_HOST inline void gpuFree(void* buffer) {
#if defined(__HIPCC__)
   gpuAssert(hipFree(buffer), "gpuFree", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaFree(buffer), "gpuFree", __LINE__);
#endif
}

// wrapper for hip/cuda free host
CARE_HOST inline void gpuFreeHost(void* buffer) {
#if defined(__HIPCC__)
   gpuAssert(hipHostFree(buffer), "gpuFreeHost", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaFreeHost(buffer), "gpuFreeHost", __LINE__);
#endif
}

// wrapper for hip/cuda mem copy
CARE_HOST inline void  gpuMemcpy(void* dst, const void* src, size_t count, gpuMemcpyKind kind) {
#if defined(__HIPCC__)
   gpuAssert(hipMemcpy(dst, src, count, kind), "gpuMemcpy", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaMemcpy(dst, src, count, kind), "gpuMemcpy", __LINE__);
#endif
}

// wrapper for hip/cuda malloc
CARE_HOST inline void gpuMalloc(void** devPtr, size_t size) {
#if defined(__HIPCC__)
   gpuAssert(hipMalloc(devPtr, size), "gpuMalloc", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaMalloc(devPtr, size), "gpuMalloc", __LINE__);
#endif
}

// wrapper for hip/cuda managed malloc
CARE_HOST inline void gpuMallocManaged(void** devPtr, size_t size) {
#if defined(__HIPCC__)
   gpuAssert(hipMallocManaged(devPtr, size), "gpuMallocManaged", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaMallocManaged(devPtr, size), "gpuMallocManaged", __LINE__);
#endif
}

// wrapper for hip/cuda host alloc
CARE_HOST inline void gpuHostAlloc(void** pHost, size_t size, unsigned int flags) {
#if defined(__HIPCC__)
   gpuAssert(hipHostMalloc(pHost, size, flags), "gpuHostAlloc", __LINE__);
#elif defined(__CUDACC__)
   gpuAssert(cudaHostAlloc(pHost, size, flags), "gpuHostAlloc", __LINE__);
#endif
}

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

#endif // #if defined (CARE_GPUCC)

} // namespace care

#if defined(CARE_GPUCC) && defined(GPU_ACTIVE) && defined(CARE_DEBUG)

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

#endif // defined(CARE_GPUCC) && defined(GPU_ACTIVE) && defined(CARE_DEBUG)

#endif // !defined(_CARE_UTIL_H_)

