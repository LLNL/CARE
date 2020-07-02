//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

#if defined(__GPUCC__)
#define GPU_ACTIVE
#endif

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/care.h"

#if defined(__GPUCC__) && GTEST_HAS_DEATH_TEST
// This asserts a crash on the GPU, but does not mark gtest as passing.
#define GPU_FAIL(code) ASSERT_DEATH(code, "")
#else
#define GPU_FAIL(code) code
#endif

// This makes it so we can use device lambdas from within a GPU_TEST
#define GPU_TEST(X, Y) static void gpu_test_ ## X_ ## Y(); \
   TEST(X, Y) { gpu_test_ ## X_ ## Y(); } \
   static void gpu_test_ ## X_ ## Y()

template<typename T>
void memAlloc(size_t size, char const * name, care::host_device_ptr<T> *ptr)
{
   *ptr = care::host_device_ptr<T>(size, name);
}

template <typename T>
inline void ArrayFill(care::host_device_ptr<T> arr, int n, T val) {
   LOOP_STREAM(i, 0, n) {
      arr[i] = val;
   } LOOP_STREAM_END
}

// This initialization pattern works fine
GPU_TEST(nestedMA, gpu_init0)
{
   const int N = 1 ;
   const int M = 1 ;
   care::host_device_ptr<care::host_device_ptr<int>> nestedMA ;

   memAlloc(N, "test", &nestedMA) ;

   LOOP_SEQUENTIAL(i, 0, N) {
      nestedMA[i] = nullptr ;
   } LOOP_SEQUENTIAL_END

   std::string name("test") ;
   LOOP_SEQUENTIAL_REF(i, 0, N, name) {
      name = name + "_" ;
      memAlloc(M, name.c_str(), &nestedMA[i]) ;
      ArrayFill<int>(nestedMA[i], M, 0) ;
   } LOOP_SEQUENTIAL_REF_END

   LOOP_SEQUENTIAL(i, 0, N) {
      for (int j = 0 ; j < M ; ++j) {
         nestedMA[i][j] = 1 ;
      }
   } LOOP_SEQUENTIAL_END

}

// This initialization pattern fails on the GPU build
GPU_TEST(nestedMA, gpu_init)
{
   const int N = 1 ;
   const int M = 1 ;
   care::host_device_ptr<care::host_device_ptr<int>> nestedMA ;

   memAlloc(N, "test", &nestedMA) ;

   LOOP_SEQUENTIAL(i, 0, N) {
      nestedMA[i] = nullptr ;
   } LOOP_SEQUENTIAL_END

   std::string name("test") ;
   LOOP_SEQUENTIAL_REF(i, 0, N, name) {
      name = name + "_" ;
      memAlloc(M, name.c_str(), &nestedMA[i]) ;
      ArrayFill<int>(nestedMA[i], M, 0) ;
   } LOOP_SEQUENTIAL_REF_END

   LOOP_STREAM(i, 0, N) {
      if (nestedMA) {}
      (void) i; // quiet compiler
   } LOOP_STREAM_END

   GPU_FAIL(
      LOOP_SEQUENTIAL(i, 0, N) {
         for (int j = 0 ; j < M ; ++j) {
            nestedMA[i][j] = 1 ;
         }
      } LOOP_SEQUENTIAL_END
   ) ;
}

// This initialization pattern works fine
GPU_TEST(nestedMA, cpu_init0)
{
   const int N = 1 ;
   const int M = 1 ;
   care::host_device_ptr<care::host_device_ptr<int>> nestedMA ;

   memAlloc(N, "test", &nestedMA) ;

   LOOP_SEQUENTIAL(i, 0, N) {
      nestedMA[i] = nullptr ;
   } LOOP_SEQUENTIAL_END

   std::string name("test") ;
   LOOP_SEQUENTIAL_REF(i, 0, N, name) {
      name = name + "_" ;
      memAlloc(M, name.c_str(), &nestedMA[i]) ;
      for (int j = 0 ; j < M ; ++j) {
         nestedMA[i].set(j, 0) ;
      }
   } LOOP_SEQUENTIAL_REF_END

   LOOP_SEQUENTIAL(i, 0, N) {
      for (int j = 0 ; j < M ; ++j) {
         nestedMA[i][j] = 1 ;
      }
   } LOOP_SEQUENTIAL_END

}
// This initialization pattern fails on the GPU build
GPU_TEST(nestedMA, cpu_init)
{
   const int N = 1 ;
   const int M = 1 ;
   care::host_device_ptr<care::host_device_ptr<int>> nestedMA ;

   memAlloc(N, "test", &nestedMA) ;

   LOOP_SEQUENTIAL(i, 0, N) {
      nestedMA[i] = nullptr ;
   } LOOP_SEQUENTIAL_END

   std::string name("test") ;
   LOOP_SEQUENTIAL_REF(i, 0, N, name) {
      name = name + "_" ;
      memAlloc(M, name.c_str(), &nestedMA[i]) ;
      for (int j = 0 ; j < M ; ++j) {
         nestedMA[i].set(j, 0) ;
      }
   } LOOP_SEQUENTIAL_REF_END

   LOOP_STREAM(i, 0, N) {
      if (nestedMA) {}
      (void) i; // quiet compiler
   } LOOP_STREAM_END

   GPU_FAIL(
      LOOP_SEQUENTIAL(i, 0, N) {
         for (int j = 0 ; j < M ; ++j) {
            nestedMA[i][j] = 1 ;
         }
      } LOOP_SEQUENTIAL_END
   ) ;

}

