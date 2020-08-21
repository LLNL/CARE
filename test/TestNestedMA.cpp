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
   CARE_STREAM_LOOP(i, 0, n) {
      arr[i] = val;
   } CARE_STREAM_LOOP_END
}

// This initialization pattern works fine
GPU_TEST(nestedMA, gpu_init0)
{
   const int N = 1 ;
   const int M = 1 ;
   care::host_device_ptr<care::host_device_ptr<int>> nestedMA ;

   memAlloc(N, "test", &nestedMA) ;

   CARE_SEQUENTIAL_LOOP(i, 0, N) {
      nestedMA[i] = nullptr ;
   } CARE_SEQUENTIAL_LOOP_END

   std::string name("test") ;
   CARE_SEQUENTIAL_REF_LOOP(i, 0, N, name) {
      name = name + "_" ;
      memAlloc(M, name.c_str(), &nestedMA[i]) ;
      ArrayFill<int>(nestedMA[i], M, 0) ;
   } CARE_SEQUENTIAL_REF_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, N) {
      for (int j = 0 ; j < M ; ++j) {
         nestedMA[i][j] = 1 ;
      }
   } CARE_SEQUENTIAL_LOOP_END

}

// This initialization pattern fails on the GPU build
GPU_TEST(nestedMA, gpu_init)
{
   const int N = 1 ;
   const int M = 1 ;
   care::host_device_ptr<care::host_device_ptr<int>> nestedMA ;

   memAlloc(N, "test", &nestedMA) ;

   CARE_SEQUENTIAL_LOOP(i, 0, N) {
      nestedMA[i] = nullptr ;
   } CARE_SEQUENTIAL_LOOP_END

   std::string name("test") ;
   CARE_SEQUENTIAL_REF_LOOP(i, 0, N, name) {
      name = name + "_" ;
      memAlloc(M, name.c_str(), &nestedMA[i]) ;
      ArrayFill<int>(nestedMA[i], M, 0) ;
   } CARE_SEQUENTIAL_REF_LOOP_END

   CARE_STREAM_LOOP(i, 0, N) {
      if (nestedMA) {}
      (void) i; // quiet compiler
   } CARE_STREAM_LOOP_END

   GPU_FAIL(
      CARE_SEQUENTIAL_LOOP(i, 0, N) {
         for (int j = 0 ; j < M ; ++j) {
            nestedMA[i][j] = 1 ;
         }
      } CARE_SEQUENTIAL_LOOP_END
   ) ;
}

// This initialization pattern works fine
GPU_TEST(nestedMA, cpu_init0)
{
   const int N = 1 ;
   const int M = 1 ;
   care::host_device_ptr<care::host_device_ptr<int>> nestedMA ;

   memAlloc(N, "test", &nestedMA) ;

   CARE_SEQUENTIAL_LOOP(i, 0, N) {
      nestedMA[i] = nullptr ;
   } CARE_SEQUENTIAL_LOOP_END

   std::string name("test") ;
   CARE_SEQUENTIAL_REF_LOOP(i, 0, N, name) {
      name = name + "_" ;
      memAlloc(M, name.c_str(), &nestedMA[i]) ;
      for (int j = 0 ; j < M ; ++j) {
         nestedMA[i].set(j, 0) ;
      }
   } CARE_SEQUENTIAL_REF_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, N) {
      for (int j = 0 ; j < M ; ++j) {
         nestedMA[i][j] = 1 ;
      }
   } CARE_SEQUENTIAL_LOOP_END

}
// This initialization pattern fails on the GPU build
GPU_TEST(nestedMA, cpu_init)
{
   const int N = 1 ;
   const int M = 1 ;
   care::host_device_ptr<care::host_device_ptr<int>> nestedMA ;

   memAlloc(N, "test", &nestedMA) ;

   CARE_SEQUENTIAL_LOOP(i, 0, N) {
      nestedMA[i] = nullptr ;
   } CARE_SEQUENTIAL_LOOP_END

   std::string name("test") ;
   CARE_SEQUENTIAL_REF_LOOP(i, 0, N, name) {
      name = name + "_" ;
      memAlloc(M, name.c_str(), &nestedMA[i]) ;
      for (int j = 0 ; j < M ; ++j) {
         nestedMA[i].set(j, 0) ;
      }
   } CARE_SEQUENTIAL_REF_LOOP_END

   CARE_STREAM_LOOP(i, 0, N) {
      if (nestedMA) {}
      (void) i; // quiet compiler
   } CARE_STREAM_LOOP_END

   GPU_FAIL(
      CARE_SEQUENTIAL_LOOP(i, 0, N) {
         for (int j = 0 ; j < M ; ++j) {
            nestedMA[i][j] = 1 ;
         }
      } CARE_SEQUENTIAL_LOOP_END
   ) ;

}

