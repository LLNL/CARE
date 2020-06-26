//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// Makes LOOP_REDUCE run on the device
#if defined(__CUDACC__) || defined(__HIPCC__)
#define GPU_ACTIVE
#endif

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/care.h"
#include "care/host_device_ptr.h"
#include "care/numeric.h"

TEST(numeric, iota)
{
   // Set up
   const int size = 3;
   care::host_device_ptr<int> array(size);

   // Check negative value works
   int offset = -1;
   care::iota(RAJA::seq_exec{}, array, size, offset);

   LOOP_SEQUENTIAL(i, 0, size) {
      EXPECT_EQ(array[i], i + offset);
   } LOOP_SEQUENTIAL_END

   // Check zero value works
   offset = 0;
   care::iota(RAJA::seq_exec{}, array, size, offset);

   LOOP_SEQUENTIAL(i, 0, size) {
      EXPECT_EQ(array[i], i + offset);
   } LOOP_SEQUENTIAL_END

   // Check positive value works
   offset = 1;
   care::iota(RAJA::seq_exec{}, array, size, offset);

   LOOP_SEQUENTIAL(i, 0, size) {
      EXPECT_EQ(array[i], i + offset);
   } LOOP_SEQUENTIAL_END
}

#if defined(__CUDACC__) || defined(__HIPCC__)

// Adapted from CHAI
#define GPU_TEST(X, Y) \
   static void gpu_test_##X##Y(); \
   TEST(X, gpu_test_##Y) { gpu_test_##X##Y(); } \
   static void gpu_test_##X##Y()

GPU_TEST(numeric, iota)
{
   // Set up
   const int size = 3;
   care::host_device_ptr<int> array(size);

   // Check negative value works
   int offset = -1;
#if defined(__CUDACC__)
   care::iota(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#elif defined(__HIPCC__)
   care::iota(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#endif

   RAJAReduceMin<bool> passed{true};

   LOOP_REDUCE(i, 0, size) {
      if (array[i] != i + offset) {
         passed.min(false);
      }
   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);

   // Check zero value works
   offset = 0;
#if defined(__CUDACC__)
   care::iota(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#elif defined(__HIPCC__)
   care::iota(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#endif

   LOOP_REDUCE(i, 0, size) {
      if (array[i] != i + offset) {
         passed.min(false);
      }
   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);

   // Check positive value works
   offset = 1;
#if defined(__CUDACC__)
   care::iota(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#elif defined(__HIPCC__)
   care::iota(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#endif

   LOOP_REDUCE(i, 0, size) {
      if (array[i] != i + offset) {
         passed.min(false);
      }
   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);
}

#endif // __CUDACC__

