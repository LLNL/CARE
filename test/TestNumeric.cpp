//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/numeric.h"
#include "care/detail/test_utils.h"

#if defined(CARE_GPUCC)
GPU_TEST(forall, Initialization) {
   printf("Initializing\n");
   init_care_for_testing();
   printf("Initialized... Testing care::numeric\n");
}
#endif

TEST(numeric, iota)
{
   // Set up
   const int size = 3;
   care::host_device_ptr<int> array(size);

   // Check negative value works
   int offset = -1;
   care::iota(RAJA::seq_exec{}, array, size, offset);

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      EXPECT_EQ(array[i], i + offset);
   } CARE_SEQUENTIAL_LOOP_END

   // Check zero value works
   offset = 0;
   care::iota(RAJA::seq_exec{}, array, size, offset);

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      EXPECT_EQ(array[i], i + offset);
   } CARE_SEQUENTIAL_LOOP_END

   // Check positive value works
   offset = 1;
   care::iota(RAJA::seq_exec{}, array, size, offset);

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      EXPECT_EQ(array[i], i + offset);
   } CARE_SEQUENTIAL_LOOP_END
}

#if defined(CARE_GPUCC)

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

   CARE_REDUCE_LOOP(i, 0, size) {
      if (array[i] != i + offset) {
         passed.min(false);
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);

   // Check zero value works
   offset = 0;
#if defined(__CUDACC__)
   care::iota(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#elif defined(__HIPCC__)
   care::iota(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#endif

   CARE_REDUCE_LOOP(i, 0, size) {
      if (array[i] != i + offset) {
         passed.min(false);
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);

   // Check positive value works
   offset = 1;
#if defined(__CUDACC__)
   care::iota(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#elif defined(__HIPCC__)
   care::iota(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{}, array, size, offset);
#endif

   CARE_REDUCE_LOOP(i, 0, size) {
      if (array[i] != i + offset) {
         passed.min(false);
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

#endif // __CUDACC__

