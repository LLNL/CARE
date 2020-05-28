//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// Makes LOOP_REDUCE run on the device
#ifdef __CUDACC__
#define GPU_ACTIVE
#endif

// std library headers
#include <array>

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/array.h"
#include "care/array_utils.h"
#include "care/care.h"

TEST(array_utils, fill_empty)
{
   care::host_ptr<int> a;
   
   care_utils::ArrayFill<int>(a, 0, -12);
}

TEST(array_utils, fill_one)
{
   int temp[1] = {1};
   care::host_ptr<int> a(temp);
   care_utils::ArrayFill<int>(a, 1, -12);

   EXPECT_EQ(a[0], -12);
}

TEST(array_utils, fill_three)
{
   int temp[3] = {1, 2, 3};
   care::host_ptr<int> a(temp);
   care_utils::ArrayFill<int>(a, 3, -12);

   EXPECT_EQ(a[0], -12);
   EXPECT_EQ(a[1], -12);
   EXPECT_EQ(a[2], -12);
}

#ifdef __CUDACC__

// Adapted from CHAI
#define GPU_TEST(X, Y) \
   static void gpu_test_##X##Y(); \
   TEST(X, gpu_test_##Y) { gpu_test_##X##Y(); } \
   static void gpu_test_##X##Y()

GPU_TEST(array, constructor)
{
   care::array<int, 3> a{{1, 2, 3}};

   RAJAReduceMin<bool> passed{true};

   LOOP_REDUCE(i, 0, 1) {
      if (a[0] != 1) {
         passed.min(false);
         return;
      }
      else if (a[1] != 2) {
         passed.min(false);
         return;
      }
      else if (a[2] != 3) {
         passed.min(false);
         return;
      }
   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, write)
{
   care::array<int, 3> a;

   a[0] = 7;
   a[1] = 3;
   a[2] = 6;

   RAJAReduceMin<bool> passed{true};

   LOOP_REDUCE(i, 0, 1) {
      if (a[0] != 7) {
         passed.min(false);
         return;
      }
      else if (a[1] != 3) {
         passed.min(false);
         return;
      }
      else if (a[2] != 6) {
         passed.min(false);
         return;
      }
   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, front)
{
   care::array<int, 2> a{{7, 3}};

   RAJAReduceMin<bool> passed{true};

   LOOP_REDUCE(i, 0, 1) {
      if (a.front() != 7) {
         passed.min(false);
      }
   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);
}

#endif // __CUDACC__

