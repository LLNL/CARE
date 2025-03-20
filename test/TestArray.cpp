//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

// std library headers
#include <array>

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/array.h"
#include "care/DefaultMacros.h"
#include "care/policies.h"
#include "care/detail/test_utils.h"

#if defined(CARE_GPUCC)
GPU_TEST(array, gpu_initialization) {
   printf("Initializing\n");
   init_care_for_testing();
   printf("Initialized... Testing care::array\n");
}
#endif

TEST(array, constructor)
{
   care::array<int, 3> a{{1, 2, 3}};

   EXPECT_EQ(a[0], 1);
   EXPECT_EQ(a[1], 2);
   EXPECT_EQ(a[2], 3);
}

TEST(array, construct_from_fixed_size_array)
{
   int temp[3] = {1, 2, 3};
   care::array<int, 3> a(temp);

   EXPECT_EQ(a[0], 1);
   EXPECT_EQ(a[1], 2);
   EXPECT_EQ(a[2], 3);
}

TEST(array, construct_from_std_array)
{
   std::array<int, 3> temp{1, 2, 3};
   care::array<int, 3> a(temp);

   EXPECT_EQ(a[0], 1);
   EXPECT_EQ(a[1], 2);
   EXPECT_EQ(a[2], 3);
}

TEST(array, write)
{
   care::array<int, 3> a;

   a[0] = 7;
   a[1] = 3;
   a[2] = 6;

   EXPECT_EQ(a[0], 7);
   EXPECT_EQ(a[1], 3);
   EXPECT_EQ(a[2], 6);
}

TEST(array, front)
{
   care::array<int, 2> a{{7, 3}};
   EXPECT_EQ(7, a.front());
}

TEST(array, back)
{
   care::array<int, 2> a{{7, 3}};
   EXPECT_EQ(3, a.back());
}

TEST(array, data)
{
   care::array<int, 2> a{{6, 2}};
   int* temp = a.data();

   EXPECT_EQ(temp[0], 6);
   EXPECT_EQ(temp[1], 2);
}

TEST(array, empty)
{
   care::array<float, 0> a1;
   EXPECT_TRUE(a1.empty());

   care::array<float, 1> a2;
   EXPECT_FALSE(a2.empty());
}

TEST(array, size)
{
   care::array<float, 0> a1;
   EXPECT_EQ(0, a1.size());

   care::array<float, 4> a2;
   EXPECT_EQ(4, a2.size());
}

TEST(array, fill)
{
   care::array<int, 4> a;
   a.fill(13);

   for (std::size_t i = 0; i < 4; ++i) {
      EXPECT_EQ(13, a[i]);
   }
}

TEST(array, swap)
{
   care::array<int, 3> a1{{1, 1, 1}};
   care::array<int, 3> a2{{5, 5, 5}};

   a1.swap(a2);

   for (std::size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(5, a1[i]);
      EXPECT_EQ(1, a2[i]);
   }
}

TEST(array, equal_to)
{
   care::array<int, 3> a1{{7, 1, 1}};
   care::array<int, 3> a2{{7, 1, 1}};
   care::array<int, 3> a3{{1, 1, 7}};

   EXPECT_TRUE(a1 == a2);
   EXPECT_FALSE(a2 == a3);
}

TEST(array, not_equal_to)
{
   care::array<int, 3> a1{{7, 1, 1}};
   care::array<int, 3> a2{{7, 1, 1}};
   care::array<int, 3> a3{{1, 1, 7}};

   EXPECT_FALSE(a1 != a2);
   EXPECT_TRUE(a2 != a3);
}

TEST(array, less_than)
{
   care::array<int, 3> a1{{5, -2, -1}};
   care::array<int, 3> a2{{6, -1, 0}};
   care::array<int, 3> a3{{7, 0, 0}};
   care::array<int, 3> a4{{8, -1, 1}};

   EXPECT_TRUE(a1 < a2);
   EXPECT_FALSE(a2 < a3);
   EXPECT_FALSE(a3 < a4);
}

TEST(array, less_than_or_equal_to)
{
   care::array<int, 3> a1{{5, -2, -1}};
   care::array<int, 3> a2{{6, -1, 0}};
   care::array<int, 3> a3{{6, -1, 0}};
   care::array<int, 3> a4{{5, -1, 0}};

   EXPECT_TRUE(a1 <= a2);
   EXPECT_TRUE(a2 <= a3);
   EXPECT_FALSE(a3 <= a4);
}

TEST(array, greater_than)
{
   care::array<int, 3> a1{{5, -2, -1}};
   care::array<int, 3> a2{{4, -3, -2}};
   care::array<int, 3> a3{{3, -4, -2}};
   care::array<int, 3> a4{{2, -5, -1}};

   EXPECT_TRUE(a1 > a2);
   EXPECT_FALSE(a2 > a3);
   EXPECT_FALSE(a3 > a4);
}

TEST(array, greater_than_or_equal_to)
{
   care::array<int, 3> a1{{5, -2, -1}};
   care::array<int, 3> a2{{4, -3, -2}};
   care::array<int, 3> a3{{4, -3, -2}};
   care::array<int, 3> a4{{3, -4, -1}};

   EXPECT_TRUE(a1 >= a2);
   EXPECT_TRUE(a2 >= a3);
   EXPECT_FALSE(a3 >= a4);
}

#if defined(CARE_GPUCC)

GPU_TEST(array, constructor)
{
   care::array<int, 3> a{{1, 2, 3}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
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
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, write)
{
   care::array<int, 3> a;

   a[0] = 7;
   a[1] = 3;
   a[2] = 6;

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
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
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, front)
{
   care::array<int, 2> a{{7, 3}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      if (a.front() != 7) {
         passed.min(false);
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, back)
{
   care::array<int, 2> a{{7, 3}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      if (a.back() != 3) {
         passed.min(false);
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, data)
{
   care::array<int, 2> a{{6, 2}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      int const * temp = a.data();

      if (temp[0] != 6) {
         passed.min(false);
         return;
      }
      else if (temp[1] != 2) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, empty)
{
   care::array<float, 0> a1{};
   care::array<float, 1> a2{{1}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      if (!a1.empty()) {
         passed.min(false);
         return;
      }
      else if (a2.empty()) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, size)
{
   care::array<float, 0> a1{};
   care::array<float, 4> a2{{1, 2, 3, 4}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      if (a1.size() != 0) {
         passed.min(false);
         return;
      }
      else if (a2.size() != 4) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, fill)
{
   care::array<int, 4> a;
   a.fill(13);

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 4) {
      if (a[i] != 13) {
         passed.min(false);
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, swap)
{
   care::array<int, 3> a1{{1, 1, 1}};
   care::array<int, 3> a2{{5, 5, 5}};

   a1.swap(a2);

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 3) {
      if (a1[i] != 5) {
         passed.min(false);
         return;
      }
      else if (a2[i] != 1) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, equal_to)
{
   care::array<int, 3> a1{{7, 1, 1}};
   care::array<int, 3> a2{{7, 1, 1}};
   care::array<int, 3> a3{{1, 1, 7}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 3) {
      if (!(a1 == a2)) {
         passed.min(false);
         return;
      }
      else if (a2 == a3) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, not_equal_to)
{
   care::array<int, 3> a1{{7, 1, 1}};
   care::array<int, 3> a2{{7, 1, 1}};
   care::array<int, 3> a3{{1, 1, 7}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 3) {
      if (a1 != a2) {
         passed.min(false);
         return;
      }
      else if (!(a2 != a3)) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, less_than)
{
   care::array<int, 3> a1{{5, -2, -1}};
   care::array<int, 3> a2{{6, -1, 0}};
   care::array<int, 3> a3{{7, 0, 0}};
   care::array<int, 3> a4{{8, -1, 1}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 3) {
      if (!(a1 < a2)) {
         passed.min(false);
         return;
      }
      else if (a2 < a3) {
         passed.min(false);
         return;
      }
      else if (a3 < a4) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, less_than_or_equal_to)
{
   care::array<int, 3> a1{{5, -2, -1}};
   care::array<int, 3> a2{{6, -1, 0}};
   care::array<int, 3> a3{{6, -1, 0}};
   care::array<int, 3> a4{{5, -1, 0}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 3) {
      if (!(a1 <= a2)) {
         passed.min(false);
         return;
      }
      else if (!(a2 <= a3)) {
         passed.min(false);
         return;
      }
      else if (a3 <= a4) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, greater_than)
{
   care::array<int, 3> a1{{5, -2, -1}};
   care::array<int, 3> a2{{4, -3, -2}};
   care::array<int, 3> a3{{3, -4, -2}};
   care::array<int, 3> a4{{2, -5, -1}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 3) {
      if (!(a1 > a2)) {
         passed.min(false);
         return;
      }
      else if (a2 > a3) {
         passed.min(false);
         return;
      }
      else if (a3 > a4) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array, greater_than_or_equal_to)
{
   care::array<int, 3> a1{{5, -2, -1}};
   care::array<int, 3> a2{{4, -3, -2}};
   care::array<int, 3> a3{{4, -3, -2}};
   care::array<int, 3> a4{{3, -4, -1}};

   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 3) {
      if (!(a1 >= a2)) {
         passed.min(false);
         return;
      }
      else if (!(a2 >= a3)) {
         passed.min(false);
         return;
      }
      else if (a3 >= a4) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);
}

#endif // CARE_GPUCC

