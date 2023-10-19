//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

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

TEST(array, initialization)
{
   care::array<int, 3> a{1, 2, 10};

   EXPECT_EQ(a[0], 1);
   EXPECT_EQ(a[1], 2);
   EXPECT_EQ(a[2], 10);
}

TEST(array, copy_initialization)
{
   care::array<int, 3> a = {10, 2, 1};

   EXPECT_EQ(a[0], 10);
   EXPECT_EQ(a[1], 2);
   EXPECT_EQ(a[2], 1);
}

TEST(array, copy_construct)
{
   care::array<int, 3> a{1, 2, 10};
   care::array<int, 3> b{a};

   EXPECT_EQ(b[0], 1);
   EXPECT_EQ(b[1], 2);
   EXPECT_EQ(b[2], 10);
}

TEST(array, copy_assignment)
{
   care::array<int, 3> a{1, 2, 10};
   a = care::array<int, 3>{3, 4, 6};

   EXPECT_EQ(a[0], 3);
   EXPECT_EQ(a[1], 4);
   EXPECT_EQ(a[2], 6);
}

TEST(array, at)
{
   care::array<int, 1> a = {-4};

   int resultAt0 = -1;
   int resultAt1 = -1;
   bool exception = false;

   try {
      resultAt0 = a.at(0);
      resultAt1 = a.at(1);
   }
   catch (std::out_of_range e) {
      exception = true;
   }

   EXPECT_EQ(resultAt0, -4);
   EXPECT_EQ(resultAt1, -1);
   EXPECT_TRUE(exception);
}

TEST(array, access)
{
   care::array<int, 2> a = {1, 12};
   a[0] = 3;
   EXPECT_EQ(a[0], 3);

   const care::array<int, 2>& b = a;
   EXPECT_EQ(b[0], 3);
}

TEST(array, front)
{
   care::array<int, 2> a = {1, 12};
   a.front() = 3;
   EXPECT_EQ(a[0], 3);

   const care::array<int, 2>& b = a;
   EXPECT_EQ(b.front(), 3);
   EXPECT_EQ(b[0], 3);
}

TEST(array, back)
{
   care::array<int, 2> a = {1, 12};
   a.back() = 5;
   EXPECT_EQ(a[1], 5);

   const care::array<int, 2>& b = a;
   EXPECT_EQ(b.back(), 5);
   EXPECT_EQ(b[1], 5);
}

TEST(array, data)
{
   care::array<int, 2> a = {1, 12};
   int* a_data = a.data();
   EXPECT_EQ(a_data[0], a[0]);
   EXPECT_EQ(a_data[1], a[1]);

   const care::array<int, 2>& b = a;
   const int* b_data = b.data();
   EXPECT_EQ(b_data[0], b[0]);
   EXPECT_EQ(b_data[1], b[1]);
}

TEST(array, begin)
{
   care::array<int, 2> a = {1, 12};
   auto a_it = a.begin();
   *a_it = 4;
   EXPECT_EQ(a[0], 4);
   *(++a_it) = 6;
   EXPECT_EQ(a[1], 6);

   const care::array<int, 2>& b = a;
   auto b_it = b.begin();
   EXPECT_EQ(*b_it, b[0]);
   EXPECT_EQ(*(++b_it), b[1]);
}

TEST(array, cbegin)
{
   care::array<int, 2> a = {1, 12};
   auto a_it = a.cbegin();
   EXPECT_EQ(*a_it, a[0]);
   EXPECT_EQ(*(++a_it), a[1]);

   const care::array<int, 2>& b = a;
   auto b_it = b.begin();
   EXPECT_EQ(*b_it, b[0]);
   EXPECT_EQ(*(++b_it), b[1]);
}

TEST(array, end)
{
   care::array<int, 2> a = {1, 12};
   auto a_it = a.end();
   *(--a_it) = 4;
   EXPECT_EQ(a[1], 4);
   *(--a_it) = 6;
   EXPECT_EQ(a[0], 6);

   const care::array<int, 2>& b = a;
   auto b_it = b.end();
   EXPECT_EQ(*(--b_it), b[1]);
   EXPECT_EQ(*(--b_it), b[0]);
}

TEST(array, cend)
{
   care::array<int, 2> a = {1, 12};
   auto a_it = a.cend();
   EXPECT_EQ(*(--a_it), a[1]);
   EXPECT_EQ(*(--a_it), a[0]);

   const care::array<int, 2>& b = a;
   auto b_it = b.cend();
   EXPECT_EQ(*(--b_it), b[1]);
   EXPECT_EQ(*(--b_it), b[0]);
}

TEST(array, empty)
{
   care::array<double, 0> a{};
   EXPECT_TRUE(a.empty());

   care::array<double, 1> b{1.0};
   EXPECT_FALSE(b.empty());
}

TEST(array, size)
{
   care::array<double, 0> a{};
   EXPECT_EQ(a.size(), 0);

   care::array<double, 2> b{1.0, 3.0};
   EXPECT_EQ(b.size(), 2);
}

TEST(array, max_size)
{
   care::array<double, 0> a{};
   EXPECT_EQ(a.max_size(), 0);

   care::array<double, 2> b{1.0, 3.0};
   EXPECT_EQ(b.max_size(), 2);
}

TEST(array, fill)
{
   care::array<int, 3> a{1, 2, 3};
   a.fill(0);

   for (size_t i = 0; i < 3; ++i) {
      EXPECT_EQ(a[i], 0);
   }
}

TEST(array, swap)
{
   care::array<int, 2> a{1, 2};
   care::array<int, 2> b{3, 4};

   a.swap(b);

   EXPECT_EQ(a[0], 3);
   EXPECT_EQ(a[1], 4);

   EXPECT_EQ(b[0], 1);
   EXPECT_EQ(b[1], 2);
}

TEST(array, equal)
{
   care::array<int, 2> a{1, 2};
   care::array<int, 2> b{1, 3};

   EXPECT_TRUE(a == a);
   EXPECT_FALSE(a == b);
}

TEST(array, not_equal)
{
   care::array<int, 2> a{1, 2};
   care::array<int, 2> b{1, 3};

   EXPECT_TRUE(a != b);
   EXPECT_FALSE(a != a);
}

TEST(array, less_than)
{
   care::array<int, 2> a{1, 2};
   care::array<int, 2> b{1, 3};

   EXPECT_TRUE(a < b);
   EXPECT_FALSE(b < a);
}

TEST(array, less_than_or_equal)
{
   care::array<int, 2> a{1, 2};
   care::array<int, 2> b{1, 3};

   EXPECT_TRUE(a <= a);
   EXPECT_TRUE(a <= b);
   EXPECT_FALSE(b <= a);
}

TEST(array, greater_than)
{
   care::array<int, 2> a{1, 2};
   care::array<int, 2> b{1, 3};

   EXPECT_TRUE(b > a);
   EXPECT_FALSE(a > b);
}

TEST(array, greater_than_or_equal)
{
   care::array<int, 2> a{1, 2};
   care::array<int, 2> b{1, 3};

   EXPECT_TRUE(a >= a);
   EXPECT_TRUE(b >= a);
   EXPECT_FALSE(a >= b);
}

TEST(array, get_lvalue_reference)
{
   care::array<int, 2> a = {1, 12};
   care::get<0>(a) = 3;
   EXPECT_EQ(a[0], 3);

   const care::array<int, 2>& b = a;
   EXPECT_EQ(care::get<0>(b), 3);
}

TEST(array, get_rvalue_reference)
{
   care::array<int, 2> a = {1, 12};
   int&& a0 = care::get<0>(care::move(a));
   EXPECT_EQ(a0, 1);
   EXPECT_EQ(a[0], 1);

   const care::array<int, 2> b{6, 8};
   const int&& b1 = care::get<1>(care::move(b));
   EXPECT_EQ(b1, 8);
   EXPECT_EQ(b[1], 8);
}

TEST(array, generic_swap)
{
   care::array<int, 2> a{1, 2};
   care::array<int, 2> b{3, 4};

   care::swap(a, b);

   EXPECT_EQ(a[0], 3);
   EXPECT_EQ(a[1], 4);

   EXPECT_EQ(b[0], 1);
   EXPECT_EQ(b[1], 2);
}

TEST(array, to_array)
{
   int temp[3] = {1, 2, 10};

   care::array<int, 3> a = care::to_array(temp);
   EXPECT_EQ(a[0], 1);
   EXPECT_EQ(a[1], 2);
   EXPECT_EQ(a[2], 10);

   care::array<int, 3> b = care::to_array(care::move(temp));
   EXPECT_EQ(b[0], 1);
   EXPECT_EQ(b[1], 2);
   EXPECT_EQ(b[2], 10);
}

TEST(array, tuple_size)
{
   constexpr std::size_t size = std::tuple_size<care::array<double, 7>>::value;
   constexpr std::size_t size_v = std::tuple_size_v<care::array<double, 11>>;

   EXPECT_EQ(size, 7);
   EXPECT_EQ(size_v, 11);
}

TEST(array, tuple_element)
{
   constexpr bool element0 = std::is_same_v<double, std::tuple_element_t<0, care::array<double, 5>>>;
   constexpr bool element4 = std::is_same_v<double, std::tuple_element_t<4, care::array<double, 5>>>;

   EXPECT_TRUE(element0);
   EXPECT_TRUE(element4);
}

TEST(array, structured_binding)
{
   care::array<int, 2> a{-1, 1};
   auto& [a0, a1] = a;
   EXPECT_EQ(a0, -1);
   EXPECT_EQ(a1, 1);

   a1 = 3;
   EXPECT_EQ(a[1], 3);
}

TEST(array, deduction_guide)
{
   care::array a{-1, 1};
   EXPECT_EQ(a[0], -1);
   EXPECT_EQ(a[1], 1);
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

