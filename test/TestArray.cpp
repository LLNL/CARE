//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

// care headers
#include "care/array.h"
#include "care/DefaultMacros.h"
#include "care/policies.h"
#include "care/detail/test_utils.h"

// other library headers
#include "gtest/gtest.h"


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

#if 0
TEST(array, deduction_guide)
{
   care::array a{-1, 1};
   EXPECT_EQ(a[0], -1);
   EXPECT_EQ(a[1], 1);
}
#endif

#if defined(CARE_GPUCC)

GPU_TEST(array, initialization)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 3> a{1, 2, 10};

      passed.min(a[0] == 1);
      passed.min(a[1] == 2);
      passed.min(a[2] == 10);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, copy_initialization)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 3> a = {1, 2, 10};

      passed.min(a[0] == 1);
      passed.min(a[1] == 2);
      passed.min(a[2] == 10);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, copy_construct)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 3> a = {1, 2, 10};
      care::array<int, 3> b{a};

      passed.min(b[0] == 1);
      passed.min(b[1] == 2);
      passed.min(b[2] == 10);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, copy_construct_host_to_device)
{
   RAJAReduceMin<bool> passed{true};

   care::array<int, 3> a = {1, 2, 10};

   CARE_REDUCE_LOOP(i, 0, 1) {
      passed.min(a[0] == 1);
      passed.min(a[1] == 2);
      passed.min(a[2] == 10);
      a[0] = 3; // Should fail to compile
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, copy_assignment)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 3> a = {1, 2, 10};
      a = care::array<int, 3>{3, 4, 6};

      passed.min(a[0] == 3);
      passed.min(a[1] == 4);
      passed.min(a[2] == 6);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, access)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      a[0] = 3;
      passed.min(a[0] == 3);

      const care::array<int, 2>& b = a;
      passed.min(b[0] == 3);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, front)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      a.front() = 3;
      passed.min(a[0] == 3);

      const care::array<int, 2>& b = a;
      passed.min(b.front() == 3);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, back)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      a.back() = 5;
      passed.min(a[1] == 5);

      const care::array<int, 2>& b = a;
      passed.min(b.back() == 5);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, data)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      int* a_data = a.data();
      passed.min(a_data[0] == a[0]);
      passed.min(a_data[1] == a[1]);

      const care::array<int, 2>& b = a;
      const int* b_data = b.data();
      passed.min(b_data[0] == b[0]);
      passed.min(b_data[1] == b[1]);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, begin)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      auto a_it = a.begin();
      *a_it = 4;
      passed.min(a[0] == 4);
      *(++a_it) = 6;
      passed.min(a[1] == 6);

      const care::array<int, 2>& b = a;
      auto b_it = b.begin();
      passed.min(*b_it == b[0]);
      passed.min(*(++b_it) == b[1]);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, cbegin)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      auto a_it = a.cbegin();
      passed.min(*a_it == a[0]);
      passed.min(*(++a_it) == a[1]);

      const care::array<int, 2>& b = a;
      auto b_it = b.begin();
      passed.min(*b_it == b[0]);
      passed.min(*(++b_it) == b[1]);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, end)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      auto a_it = a.end();
      *(--a_it) = 4;
      passed.min(a[1] == 4);
      *(--a_it) = 6;
      passed.min(a[0] == 6);

      const care::array<int, 2>& b = a;
      auto b_it = b.end();
      passed.min(*(--b_it) == b[1]);
      passed.min(*(--b_it) == b[0]);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, cend)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      auto a_it = a.cend();
      passed.min(*(--a_it) == a[1]);
      passed.min(*(--a_it) == a[0]);

      const care::array<int, 2>& b = a;
      auto b_it = b.cend();
      passed.min(*(--b_it) == b[1]);
      passed.min(*(--b_it) == b[0]);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, empty)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<double, 0> a{};
      passed.min(a.empty());

      care::array<double, 1> b{1.0};
      passed.min(!b.empty());
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, size)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<double, 0> a{};
      passed.min(a.size() == 0);

      care::array<double, 2> b{1.0, 3.0};
      passed.min(b.size() == 2);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, max_size)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<double, 0> a{};
      passed.min(a.max_size() == 0);

      care::array<double, 2> b{1.0, 3.0};
      passed.min(b.max_size() == 2);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, fill)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 3> a{1, 2, 3};
      a.fill(0);

      for (size_t i = 0; i < 3; ++i) {
         passed.min(a[i] == 0);
      }
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, swap)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a{1, 2};
      care::array<int, 2> b{3, 4};

      a.swap(b);

      passed.min(a[0] == 3);
      passed.min(a[1] == 4);

      passed.min(b[0] == 1);
      passed.min(b[1] == 2);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, equal)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a{1, 2};
      care::array<int, 2> b{1, 3};

      passed.min(a == a);
      passed.min(!(a == b));
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, not_equal)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a{1, 2};
      care::array<int, 2> b{1, 3};

      passed.min(a != b);
      passed.min(!(a != a));
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, less_than)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a{1, 2};
      care::array<int, 2> b{1, 3};

      passed.min(a < b);
      passed.min(!(b < a));
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, less_than_or_equal)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a{1, 2};
      care::array<int, 2> b{1, 3};

      passed.min(a <= a);
      passed.min(a <= b);
      passed.min(!(b <= a));
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, greater_than)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a{1, 2};
      care::array<int, 2> b{1, 3};

      passed.min(b > a);
      passed.min(!(a > b));
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, greater_than_or_equal)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a{1, 2};
      care::array<int, 2> b{1, 3};

      passed.min(a >= a);
      passed.min(b >= a);
      passed.min(!(a >= b));
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, get_lvalue_reference)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      care::get<0>(a) = 3;
      passed.min(a[0] == 3);

      const care::array<int, 2>& b = a;
      passed.min(care::get<0>(b) == 3);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, get_rvalue_reference)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a = {1, 12};
      int&& a0 = care::get<0>(care::move(a));
      passed.min(a0 == 1);
      passed.min(a[0] == 1);

      const care::array<int, 2> b{6, 8};
      const int&& b1 = care::get<1>(care::move(b));
      passed.min(b1 == 8);
      passed.min(b[1] == 8);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, generic_swap)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a{1, 2};
      care::array<int, 2> b{3, 4};

      care::swap(a, b);

      passed.min(a[0] == 3);
      passed.min(a[1] == 4);

      passed.min(b[0] == 1);
      passed.min(b[1] == 2);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, to_array)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      int temp[3] = {1, 2, 10};

      care::array<int, 3> a = care::to_array(temp);
      passed.min(a[0] == 1);
      passed.min(a[1] == 2);
      passed.min(a[2] == 10);

      care::array<int, 3> b = care::to_array(care::move(temp));
      passed.min(b[0] == 1);
      passed.min(b[1] == 2);
      passed.min(b[2] == 10);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, tuple_size)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      constexpr std::size_t size = std::tuple_size<care::array<double, 7>>::value;
      constexpr std::size_t size_v = std::tuple_size_v<care::array<double, 11>>;

      passed.min(size == 7);
      passed.min(size_v == 11);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, tuple_element)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      constexpr bool element0 = std::is_same_v<double, std::tuple_element_t<0, care::array<double, 5>>>;
      constexpr bool element4 = std::is_same_v<double, std::tuple_element_t<4, care::array<double, 5>>>;

      passed.min(element0);
      passed.min(element4);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

GPU_TEST(array, structured_binding)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array<int, 2> a{-1, 1};
      auto& [a0, a1] = a;
      passed.min(a0 == -1);
      passed.min(a1 == 1);

      a1 = 3;
      passed.min(a[1] == 3);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}

#if 0
GPU_TEST(array, deduction_guide)
{
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      care::array a{-1, 1};
      passed.min(a[0] == -1);
      passed.min(a[1] == 1);
   } CARE_REDUCE_LOOP_END

   EXPECT_TRUE((bool) passed);
}
#endif

#endif // CARE_GPUCC

