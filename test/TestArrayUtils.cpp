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

// Array Fill Tests
TEST(array_utils, fill_empty)
{
   care::host_ptr<int> a;
   care::host_device_ptr<int> b;
 
   care_utils::ArrayFill<int>(a, 0, -12);
   care_utils::ArrayFill<int>(b, 0, -12);
}

TEST(array_utils, fill_one)
{
   int tempa[1] = {1};
   int tempb[1] = {1};
   care::host_ptr<int> a(tempa);
   // if you attempt to initialize this as b(tempb) without the size, the test fails. Maybe that constructor should be
   // disabled for host_device pointer with cuda active?
   care::host_device_ptr<int> b(tempb, 1, "fillhostdev");

   care_utils::ArrayFill<int>(a, 1, -12);
   care_utils::ArrayFill<int>(b, 1, -12);

   EXPECT_EQ(a[0], -12);
   EXPECT_EQ(b.pick(0), -12);
}

TEST(array_utils, fill_three)
{
   int tempa[3] = {1, 2, 3};
   int tempb[3] = {1, 2, 3};

   care::host_ptr<int> a(tempa);
   // if you attempt to initialize this as b(tempb) without the size, the test fails. Maybe that constructor should be
   // disabled for host_device pointer with cuda active?
   care::host_device_ptr<int> b(tempb, 3, "fillhostdev3");

   care_utils::ArrayFill<int>(a, 3, -12);
   care_utils::ArrayFill<int>(b, 3, -12);

   EXPECT_EQ(a[0], -12);
   EXPECT_EQ(a[1], -12);
   EXPECT_EQ(a[2], -12);

   EXPECT_EQ(b.pick(0), -12);
   EXPECT_EQ(b.pick(1), -12);
   EXPECT_EQ(b.pick(2), -12);
}

TEST(array_utils, min_empty)
{
  // If the parameter type is int (instead of const int), the code does not compile because
  // it cannot choose the correct ArrayMin signature. Can we change the signature?
  care::host_device_ptr<const int> a;
  // this works even when the start index is greater than length. Was that intended, or should it die with an error?
  int initVal = -1;
  int result = care_utils::ArrayMin<int>(a, 0, initVal, 567);
  EXPECT_EQ(result, initVal);
}

TEST(array_utils, min_seven)
{
  int temp[7] = {2, 1, 1, 8, 3, 5, 7};
  // If the parameter type is int (instead of const int), the code does not compile because
  // it cannot choose the correct ArrayMin signature. Can we change the signature?
  care::host_device_ptr<const int> a(temp, 6, "minseven");
  int initVal = -1;
  // min of whole array
  int result = care_utils::ArrayMin<int>(a, 7, initVal, 0);
  EXPECT_EQ(result, 1);
  
  // min starting at index 3
  result = care_utils::ArrayMin<int>(a, 7, initVal, 3);
  EXPECT_EQ(result, 3);

  // min of values at least 6
  initVal = 6;
  result = care_utils::ArrayMin<int>(a, 7, initVal, 0);
  EXPECT_EQ(result, 7);
}

#ifdef __CUDACC__

// Adapted from CHAI
#define GPU_TEST(X, Y) \
   static void gpu_test_##X##Y(); \
   TEST(X, gpu_test_##Y) { gpu_test_##X##Y(); } \
   static void gpu_test_##X##Y()

// Array Fill Tests
GPU_TEST(array_utils, fill_empty)
{  
   care::host_device_ptr<int> a;
   
   care_utils::ArrayFill<int>(a, 0, -12);
}

GPU_TEST(array_utils, fill_one)
{  
   int temp[1] = {1};
   care::host_device_ptr<int> a(temp);
   
   care_utils::ArrayFill<int>(a, 1, -12);

   RAJAReduceMin<bool> passed{true};
   LOOP_REDUCE(i, 0, 1) {
      if (a[i] != -12) {
         passed.min(false);
      }
   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array_utils, fill_three)
{
   int temp[3] = {1, 2, 3};
   care::host_device_ptr<int> a(temp);

   care_utils::ArrayFill<int>(a, 3, -12);

   RAJAReduceMin<bool> passed{true};
   LOOP_REDUCE(i, 0, 3) {
      if (a[i] != -12) {
         passed.min(false);
      }
   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);
}

#endif // __CUDACC__

