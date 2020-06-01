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

// min tests
TEST(array_utils, min_empty)
{
  care::host_device_ptr<int> a;
  // this works even when the start index is greater than length. Was that intended, or should it die with an error?
  int initVal = -1;
  int result = care_utils::ArrayMin<int>(a, 0, initVal, 567);
  EXPECT_EQ(result, initVal);
}

TEST(array_utils, min_seven)
{
  int temp[7] = {2, 1, 1, 8, 3, 5, 7};
  care::host_device_ptr<int> a(temp, 7, "minseven");
  int initVal = 99;
  // min of whole array
  int result = care_utils::ArrayMin<int>(a, 7, initVal, 0);
  EXPECT_EQ(result, 1);
  
  // min starting at index 3
  result = care_utils::ArrayMin<int>(a, 7, initVal, 3);
  EXPECT_EQ(result, 3);

  // test init val -1
  initVal = -1;
  result = care_utils::ArrayMin<int>(a, 7, initVal, 0);
  EXPECT_EQ(result, -1);
}

// max tests
TEST(array_utils, max_empty)
{
  care::host_device_ptr<int> a;
  // ArrayMin has a value for start index but ArrayMax does not. Why?
  int initVal = -1;
  int result = care_utils::ArrayMax<int>(a, 0, initVal);
  EXPECT_EQ(result, initVal);
}

TEST(array_utils, max_seven)
{
  int temp[7] = {2, 1, 1, 8, 3, 5, 7};
  care::host_device_ptr<int> a(temp, 7, "maxseven");
  int initVal = -1;
  // max of whole array
  int result = care_utils::ArrayMax<int>(a, 7, initVal);
  EXPECT_EQ(result, 8);

  double tempd[7] = {1.2, 3.0/2.0, 9.2, 11.0/5.0, 1/2, 97.8, -12.2};
  care::host_device_ptr<double> b(tempd, 7, "maxsevend"); 
  double resultd = care_utils::ArrayMax<double>(b, 7, initVal);
  EXPECT_EQ(resultd, 97.8);

  // test init val 99
  initVal = 99;
  result = care_utils::ArrayMax<int>(a, 7, initVal);
  EXPECT_EQ(result, 99);
}

TEST(array_utils, min_max_base)
{
  care::host_device_ptr<int> nill;
  double min[1] = {-1};
  double max[1] = {-1};
  
  int result = care_utils::ArrayMinMax<int>(nill, nill, 0, min, max);
  EXPECT_EQ(min[0], -DBL_MAX);
  EXPECT_EQ(max[0], DBL_MAX);
  EXPECT_EQ(result, false);

  int offmask[4] = {0};
  care::host_device_ptr<int> mask(offmask, 4, "offmask");
  min[0] = -1;
  max[0] = -1;
  result = care_utils::ArrayMinMax<int>(nill, mask, 0, min, max);
  EXPECT_EQ(min[0], -DBL_MAX);
  EXPECT_EQ(max[0], DBL_MAX);
  EXPECT_EQ(result, false);

  // the mask is set to off for the whole array, so we should still get the DBL_MAX base case here
  int vals[4] = {1, 2, 3, 4};
  care::host_device_ptr<int> skippedvals(vals, 4, "skipped");
  min[0] = -1;
  max[0] = -1;
  result = care_utils::ArrayMinMax<int>(skippedvals, mask, 0, min, max);
  EXPECT_EQ(min[0], -DBL_MAX);
  EXPECT_EQ(max[0], DBL_MAX);
  EXPECT_EQ(result, false);
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

GPU_TEST(array_utils, min_gpu)
{
  int temp0[7] = {2, 1, 1, 8, 3, 5, 7};
  int temp1[7] = {3, 1, 9, 10, 0, 12, 12};
  care::host_device_ptr<int> ind0(temp0, 7, "mingpu0");
  care::host_device_ptr<int> ind1(temp1, 7, "mingpu1");

  care::host_device_ptr<int> a[2] = {ind0, ind1};

   RAJAReduceMin<bool> passed{true};
   LOOP_REDUCE(i, 0, 1) {
      care::local_ptr<int> arr0 = a[0];
      care::local_ptr<int> arr1 = a[1]; 
      
      // min of entire array arr0
      int result = care_utils::ArrayMin<int>(arr0, 7, 99, 0);
      if (result != 1) {
         passed.min(false);
      }
      
      // min of arr0 starting at index 6
      result = care_utils::ArrayMin<int>(arr0, 7, 99, 6);
      if (result != 7) {
         passed.min(false);
      }

      // min of entire array arr1
      result = care_utils::ArrayMin<int>(arr1, 7, 99, 0);
      if (result != 0) {
         passed.min(false);
      }

      // value min of arr1 with init val -1
      result = care_utils::ArrayMin<int>(arr1, 7, -1, 7);
      if (result != -1) {
         passed.min(false);
      }

   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);
}

GPU_TEST(array_utils, max_gpu)
{
  int temp0[7] = {2, 1, 1, 8, 3, 5, 7};
  int temp1[7] = {3, 1, 9, 10, 0, 12, 12};
  care::host_device_ptr<int> ind0(temp0, 7, "maxgpu0");
  care::host_device_ptr<int> ind1(temp1, 7, "maxgpu1");

  care::host_device_ptr<int> a[2] = {ind0, ind1};

   RAJAReduceMin<bool> passed{true};
   LOOP_REDUCE(i, 0, 1) {
      care::local_ptr<int> arr0 = a[0];
      care::local_ptr<int> arr1 = a[1];

      // max of entire array arr0
      int result = care_utils::ArrayMax<int>(arr0, 7, -1);
      if (result != 8) {
         passed.min(false);
      }

      // max of arr0 starting at index 6
      //result = care_utils::ArrayMax<int>(arr0, 7, -1, 6);
      //if (result != 7) {
      //   passed.min(false);
      //}

      // max of entire array arr1
      result = care_utils::ArrayMax<int>(arr1, 7, -1);
      if (result != 12) {
         passed.min(false);
      }

      // value max of arr1 with init val 99
      result = care_utils::ArrayMax<int>(arr1, 7, 99);
      if (result != 99) {
         passed.min(false);
      }

   } LOOP_REDUCE_END

   ASSERT_TRUE((bool) passed);
}
#endif // __CUDACC__

