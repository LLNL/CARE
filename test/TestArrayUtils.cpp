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

TEST(array_utils, min_max_general)
{ 
  double min[1] = {-1};
  double max[1] = {-1};
  int vals1[1] = {2};
  int vals7[7] = {1, 5, 4, 3, -2, 9, 0};
  int mask7[7] = {1, 1, 1, 1, 0, 0, 1}; // this mask would skip the existing max/min as a check

  care::host_device_ptr<int> mask(mask7, 7, "skippedvals");
  care::host_device_ptr<int> a1(vals1, 1, "minmax1");
  care::host_device_ptr<int> a7(vals7, 7, "minmax7");
  
  // note that output min/max are double whereas input is int. I am not testing for casting failures because
  // I'm treating the output value as a given design decision
  care_utils::ArrayMinMax<int>(a7, nullptr, 7, min, max);
  EXPECT_EQ(min[0], -2);
  EXPECT_EQ(max[0], 9);

  care_utils::ArrayMinMax<int>(a7, mask, 7, min, max);
  EXPECT_EQ(min[0], 0);
  EXPECT_EQ(max[0], 5);

  care_utils::ArrayMinMax<int>(a1, nullptr, 1, min, max);
  EXPECT_EQ(min[0], 2);
  EXPECT_EQ(max[0], 2);
}


// minloc tests
TEST(array_utils, minloc_empty)
{ 
  care::host_device_ptr<int> a;
  int loc = 10;
  int initVal = -1;
  int result = care_utils::ArrayMinLoc<int>(a, 0, initVal, loc);
  EXPECT_EQ(result, initVal);
  EXPECT_EQ(loc, -1);
}

TEST(array_utils, minloc_seven)
{
  int loc = -10;
  int temp[7] = {2, 1, 1, 8, 3, 5, 7};
  care::host_device_ptr<int> a(temp, 7, "minseven");
  int initVal = 99;
  // min of whole array
  int result = care_utils::ArrayMinLoc<int>(a, 7, initVal, loc);
  EXPECT_EQ(result, 1);
  EXPECT_EQ(loc, 1);

  // test init val -1
  initVal = -1;
  result = care_utils::ArrayMinLoc<int>(a, 7, initVal, loc);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(loc, -1);
}

// maxloc tests
TEST(array_utils, maxloc_empty)
{
  care::host_device_ptr<int> a;
  int loc = 7;
  int initVal = -1;
  int result = care_utils::ArrayMaxLoc<int>(a, 0, initVal, loc);
  EXPECT_EQ(result, initVal);
  EXPECT_EQ(loc, -1);
}

TEST(array_utils, maxloc_seven)
{
  int loc = -1;
  int temp[7] = {2, 1, 1, 8, 3, 5, 7};
  care::host_device_ptr<int> a(temp, 7, "maxseven");
  int initVal = -1;
  // max of whole array
  int result = care_utils::ArrayMaxLoc<int>(a, 7, initVal, loc);
  EXPECT_EQ(result, 8);
  EXPECT_EQ(loc, 3);

  double tempd[7] = {1.2, 3.0/2.0, 9.2, 11.0/5.0, 1/2, 97.8, -12.2};
  care::host_device_ptr<double> b(tempd, 7, "maxsevend");
  double resultd = care_utils::ArrayMaxLoc<double>(b, 7, initVal, loc);
  EXPECT_EQ(resultd, 97.8);
  EXPECT_EQ(loc, 5);

  // test init val 99
  initVal = 99;
  result = care_utils::ArrayMaxLoc<int>(a, 7, initVal, loc);
  EXPECT_EQ(result, 99);
  EXPECT_EQ(loc, -1);
}

TEST(array_utils, arrayfind)
{ 
  int loc = 99;
  int temp1[1] = {10};
  int temp7[7] = {2, 1, 8, 3, 5, 7, 1};
  care::host_device_ptr<int> a1(temp1, 7, "find1");
  care::host_device_ptr<int> a7(temp7, 7, "find7");

  // empty array
  loc = care_utils::ArrayFind<int>(nullptr, 0, 10, 0);
  EXPECT_EQ(loc, -1);

  loc = care_utils::ArrayFind<int>(a1, 1, 10, 0);
  EXPECT_EQ(loc, 0);

  // not present
  loc = care_utils::ArrayFind<int>(a1, 1, 77, 0);
  EXPECT_EQ(loc, -1);
 
  // start location past the end of the array 
  loc = 4;
  loc = care_utils::ArrayFind<int>(a1, 1, 77, 99);
  EXPECT_EQ(loc, -1);

  loc = care_utils::ArrayFind<int>(a7, 7, 1, 0);
  EXPECT_EQ(loc, 1);

  // start search at index 3
  loc = care_utils::ArrayFind<int>(a7, 7, 1, 3);
  EXPECT_EQ(loc, 6);

  loc = care_utils::ArrayFind<int>(a7, 7, 8, 0);
  EXPECT_EQ(loc, 2);
}

TEST(array_utils, findabovethreshold)
{
  int result = -1;
  int threshIdx[1] = {0};
  int temp[7] = {2, 1, 1, 8, 3, 5, 7};
  double thresh[7] = {0, 0, 0, 10, 10, 10, 10}; // this must be double, cannot be int
  care::host_device_ptr<int> a(temp, 7, "array");
  care::host_device_ptr<double> threshold(thresh, 7, "thresh");
  double cutoff = -10;
  
  // null array
  result = care_utils::FindIndexMinAboveThresholds<int>(nullptr, 0, threshold, cutoff, threshIdx);
  EXPECT_EQ(result, -1);

  // null threshold
  result = care_utils::FindIndexMinAboveThresholds<int>(a, 7, nullptr, cutoff, threshIdx);
  EXPECT_EQ(result, 1);

  // threshold always triggers
  result = care_utils::FindIndexMinAboveThresholds<int>(a, 7, threshold, cutoff, threshIdx);
  EXPECT_EQ(result, 1);
  EXPECT_EQ(threshIdx[0], 1);

  // set threshold higher to alter result
  cutoff = 5;
  result = care_utils::FindIndexMinAboveThresholds<int>(a, 7, threshold, cutoff, threshIdx);
  EXPECT_EQ(result, 4);
  EXPECT_EQ(threshIdx[0], 4);

  // threshold never triggers
  cutoff = 999;
  result = care_utils::FindIndexMinAboveThresholds<int>(a, 7, threshold, cutoff, threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx[0], -1);
}

TEST(array_utils, minindexsubset)
{ 
  int temp[7] = {2, 1, 1, 8, 3, 5, 7};
  int rev[7] = {6, 5, 4, 3, 2, 1, 0};
  int sub3[3] = {5, 0, 6};
  int sub1[1] = {3};
  int result = 99;

  care::host_device_ptr<int> a(temp, 7, "arrseven");
  care::host_device_ptr<int> subset1(sub1, 1, "sub1");
  care::host_device_ptr<int> subset3(sub3, 1, "sub3");
  care::host_device_ptr<int> subsetrev(rev, 7, "rev");

  // null subset
  result = care_utils::FindIndexMinSubset<int>(a, nullptr, 0);
  EXPECT_EQ(result, -1);
  
  // all elements but reverse order
  result = care_utils::FindIndexMinSubset<int>(a, subsetrev, 7);
  // NOTE: Since we are going in reverse order, this is 2 NOT 1
  EXPECT_EQ(result, 2);

  // len 3 subset
  result = care_utils::FindIndexMinSubset<int>(a, subset3, 3);
  EXPECT_EQ(result, 0);

  // len 1 subset
  result = care_utils::FindIndexMinSubset<int>(a, subset1, 1);
  EXPECT_EQ(result, 3);  
}

TEST(array_utils, minindexsubsetabovethresh)
{
  int temp[7] = {2, 1, 1, 8, 5, 3, 7};
  int rev[7] = {6, 5, 4, 3, 2, 1, 0};
  int sub3[3] = {5, 0, 6};
  int sub1[1] = {3};
  int threshIdx[1] = {-1};
  int result = 99;

  care::host_device_ptr<int> a(temp, 7, "arrseven");
  care::host_device_ptr<int> subset1(sub1, 1, "sub1");
  care::host_device_ptr<int> subset3(sub3, 1, "sub3");
  care::host_device_ptr<int> subsetrev(rev, 7, "rev");
  double thresh7[7] = {10, 10, 10, 10, 0, 0, 10}; // this must be double, cannot be int
  double thresh3[3] = {0, 10, 10};
  double thresh1[1] = {10};
  care::host_device_ptr<double> threshold7(thresh7, 7, "thresh7");
  care::host_device_ptr<double> threshold3(thresh3, 3, "thresh3");
  care::host_device_ptr<double> threshold1(thresh1, 1, "thresh1");

  double cutoff = -10;

  // null subset
  result = care_utils::FindIndexMinSubsetAboveThresholds<int>(a, nullptr, 0, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx[0], -1);

  // all elements but reverse order, no threshold
  result = care_utils::FindIndexMinSubsetAboveThresholds<int>(a, subsetrev, 7, nullptr, cutoff, threshIdx);
  // NOTE: Since we are going in reverse order, this is 2 NOT 1
  EXPECT_EQ(result, 2);

  // all elements in reverse order, cutoff not triggered
  result = care_utils::FindIndexMinSubsetAboveThresholds<int>(a, subsetrev, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, 2); // this is the index in the original array
  EXPECT_EQ(threshIdx[0], 4); // this is the index in the subset

  // change the cutoff
  cutoff = 5.0;
  result = care_utils::FindIndexMinSubsetAboveThresholds<int>(a, subsetrev, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, 0); // this is the index in the original array
  EXPECT_EQ(threshIdx[0], 6); // this is the index in the subset

  // len 3 subset
  cutoff = 5.0;
  result = care_utils::FindIndexMinSubsetAboveThresholds<int>(a, subset3, 3, threshold3, cutoff, threshIdx);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(threshIdx[0], 1);

  // len 1 subset
  cutoff = 5.0;
  result = care_utils::FindIndexMinSubsetAboveThresholds<int>(a, subset1, 1, threshold1, cutoff, threshIdx);
  EXPECT_EQ(result, 3);
  EXPECT_EQ(threshIdx[0], 0);

  // len 1 subset not found (cutoff too high)
  cutoff = 25.67;
  result = care_utils::FindIndexMinSubsetAboveThresholds<int>(a, subset1, 1, threshold1, cutoff, threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx[0], -1);
}

TEST(array_utils, maxindexsubset)
{
  int temp[7] = {2, 1, 1, 8, 3, 5, 7};
  int rev[7] = {6, 5, 4, 3, 2, 1, 0};
  int sub3[3] = {5, 0, 6};
  int sub1[1] = {2};
  int result = 99;

  care::host_device_ptr<int> a(temp, 7, "arrseven");
  care::host_device_ptr<int> subset1(sub1, 1, "sub1");
  care::host_device_ptr<int> subset3(sub3, 1, "sub3");
  care::host_device_ptr<int> subsetrev(rev, 7, "rev");

  // null subset
  result = care_utils::FindIndexMaxSubset<int>(a, nullptr, 0);
  EXPECT_EQ(result, -1);

  // all elements but reverse order
  result = care_utils::FindIndexMaxSubset<int>(a, subsetrev, 7);
  EXPECT_EQ(result, 3);

  // len 3 subset
  result = care_utils::FindIndexMaxSubset<int>(a, subset3, 3);
  EXPECT_EQ(result, 6);

  // len 1 subset
  result = care_utils::FindIndexMaxSubset<int>(a, subset1, 1);
  EXPECT_EQ(result, 2);
}

TEST(array_utils, maxindexsubsetabovethresh)
{
  int temp[7] = {2, 1, 1, 8, 5, 3, 7};
  int rev[7] = {6, 5, 4, 3, 2, 1, 0};
  int sub3[3] = {5, 0, 6};
  int sub1[1] = {2};
  int threshIdx[1] = {-1};
  int result = 99;

  care::host_device_ptr<int> a(temp, 7, "arrseven");
  care::host_device_ptr<int> subset1(sub1, 1, "sub1");
  care::host_device_ptr<int> subset3(sub3, 1, "sub3");
  care::host_device_ptr<int> subsetrev(rev, 7, "rev");
  double thresh7[7] = {10, 10, 10, 0, 0, 0, 10}; // this must be double, cannot be int
  double thresh3[3] = {0, 10, 10};
  double thresh1[1] = {10};
  care::host_device_ptr<double> threshold7(thresh7, 7, "thresh7");
  care::host_device_ptr<double> threshold3(thresh3, 3, "thresh3");
  care::host_device_ptr<double> threshold1(thresh1, 1, "thresh1");

  double cutoff = -10;

  // null subset
  result = care_utils::FindIndexMaxSubsetAboveThresholds<int>(a, nullptr, 0, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx[0], -1);

  // all elements but reverse order, no threshold
  result = care_utils::FindIndexMaxSubsetAboveThresholds<int>(a, subsetrev, 7, nullptr, cutoff, threshIdx);
  EXPECT_EQ(result, 3);

  // all elements in reverse order, cutoff not triggered
  result = care_utils::FindIndexMaxSubsetAboveThresholds<int>(a, subsetrev, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, 3); // this is the index in the original array
  EXPECT_EQ(threshIdx[0], 3); // this is the index in the subset

  // change the cutoff
  cutoff = 5.0;
  result = care_utils::FindIndexMaxSubsetAboveThresholds<int>(a, subsetrev, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, 6); // this is the index in the original array
  EXPECT_EQ(threshIdx[0], 0); // this is the index in the subset

  // len 3 subset
  cutoff = 5.0;
  result = care_utils::FindIndexMaxSubsetAboveThresholds<int>(a, subset3, 3, threshold3, cutoff, threshIdx);
  EXPECT_EQ(result, 6);
  EXPECT_EQ(threshIdx[0], 2);

  // len 1 subset
  cutoff = 5.0;
  result = care_utils::FindIndexMaxSubsetAboveThresholds<int>(a, subset1, 1, threshold1, cutoff, threshIdx);
  EXPECT_EQ(result, 2);
  EXPECT_EQ(threshIdx[0], 0);

  // len 1 subset not found (cutoff too high)
  cutoff = 25.67;
  result = care_utils::FindIndexMaxSubsetAboveThresholds<int>(a, subset1, 1, threshold1, cutoff, threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx[0], -1);
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

GPU_TEST(array_utils, min_max_general)
{
  int vals1[1] = {2};
  int vals7[7] = {1, 5, 4, 3, -2, 9, 0};
  int mask7[7] = {1, 1, 1, 1, 0, 0, 1}; // this mask would skip the existing max/min as a check
  
  care::host_device_ptr<int> mask(mask7, 7, "skippedvals");
  care::host_device_ptr<int> a1(vals1, 1, "minmax1");
  care::host_device_ptr<int> a7(vals7, 7, "minmax7");

  care::host_device_ptr<int> a[3] = {a1, a7, mask};

  RAJAReduceMin<bool> passed{true};
  LOOP_REDUCE(i, 0, 1) {
     double min[1] = {-1};
     double max[1] = {-1};
     care::local_ptr<int> arr1 = a[0];
     care::local_ptr<int> arr7 = a[1];
     care::local_ptr<int> mask7 = a[2];
      
     care_utils::ArrayMinMax<int, double>(arr7, nullptr, 7, min, max);
     if (min[0] != -2 && max[0] != 9) {
        passed.min(false);
     }
      
     care_utils::ArrayMinMax<int, double>(arr7, mask7, 7, min, max);
     if (min[0] != 0 && max[0] != 5) {
        passed.min(false);
     }

     care_utils::ArrayMinMax<int, double>(arr1, nullptr, 1, min, max);
     if (min[0] != 2 && max[0] != 2) {
        passed.min(false);
     }

  } LOOP_REDUCE_END

  ASSERT_TRUE((bool)passed);
}

#endif // __CUDACC__

