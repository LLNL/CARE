//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#define GPU_ACTIVE

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
   care::host_ptr<int> a = nullptr;
 
   care_utils::ArrayFill<int>(a, 0, -12);
   EXPECT_EQ(a, nullptr);
}

TEST(array_utils, fill_one)
{
   int tempa[1] = {1};
   care::host_ptr<int> a(tempa);

   care_utils::ArrayFill<int>(a, 1, -12);
   EXPECT_EQ(a[0], -12);
}

TEST(array_utils, fill_three)
{
   int tempa[3] = {1, 2, 3};
   care::host_ptr<int> a(tempa);

   care_utils::ArrayFill<int>(a, 3, -12);

   EXPECT_EQ(a[0], -12);
   EXPECT_EQ(a[1], -12);
   EXPECT_EQ(a[2], -12);
}

// NOTE: if an array is not sorted, the checkSorted function will print out an error message.
// When you run the unit tests, please ignore the spurious print statements.
TEST(array_utils, checkSorted) {
  const int sorted[7]    = {-1, 2, 3, 4, 5, 6, 23};
  const int notsorted[7] = {-1, 2, 1, 3, 4, 5, 6};
  const int sorteddup[7] = {-1, 0, 0, 0, 2, 3, 4};

  bool result = false;

  // nil array is considered sorted
  result = care_utils::checkSorted<int>(nullptr, 0, "test", "nil");
  ASSERT_TRUE(result);

  // sorted array is sorted
  result = care_utils::checkSorted<int>(sorted, 7, "test", "sorted", true);
  ASSERT_TRUE(result);

  result = care_utils::checkSorted<int>(sorted, 7, "test", "sorted", false);
  ASSERT_TRUE(result);

  // sorteddup is sorted but it has a duplicate. Will succeed or fail depending
  // on whether duplicates are allowed (last param)
  result = care_utils::checkSorted<int>(sorteddup, 7, "test", "sorteddup", true);
  ASSERT_TRUE(result);

  result = care_utils::checkSorted<int>(sorteddup, 7, "test", "sorteddup", false);
  ASSERT_FALSE(result);

  // not sorted
  result = care_utils::checkSorted<int>(notsorted, 7, "test", "sorteddup", true);
  ASSERT_FALSE(result);

  result = care_utils::checkSorted<int>(notsorted, 7, "test", "sorteddup", false);
  ASSERT_FALSE(result);
}

TEST(array_utils, binarysearch) {
   int* nil = nullptr;
   int  a[7] = {-9, 0, 3, 7, 77, 500, 999}; // sorted no duplicates
   int  b[7] = {0, 1, 1, 1, 1, 1, 6};       // sorted with duplicates
   int  c[7] = {1, 1, 1, 1, 1, 1, 1};       // uniform array edge case.
   int result = 0;
  
   // nil test
   result = care_utils::BinarySearch<int>(nil, 0, 0, 77, false);
   EXPECT_EQ(result, -1);

   // search for 77
   result = care_utils::BinarySearch<int>(a, 0, 7, 77, false);
   EXPECT_EQ(result, 4);

   // start after the number
   // NOTE: input mapSize is NOT the size of the map. It is the length is the region you want to search. So if the array is length
   // 7 and you start at index 2, feed BinarySearch 7-2=5.
   result = care_utils::BinarySearch<int>(a, 5, 7-5, 77, false);
   EXPECT_EQ(result, -1);

   // upper bound is one after
   result = care_utils::BinarySearch<int>(a, 2, 7-2, 77, true);
   EXPECT_EQ(result, 5);

   result = care_utils::BinarySearch<int>(b, 0, 7, 0, false);
   EXPECT_EQ(result, 0);

   result = care_utils::BinarySearch<int>(b, 0, 7, 6, false);
   EXPECT_EQ(result, 6);

   // one is repeated, could be a range of answers
   result = care_utils::BinarySearch<int>(b, 0, 7, 1, false);
   ASSERT_TRUE(result > 0 && result < 6);

   // turn on upper bound, should ge the value after all of the ones.
   result = care_utils::BinarySearch<int>(b, 0, 7, 1, true);
   EXPECT_EQ(result, 6);

   // check upper bound on uniform arrary as an edge case
   result = care_utils::BinarySearch<int>(c, 0, 7, 1, true);
   EXPECT_EQ(result, -1);
}

TEST(array_utils, intersectarrays) {
   int tempa[3] = {1, 2, 5};
   int tempb[5] = {2, 3, 4, 5, 6};
   int tempc[7] = {-1, 0, 2, 3, 6, 120, 360};
   int tempd[9] = {1001, 1002, 2003, 3004, 4005, 5006, 6007, 7008, 8009};
   int* nil = nullptr;
   care::host_ptr<int> a(tempa);
   care::host_ptr<int> b(tempb);
   care::host_ptr<int> c(tempc);
   care::host_ptr<int> d(tempd);

   care::host_ptr<int> matches1, matches2;
   int numMatches[1] = {77};

   // nil test
   care_utils::IntersectArrays<int>(RAJA::seq_exec(), c, 7, 0, nil, 0, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0); 

   // intersect c and b
   care_utils::IntersectArrays<int>(RAJA::seq_exec(), c, 7, 0, b, 5, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 3);
   EXPECT_EQ(matches1[0], 2);
   EXPECT_EQ(matches1[1], 3);  
   EXPECT_EQ(matches1[2], 4);
   EXPECT_EQ(matches2[0], 0); 
   EXPECT_EQ(matches2[1], 1);
   EXPECT_EQ(matches2[2], 4);

   // introduce non-zero starting locations. In this case, matches are given as offsets from those starting locations
   // and not the zero position of the arrays.
   care_utils::IntersectArrays<int>(RAJA::seq_exec(), c, 7, 3, b, 5, 1, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 2); 
   EXPECT_EQ(matches1[0], 0);
   EXPECT_EQ(matches1[1], 1);
   EXPECT_EQ(matches2[0], 0);
   EXPECT_EQ(matches2[1], 3);

   // intersect a and b
   care_utils::IntersectArrays<int>(RAJA::seq_exec(), a, 3, 0, b, 5, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 2);
   EXPECT_EQ(matches1[0], 1); 
   EXPECT_EQ(matches1[1], 2);
   EXPECT_EQ(matches2[0], 0);
   EXPECT_EQ(matches2[1], 3); 

   // offset one past the end
   care_utils::IntersectArrays<int>(RAJA::seq_exec(), a, 3, 0, b, 5, 98, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0);

   // no matches
   care_utils::IntersectArrays<int>(RAJA::seq_exec(), a, 3, 0, d, 9, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0);
}

#ifdef __CUDACC__

// Adapted from CHAI
#define GPU_TEST(X, Y) \
   static void gpu_test_##X##Y(); \
   TEST(X, gpu_test_##Y) { gpu_test_##X##Y(); } \
   static void gpu_test_##X##Y()

GPU_TEST(array_utils, fill_empty) {
   care::host_device_ptr<int> a;
   care_utils::ArrayFill<int>(a, 0, -12); // hopefully nothing explodes
}

GPU_TEST(array_utils, fill_one)
{
   int tempb[1] = {1};
   // if you attempt to initialize this as b(tempb) without the size, the test fails. Maybe that constructor should be
   // disabled for host_device pointer with cuda active?
   care::host_device_ptr<int> b(tempb, 1, "fillhostdev");

   care_utils::ArrayFill<int>(b, 1, -12);

   EXPECT_EQ(b.pick(0), -12);
}

GPU_TEST(array_utils, fill_three) {
   int tempb[3] = {1, 2, 3};
   // if you attempt to initialize this as b(tempb) without the size, the test fails. Maybe that constructor should be
   // disabled for host_device pointer with cuda active?
   care::host_device_ptr<int> b(tempb, 3, "fillhostdev3");

   care_utils::ArrayFill<int>(b, 3, -12);
   EXPECT_EQ(b.pick(0), -12);
   EXPECT_EQ(b.pick(1), -12);
   EXPECT_EQ(b.pick(2), -12);
}

GPU_TEST(array_utils, min_empty)
{
  care::host_device_ptr<int> a;
  // this works even when the start index is greater than length.
  int initVal = -1;
  int result = care_utils::ArrayMin<int>(a, 0, initVal, 567);
  EXPECT_EQ(result, initVal);
}

GPU_TEST(array_utils, min_innerloop)
{
  // this tests the local_ptr version of min
  int temp0[7] = {2, 1, 1, 8, 3, 5, 7};
  int temp1[7] = {3, 1, 9, 10, 0, 12, 12};
  care::host_device_ptr<int> ind0(temp0, 7, "mingpu0");
  care::host_device_ptr<int> ind1(temp1, 7, "mingpu1");

  RAJAReduceMin<bool> passed{true};
  LOOP_REDUCE(i, 0, 1) {
     care::local_ptr<int> arr0 = ind0;
     care::local_ptr<int> arr1 = ind1;

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


GPU_TEST(array_utils, min_seven)
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

GPU_TEST(array_utils, max_empty)
{
  care::host_device_ptr<int> a;
  // ArrayMin has a value for start index but ArrayMax does not. TODO: use slicing or pointer arithmatic instead.
  int initVal = -1;
  int result = care_utils::ArrayMax<int>(a, 0, initVal);
  EXPECT_EQ(result, initVal);
}

GPU_TEST(array_utils, max_seven)
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

GPU_TEST(array_utils, max_innerloop)
{
  // test the local_ptr version of max
  int temp0[7] = {2, 1, 1, 8, 3, 5, 7};
  int temp1[7] = {3, 1, 9, 10, 0, 12, 12};
  care::host_device_ptr<int> ind0(temp0, 7, "maxgpu0");
  care::host_device_ptr<int> ind1(temp1, 7, "maxgpu1");

  RAJAReduceMin<bool> passed{true};
  LOOP_REDUCE(i, 0, 1) {
     care::local_ptr<int> arr0 = ind0;
     care::local_ptr<int> arr1 = ind1;

     // max of entire array arr0
     int result = care_utils::ArrayMax<int>(arr0, 7, -1);
     if (result != 8) {
        passed.min(false);
     }

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

GPU_TEST(array_utils, min_max_notfound)
{
  // some minmax tests for empty arrays
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

GPU_TEST(array_utils, min_max_general)
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

  // test with a mask
  care_utils::ArrayMinMax<int>(a7, mask, 7, min, max);
  EXPECT_EQ(min[0], 0);
  EXPECT_EQ(max[0], 5);

  // no mask, just find the min/max in the array
  care_utils::ArrayMinMax<int>(a1, nullptr, 1, min, max);
  EXPECT_EQ(min[0], 2);
  EXPECT_EQ(max[0], 2);
}

GPU_TEST(array_utils, min_max_innerloop)
{
  // tests for the local_ptr version of minmax
  int vals1[1] = {2};
  int vals7[7] = {1, 5, 4, 3, -2, 9, 0};
  int valsmask7[7] = {1, 1, 1, 1, 0, 0, 1}; // this mask would skip the existing max/min as a check

  care::host_device_ptr<int> mask(valsmask7, 7, "skippedvals");
  care::host_device_ptr<int> a1(vals1, 1, "minmax1");
  care::host_device_ptr<int> a7(vals7, 7, "minmax7");

  RAJAReduceMin<bool> passed{true};
  LOOP_REDUCE(i, 0, 1) {
     double min[1] = {-1};
     double max[1] = {-1};
     care::local_ptr<int> arr1 = a1;
     care::local_ptr<int> arr7 = a7;
     care::local_ptr<int> mask7 = mask;

     care_utils::ArrayMinMax<int>(arr7, nullptr, 7, min, max);
     if (min[0] != -2 && max[0] != 9) {
        passed.min(false);
     }

     care_utils::ArrayMinMax<int>(arr7, mask7, 7, min, max);
     if (min[0] != 0 && max[0] != 5) {
        passed.min(false);
     }

     care_utils::ArrayMinMax<int>(arr1, nullptr, 1, min, max);
     if (min[0] != 2 && max[0] != 2) {
        passed.min(false);
     }

  } LOOP_REDUCE_END

  ASSERT_TRUE((bool)passed);
}

GPU_TEST(array_utils, minloc_empty)
{ 
  care::host_device_ptr<int> a;
  int loc = 10;
  int initVal = -1;
  int result = care_utils::ArrayMinLoc<int>(a, 0, initVal, loc);
  EXPECT_EQ(result, initVal);
  EXPECT_EQ(loc, -1); // empty array, not found
}

GPU_TEST(array_utils, minloc_seven)
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

GPU_TEST(array_utils, maxloc_empty)
{
  care::host_device_ptr<int> a;
  int loc = 7;
  int initVal = -1;
  int result = care_utils::ArrayMaxLoc<int>(a, 0, initVal, loc);
  EXPECT_EQ(result, initVal);
  EXPECT_EQ(loc, -1); // empty, not found
}

GPU_TEST(array_utils, maxloc_seven)
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

GPU_TEST(array_utils, arrayfind)
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

GPU_TEST(array_utils, findabovethreshold)
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

GPU_TEST(array_utils, minindexsubset)
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

GPU_TEST(array_utils, minindexsubsetabovethresh)
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

GPU_TEST(array_utils, maxindexsubset)
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

GPU_TEST(array_utils, maxindexsubsetabovethresh)
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

GPU_TEST(array_utils, pickperformfindmax)
{
  int temp[7] = {2, 1, 1, 8, 5, 3, 7};
  int masktemp[7] = {0, 0, 0, 1, 0, 0, 0};
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
  care::host_device_ptr<int> mask(masktemp, 7, "mask");
  care::host_device_ptr<double> threshold7(thresh7, 7, "thresh7");
  care::host_device_ptr<double> threshold3(thresh3, 3, "thresh3");
  care::host_device_ptr<double> threshold1(thresh1, 1, "thresh1");

  double cutoff = -10;

  // all elements in reverse order, no mask or subset
  result = care_utils::PickAndPerformFindMaxIndex<int>(a, nullptr, nullptr, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, 3);
  EXPECT_EQ(threshIdx[0], 3);

  // all elements in reverse order, no subset
  result = care_utils::PickAndPerformFindMaxIndex<int>(a, mask, nullptr, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, -1); // This is -1 because it is masked off
  EXPECT_EQ(threshIdx[0], 3); // NOTE: even if a value is masked off, the threshold index still comes back

  // all elements but reverse order, no threshold
  result = care_utils::PickAndPerformFindMaxIndex<int>(a, mask, subsetrev, 7, nullptr, cutoff, threshIdx);
  EXPECT_EQ(result, -1); //masked off

  // all elements in reverse order, cutoff not triggered
  // NOTE: even though the element is masked off, threshIdx is valid!!! Just the return index is -1, not the threshIdx!!!!
  result = care_utils::PickAndPerformFindMaxIndex<int>(a, mask, subsetrev, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, -1); // this is the index in the original array
  EXPECT_EQ(threshIdx[0], 3); // this is the index in the subset

  // change the cutoff
  cutoff = 5.0;
  result = care_utils::PickAndPerformFindMaxIndex<int>(a, mask, subsetrev, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, 6); // this is the index in the original array

  // len 3 subset
  cutoff = 5.0;
  result = care_utils::PickAndPerformFindMaxIndex<int>(a, mask, subset3, 3, threshold3, cutoff, threshIdx);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(threshIdx[0], 1);

  // len 1 subset
  cutoff = 5.0;
  result = care_utils::PickAndPerformFindMaxIndex<int>(a, mask, subset1, 1, threshold1, cutoff, threshIdx);
  EXPECT_EQ(result, 2);
  EXPECT_EQ(threshIdx[0], 0);

  // len 1 subset not found (cutoff too high)
  cutoff = 25.67;
  result = care_utils::PickAndPerformFindMaxIndex<int>(a, mask, subset1, 1, threshold1, cutoff, threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx[0], -1);
}

GPU_TEST(array_utils, pickperformfindmin)
{
  int temp[7] = {2, 1, 1, 8, 5, 3, 7};
  int masktemp[7] = {0, 1, 1, 0, 0, 0, 0};
  int rev[7] = {6, 5, 4, 3, 2, 1, 0};
  int sub3[3] = {5, 0, 6};
  int sub1[1] = {2};
  int threshIdx[1] = {-1};
  int result = 99;

  care::host_device_ptr<int> a(temp, 7, "arrseven");
  care::host_device_ptr<int> subset1(sub1, 1, "sub1");
  care::host_device_ptr<int> subset3(sub3, 1, "sub3");
  care::host_device_ptr<int> subsetrev(rev, 7, "rev");
  double thresh7[7] = {0, 0, 0, 0, 10, 10, 0}; // this must be double, cannot be int
  double thresh3[3] = {0, 10, 10};
  double thresh1[1] = {10};
  care::host_device_ptr<int> mask(masktemp, 7, "mask");
  care::host_device_ptr<double> threshold7(thresh7, 7, "thresh7");
  care::host_device_ptr<double> threshold3(thresh3, 3, "thresh3");
  care::host_device_ptr<double> threshold1(thresh1, 1, "thresh1");

  double cutoff = -10;

  // all elements in reverse order, no mask or subset
  result = care_utils::PickAndPerformFindMinIndex<int>(a, nullptr, nullptr, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, 1);
  EXPECT_EQ(threshIdx[0], 1);

  // all elements in reverse order, no subset
  result = care_utils::PickAndPerformFindMinIndex<int>(a, mask, nullptr, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, -1); // This is -1 because it is masked off
  EXPECT_EQ(threshIdx[0], 1); // NOTE: even if a value is masked off, the threshold index still comes back

  // all elements but reverse order, no threshold
  result = care_utils::PickAndPerformFindMinIndex<int>(a, mask, subsetrev, 7, nullptr, cutoff, threshIdx);
  EXPECT_EQ(result, -1); //masked off

  // all elements in reverse order, cutoff not triggered
  // NOTE: even though the element is masked off, threshIdx is valid!!! Just the return index is -1, not the threshIdx!!!!
  result = care_utils::PickAndPerformFindMinIndex<int>(a, mask, subsetrev, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, -1); // this is the index in the original array
  EXPECT_EQ(threshIdx[0], 4); // this is the index in the subset

  // change the cutoff
  // CURRENTLY TEST GIVES DIFFERENT RESULTS WHEN COMPILED RELEASE VS DEBUG!!!
  cutoff = 5.0;
  result = care_utils::PickAndPerformFindMinIndex<int>(a, nullptr, subsetrev, 7, threshold7, cutoff, threshIdx);
  EXPECT_EQ(result, -1); // this is the index in the original array
  EXPECT_EQ(threshIdx[0], -1);

  // len 3 subset
  cutoff = 5.0;
  result = care_utils::PickAndPerformFindMinIndex<int>(a, mask, subset3, 3, threshold3, cutoff, threshIdx);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(threshIdx[0], 1);

  // len 1 subset, masked off
  cutoff = 5.0;
  result = care_utils::PickAndPerformFindMinIndex<int>(a, mask, subset1, 1, threshold1, cutoff, threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx[0], 0);
}

GPU_TEST(array_utils, arraycount)
{
  int temp[7] = {0, 1, 0, 3, 4, 0, 6};
  care::host_device_ptr<int> a(temp, 7, "arrseven");
  care::host_device_ptr<int> b = nullptr;
  
  int result = -1;

  // count element in array
  result = care_utils::ArrayCount<int>(a, 7, 0);
  EXPECT_EQ(result, 3);

  result = care_utils::ArrayCount<int>(a, 7, 6);
  EXPECT_EQ(result, 1);

  // not in the array
  result = care_utils::ArrayCount<int>(a, 7, -1);
  EXPECT_EQ(result, 0);

  // the null array test
  result = care_utils::ArrayCount<int>(b, 0, 0);
  EXPECT_EQ(result, 0);
}

GPU_TEST(array_utils, arraysum)
{
  int temp7[7] = {0, 1, 0, 3, 4, 0, 6};
  int temp1[1] = {99};
  care::host_device_ptr<int> a(temp7, 7, "arrseven");
  care::host_device_ptr<int> b(temp1, 1, "arrone");
  care::host_device_ptr<int> nil = nullptr;

  int result = -1;

  // sum everything
  result = care_utils::ArraySum<int>(a, 7, 0);
  EXPECT_EQ(result, 14);

  // sum everything with initval -12
  result = care_utils::ArraySum<int>(a, 7, -12);
  EXPECT_EQ(result, 2);

  // sum first 4 elems of length 7 array
  result = care_utils::ArraySum<int>(a, 4, 0);
  EXPECT_EQ(result, 4);

  // length 1 array
  result = care_utils::ArraySum<int>(b, 1, 0);
  EXPECT_EQ(result, 99);

  // the null array test
  result = care_utils::ArraySum<int>(nil, 0, 0);
  EXPECT_EQ(result, 0);
 
  // null array, different init val
  result = care_utils::ArraySum<int>(nil, 0, -5);
  EXPECT_EQ(result, -5);
}

GPU_TEST(array_utils, arraysumsubset)
{
  int temp7[7] = {0, 1, 0, 3, 4, 0, 6};
  int temp1[1] = {99};

  int subtemp3[3] = {3, 1, 4};
  int subtemp1[1] = {0};

  care::host_device_ptr<int> a(temp7, 7, "arrseven");
  care::host_device_ptr<int> b(temp1, 1, "arrone");
  care::host_device_ptr<int> sub3(subtemp3, 3, "suba");
  care::host_device_ptr<int> sub1(subtemp1, 1, "subb");
  care::host_device_ptr<int> nil = nullptr;

  int result = -1;

  // sum subset
  result = care_utils::ArraySumSubset<int, int>(a, sub3, 3, 0);
  EXPECT_EQ(result, 8);

  // sum everything with initval -12
  result = care_utils::ArraySumSubset<int, int>(a, sub3, 3, -12);
  EXPECT_EQ(result, -4);

  // sum first 2 elements from subset subset
  result = care_utils::ArraySumSubset<int, int>(a, sub3, 2, 0);
  EXPECT_EQ(result, 4);

  // length 1 array
  result = care_utils::ArraySumSubset<int, int>(b, sub1, 1, 0);
  EXPECT_EQ(result, 99);

  // the null array test
  result = care_utils::ArraySumSubset<int, int>(nil, nil, 0, 0);
  EXPECT_EQ(result, 0);

  // null array, different init val
  result = care_utils::ArraySumSubset<int, int>(nil, nil, 0, -5);
  EXPECT_EQ(result, -5);
}

GPU_TEST(array_utils, arraysummaskedsubset)
{ 
  int temp7[7] = {0, 1, 0, 3, 4, 0, 6};
  int temp1[1] = {99};
  
  int subtemp3[3] = {3, 1, 4};
  int subtemp1[1] = {0};
  
  int masktemp[7] = {1, 1, 0, 1, 0, 0, 0};

  care::host_device_ptr<int> a(temp7, 7, "arrseven");
  care::host_device_ptr<int> b(temp1, 1, "arrone");
  care::host_device_ptr<int> sub3(subtemp3, 3, "suba");
  care::host_device_ptr<int> sub1(subtemp1, 1, "subb");
  care::host_device_ptr<int> mask(masktemp, 7, "mask");
  care::host_device_ptr<int> nil = nullptr;
  
  int result = -1;
  
  // sum subset
  result = care_utils::ArrayMaskedSumSubset<int, int>(a, mask, sub3, 3, 0);
  EXPECT_EQ(result, 4);
  
  // sum first 2 elements from subset subset
  result = care_utils::ArrayMaskedSumSubset<int, int>(a, mask, sub3, 2, 0);
  EXPECT_EQ(result, 0);
  
  // length 1 array
  result = care_utils::ArrayMaskedSumSubset<int, int>(b, mask, sub1, 1, 0);
  EXPECT_EQ(result, 0);
  
  // the null array test
  result = care_utils::ArrayMaskedSumSubset<int, int>(nil, mask, nil, 0, 0);
  EXPECT_EQ(result, 0);
}

GPU_TEST(array_utils, findindexgt)
{
  int temp7[7] = {0, 1, 0, 3, 4, 0, 6};
  care::host_device_ptr<int> a(temp7, 7, "arrseven");
  int result = -1;
  
  // in practice it finds the element that is the highest above the limit, so 6
  // But in theory a different implementation could give the first element above the limit.
  result = care_utils::FindIndexGT<int>(a, 7, 3);
  ASSERT_TRUE(result==4 || result==6);

  result = care_utils::FindIndexGT<int>(a, 7, 0);
  ASSERT_TRUE(result==1 || result==3 || result==4 || result==6);

  // limit is higher than all elements
  result = care_utils::FindIndexGT<int>(a, 7, 99);
  EXPECT_EQ(result, -1);
}

GPU_TEST(array_utils, findindexmax)
{
  int temp7[7] = {0, 1, 0, 3, 99, 0, 6};
  int temp3[3] = {9, 10, -2};
  int temp2[2] = {5, 5};

  care::host_device_ptr<int> a(temp7, 7, "arrseven");
  care::host_device_ptr<int> b(temp3, 3, "arrthree");
  care::host_device_ptr<int> c(temp2, 2, "arrtwo");
  int result = -1;

  result = care_utils::FindIndexMax<int>(a, 7);
  EXPECT_EQ(result, 4);

  result = care_utils::FindIndexMax<int>(b, 2);
  EXPECT_EQ(result, 1);

  result = care_utils::FindIndexMax<int>(c, 2);
  EXPECT_EQ(result, 0);

  // nil test
  result = care_utils::FindIndexMax<int>(nullptr, 0);
  EXPECT_EQ(result, -1);
}

// duplicating and copying arrays
// NOTE: no test for when to and from are the same array or aliased. I'm assuming that is not allowed.
GPU_TEST(array_utils, dup_and_copy) {
  const int temp7[7] = {9, 10, -2, 67, 9, 45, -314};
  int zeros1[7] = {0};
  int zeros2[7] = {0};
  int zeros3[7] = {0};

  care::host_device_ptr<int> to1(zeros1, 7, "zeros1");
  care::host_device_ptr<int> to2(zeros2, 7, "zeros2");
  care::host_device_ptr<int> to3(zeros3, 7, "zeros3");
  care::host_device_ptr<const int> from(temp7, 7, "from7");
  care::host_device_ptr<int> nil = nullptr;

  // duplicate and check that elements are the same
  care::host_device_ptr<int> dup = care_utils::ArrayDup<int>(from, 7);
  RAJAReduceMin<bool> passeddup{true};
  LOOP_REDUCE(i, 0, 7) {
    if (dup[i] != from[i]) {
      passeddup.min(false);
    }
  } LOOP_REDUCE_END
  ASSERT_TRUE((bool) passeddup);

  // duplicating nullptr should give nullptr
  care::host_device_ptr<int> dupnil = care_utils::ArrayDup<int>(nil, 0);
  EXPECT_EQ(dupnil, nullptr);

  // copy and check that elements are the same
  care_utils::ArrayCopy<int>(to1, from, 7);
  RAJAReduceMin<bool> passed1{true};
  LOOP_REDUCE(i, 0, 7) {
    if (to1[i] != from[i]) {
      passed1.min(false);
    }
  } LOOP_REDUCE_END
  ASSERT_TRUE((bool) passed1);

  // copy 2 elements, testing different starting points
  care_utils::ArrayCopy<int>(to2, from, 2, 3, 4);
  RAJAReduceMin<bool> passed2{true};

  LOOP_REDUCE(i, 0, 1) {
    if (to2[0] != 0 || to2[1] != 0 || to2[2] != 0 || to2[5] != 0 || to2[6] != 0) {
      passed2.min(false);
    }
    if (to2[3] != 9) {
      passed2.min(false);
    }
    if (to2[4] != 45) {
      passed2.min(false);
    }
  } LOOP_REDUCE_END
  ASSERT_TRUE((bool) passed2);

  // copy 2 elements, testing different starting points
  care_utils::ArrayCopy<int>(to3, from, 2, 4, 3);
  RAJAReduceMin<bool> passed3{true};

  LOOP_REDUCE(i, 0, 1) {
    if (to3[0] != 0 || to3[1] != 0 || to3[2] != 0 || to3[3] != 0 || to3[6] != 0) {
      passed3.min(false);
    }
    if (to3[4] != 67) {
      passed3.min(false);
    }
    if (to3[5] != 9) {
      passed3.min(false);
    }
  } LOOP_REDUCE_END
  ASSERT_TRUE((bool) passed3);
}


GPU_TEST(array_utils, intersectarrays) {
   int tempa[3] = {1, 2, 5};
   int tempb[5] = {2, 3, 4, 5, 6};
   int tempc[7] = {-1, 0, 2, 3, 6, 120, 360};
   int tempd[9] = {1001, 1002, 2003, 3004, 4005, 5006, 6007, 7008, 8009};
   int* nil = nullptr;
   care::host_device_ptr<int> a(tempa, 3, "a");
   care::host_device_ptr<int> b(tempb, 5, "b");
   care::host_device_ptr<int> c(tempc, 7, "c");
   care::host_device_ptr<int> d(tempd, 9, "d");

   care::host_device_ptr<int> matches1, matches2;
   int numMatches[1] = {77};

   // nil test
   care_utils::IntersectArrays<int>(RAJAExec(), c, 7, 0, nil, 0, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0);

   // intersect c and b
   care_utils::IntersectArrays<int>(RAJAExec(), c, 7, 0, b, 5, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 3);
   EXPECT_EQ(matches1.pick(0), 2);
   EXPECT_EQ(matches1.pick(1), 3);
   EXPECT_EQ(matches1.pick(2), 4);
   EXPECT_EQ(matches2.pick(0), 0);
   EXPECT_EQ(matches2.pick(1), 1);
   EXPECT_EQ(matches2.pick(2), 4);

   // introduce non-zero starting locations. In this case, matches are given as offsets from those starting locations
   // and not the zero position of the arrays.
   care_utils::IntersectArrays<int>(RAJAExec(), c, 7, 3, b, 5, 1, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 2);
   EXPECT_EQ(matches1.pick(0), 0);
   EXPECT_EQ(matches1.pick(1), 1);
   EXPECT_EQ(matches2.pick(0), 0);
   EXPECT_EQ(matches2.pick(1), 3);

   // intersect a and b
   care_utils::IntersectArrays<int>(RAJAExec(), a, 3, 0, b, 5, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 2);
   EXPECT_EQ(matches1.pick(0), 1);
   EXPECT_EQ(matches1.pick(1), 2);
   EXPECT_EQ(matches2.pick(0), 0);
   EXPECT_EQ(matches2.pick(1), 3);

   // offset one past the end
   care_utils::IntersectArrays<int>(RAJAExec(), a, 3, 0, b, 5, 98, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0);

   // no matches
   care_utils::IntersectArrays<int>(RAJAExec(), a, 3, 0, d, 9, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0);
}

#endif // __CUDACC__

