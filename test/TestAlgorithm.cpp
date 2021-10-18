//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#define GPU_ACTIVE

// CARE config header
#include "care/config.h"

// CARE headers
#include "care/algorithm.h"

// Other library headers
#include "gtest/gtest.h"



// fill_n Tests
TEST(algorithm, fill_empty)
{
   int size = 0;
   care::host_device_ptr<int> a = nullptr;
   care::fill_n(a, size, -12);
   EXPECT_EQ(a, nullptr);
   a.free();
}

TEST(algorithm, fill_one)
{
   int size = 1;
   care::host_device_ptr<int> a(size, "a");

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      a[i] = i;
   } CARE_SEQUENTIAL_LOOP_END

   care::fill_n(a, size, -12);

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      EXPECT_EQ(a[i], -12);
   } CARE_SEQUENTIAL_LOOP_END

   a.free();
}

TEST(algorithm, fill_three)
{
   int size = 3;
   care::host_device_ptr<int> a(size, "a");

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      a[i] = i;
   } CARE_SEQUENTIAL_LOOP_END

   care::fill_n(a, size, 7);

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      EXPECT_EQ(a[i], 7);
   } CARE_SEQUENTIAL_LOOP_END

   a.free();
}

// NOTE: if an array is not sorted, the checkSorted function will print out an error message.
// When you run the unit tests, please ignore the spurious print statements.
TEST(algorithm, checkSorted) {
  const int sorted[7]    = {-1, 2, 3, 4, 5, 6, 23};
  const int notsorted[7] = {-1, 2, 1, 3, 4, 5, 6};
  const int sorteddup[7] = {-1, 0, 0, 0, 2, 3, 4};

  bool result = false;

  // nil array is considered sorted
  result = care::checkSorted<int>(nullptr, 0, "test", "nil");
  ASSERT_TRUE(result);

  // sorted array is sorted
  result = care::checkSorted<int>(sorted, 7, "test", "sorted", true);
  ASSERT_TRUE(result);

  result = care::checkSorted<int>(sorted, 7, "test", "sorted", false);
  ASSERT_TRUE(result);

  // sorteddup is sorted but it has a duplicate. Will succeed or fail depending
  // on whether duplicates are allowed (last param)
  result = care::checkSorted<int>(sorteddup, 7, "test", "sorteddup", true);
  ASSERT_TRUE(result);

  result = care::checkSorted<int>(sorteddup, 7, "test", "sorteddup", false);
  ASSERT_FALSE(result);

  // not sorted
  result = care::checkSorted<int>(notsorted, 7, "test", "sorteddup", true);
  ASSERT_FALSE(result);

  result = care::checkSorted<int>(notsorted, 7, "test", "sorteddup", false);
  ASSERT_FALSE(result);
}

TEST(algorithm, binarysearch) {
   int* nil = nullptr;
   int  a[7] = {-9, 0, 3, 7, 77, 500, 999}; // sorted no duplicates
   int  b[7] = {0, 1, 1, 1, 1, 1, 6};       // sorted with duplicates
   int  c[7] = {1, 1, 1, 1, 1, 1, 1};       // uniform array edge case.
   care::host_ptr<int> aptr(a);

   int result = 0;
  
   // nil test
   result = care::BinarySearch<int>(nil, 0, 0, 77, false);
   EXPECT_EQ(result, -1);

   // search for 77
   result = care::BinarySearch<int>(a, 0, 7, 77, false);
   EXPECT_EQ(result, 4);

   result = care::BinarySearch<int>(aptr.cdata(), 0, 7, 77, false);
   EXPECT_EQ(result, 4);

   // start after the number
   // NOTE: input mapSize is NOT the size of the map. It is the length is the region you want to search. So if the array is length
   // 7 and you start at index 2, feed BinarySearch 7-2=5.
   result = care::BinarySearch<int>(a, 5, 7-5, 77, false);
   EXPECT_EQ(result, -1);

   // upper bound is one after
   result = care::BinarySearch<int>(a, 2, 7-2, 77, true);
   EXPECT_EQ(result, 5);

   result = care::BinarySearch<int>(b, 0, 7, 0, false);
   EXPECT_EQ(result, 0);

   result = care::BinarySearch<int>(b, 0, 7, 6, false);
   EXPECT_EQ(result, 6);

   // one is repeated, could be a range of answers
   result = care::BinarySearch<int>(b, 0, 7, 1, false);
   ASSERT_TRUE(result > 0 && result < 6);

   // turn on upper bound, should ge the value after all of the ones.
   result = care::BinarySearch<int>(b, 0, 7, 1, true);
   EXPECT_EQ(result, 6);

   // check upper bound on uniform arrary as an edge case
   result = care::BinarySearch<int>(c, 0, 7, 1, true);
   EXPECT_EQ(result, -1);
}

TEST(algorithm, intersectarrays) {
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
   care::IntersectArrays<int>(RAJA::seq_exec(), c, 7, 0, nil, 0, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0); 

   // intersect c and b
   care::IntersectArrays<int>(RAJA::seq_exec(), c, 7, 0, b, 5, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 3);
   EXPECT_EQ(matches1[0], 2);
   EXPECT_EQ(matches1[1], 3);  
   EXPECT_EQ(matches1[2], 4);
   EXPECT_EQ(matches2[0], 0); 
   EXPECT_EQ(matches2[1], 1);
   EXPECT_EQ(matches2[2], 4);

   // introduce non-zero starting locations. In this case, matches are given as offsets from those starting locations
   // and not the zero position of the arrays.
   care::IntersectArrays<int>(RAJA::seq_exec(), c, 4, 3, b, 4, 1, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 2); 
   EXPECT_EQ(matches1[0], 0);
   EXPECT_EQ(matches1[1], 1);
   EXPECT_EQ(matches2[0], 0);
   EXPECT_EQ(matches2[1], 3);

   // intersect a and b
   care::IntersectArrays<int>(RAJA::seq_exec(), a, 3, 0, b, 5, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 2);
   EXPECT_EQ(matches1[0], 1); 
   EXPECT_EQ(matches1[1], 2);
   EXPECT_EQ(matches2[0], 0);
   EXPECT_EQ(matches2[1], 3); 

   // no matches
   care::IntersectArrays<int>(RAJA::seq_exec(), a, 3, 0, d, 9, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0);
}

TEST(algorithm, compressarray)
{
   // Test CompressArray with removed list mode
   int size = 10;
   care::host_device_ptr<int> a(size, "a");

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      a[i] = 100 + i;
   } CARE_SEQUENTIAL_LOOP_END

   int removedLen = 3;
   care::host_device_ptr<int> removed(removedLen, "removed");

   // Remove entries 0-2
   CARE_SEQUENTIAL_LOOP(i, 0, removedLen) {
      removed[i] = i ;
   } CARE_SEQUENTIAL_LOOP_END

   care::CompressArray<int>(RAJA::seq_exec(), a, size, removed, removedLen, care::compress_array::removed_list, true) ;

   CARE_SEQUENTIAL_LOOP(i, 0, size-removedLen) {
      EXPECT_EQ(a[i], 100 + (i + removedLen));
   } CARE_SEQUENTIAL_LOOP_END

   a.free();
   removed.free();

   // Test CompressArray with map list mode
   care::host_device_ptr<int> b(size, "b");

   CARE_SEQUENTIAL_LOOP(i, 0, size) {
      b[i] = 100 + i;
   } CARE_SEQUENTIAL_LOOP_END

   int newLen = 7;
   care::host_device_ptr<int> mapList(newLen, "mapList");

   // Keep the last 7 entries, but in reverse order
   CARE_SEQUENTIAL_LOOP(i, 0, newLen) {
      mapList[i] = size-i ;
   } CARE_SEQUENTIAL_LOOP_END

   care::CompressArray<int>(RAJA::seq_exec(), b, size, mapList, newLen, care::compress_array::mapping_list, true) ;

   CARE_SEQUENTIAL_LOOP(i, 0, newLen) {
      EXPECT_EQ(a[i], 100 + (size-i));
   } CARE_SEQUENTIAL_LOOP_END

   b.free();
   mapList.free();
}

#if defined(CARE_GPUCC)

// Adapted from CHAI
#define GPU_TEST(X, Y) \
   static void gpu_test_##X##Y(); \
   TEST(X, gpu_test_##Y) { gpu_test_##X##Y(); } \
   static void gpu_test_##X##Y()

GPU_TEST(algorithm, min_empty)
{
  const int size = 0;
  care::host_device_ptr<int> a;

  // this works even when the start index is greater than length.
  const int initVal = -1;
  const int result = care::ArrayMin<int>(a, size, initVal, 567);
  EXPECT_EQ(result, initVal);

  a.free();
}

GPU_TEST(algorithm, min_innerloop)
{
  // this tests the local_ptr version of min
  const int size = 7;
  care::host_device_ptr<int> ind0(size, "mingpu0");
  care::host_device_ptr<int> ind1(size, "mingpu1");

  CARE_GPU_KERNEL {
     ind0[0] = 2;
     ind0[1] = 1;
     ind0[2] = 1;
     ind0[3] = 8;
     ind0[4] = 3;
     ind0[5] = 5;
     ind0[6] = 7;

     ind1[0] = 3;
     ind1[1] = 1;
     ind1[2] = 9;
     ind1[3] = 10;
     ind1[4] = 0;
     ind1[5] = 12;
     ind1[6] = 12;
  } CARE_GPU_KERNEL_END

  RAJAReduceMin<bool> passed{true};

  CARE_REDUCE_LOOP(i, 0, 1) {
     care::local_ptr<int> arr0 = ind0;
     care::local_ptr<int> arr1 = ind1;

     // min of entire array arr0
     int result = care::ArrayMin<int>(arr0, size, 99, 0);

     if (result != 1) {
        passed.min(false);
     }

     // min of arr0 starting at index 6
     result = care::ArrayMin<int>(arr0, size, 99, 6);

     if (result != size) {
        passed.min(false);
     }

     // min of entire array arr1
     result = care::ArrayMin<int>(arr1, size, 99, 0);

     if (result != 0) {
        passed.min(false);
     }

     // value min of arr1 with init val -1
     result = care::ArrayMin<int>(arr1, size, -1, 7);

     if (result != -1) {
        passed.min(false);
     }

  } CARE_REDUCE_LOOP_END

   ASSERT_TRUE((bool) passed);

   ind1.free();
   ind0.free();
}


GPU_TEST(algorithm, min_seven)
{
  const int size = 7;
  care::host_device_ptr<int> a(size, "minseven");

  CARE_GPU_KERNEL {
     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 3;
     a[5] = 5;
     a[6] = 7;
  } CARE_GPU_KERNEL_END

  int initVal = 99;

  // min of whole array
  int result = care::ArrayMin<int>(a, size, initVal, 0);
  EXPECT_EQ(result, 1);
  
  // min starting at index 3
  result = care::ArrayMin<int>(a, size, initVal, 3);
  EXPECT_EQ(result, 3);

  // test init val -1
  initVal = -1;
  result = care::ArrayMin<int>(a, size, initVal, 0);
  EXPECT_EQ(result, -1);

  a.free();
}

GPU_TEST(algorithm, max_empty)
{
  const int size = 0;
  care::host_device_ptr<int> a;

  // ArrayMin has a value for start index but ArrayMax does not. TODO: use slicing or pointer arithmatic instead.
  const int initVal = -1;
  const int result = care::ArrayMax<int>(a, size, initVal);
  EXPECT_EQ(result, initVal);

  a.free();
}

GPU_TEST(algorithm, max_seven)
{
  const int size = 7;
  care::host_device_ptr<int> a(size, "maxseven");
  care::host_device_ptr<double> b(size, "maxsevend");

  CARE_GPU_KERNEL {
     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 3;
     a[5] = 5;
     a[6] = 7;

     b[0] = 1.2;
     b[1] = 3.0/2.0;
     b[2] = 9.2;
     b[3] = 11.0/5.0;
     b[4] = 1/2;
     b[5] = 97.8;
     b[6] = -12.2;
  } CARE_GPU_KERNEL_END

  int initVal = -1;

  // max of whole array
  int result = care::ArrayMax<int>(a, size, initVal);
  EXPECT_EQ(result, 8);

  double resultd = care::ArrayMax<double>(b, size, initVal);
  EXPECT_EQ(resultd, 97.8);

  // test init val 99
  initVal = 99;
  result = care::ArrayMax<int>(a, size, initVal);
  EXPECT_EQ(result, 99);

  b.free();
  a.free();
}

GPU_TEST(algorithm, max_innerloop)
{
  // test the local_ptr version of max

  const int size = 7;
  care::host_device_ptr<int> ind0(size, "maxgpu0");
  care::host_device_ptr<int> ind1(size, "maxgpu1");

  CARE_GPU_KERNEL {
     ind0[0] = 2;
     ind0[1] = 1;
     ind0[2] = 1;
     ind0[3] = 8;
     ind0[4] = 3;
     ind0[5] = 5;
     ind0[6] = 7;

     ind1[0] = 3;
     ind1[1] = 1;
     ind1[2] = 9;
     ind1[3] = 10;
     ind1[4] = 0;
     ind1[5] = 12;
     ind1[6] = 12;
  } CARE_GPU_KERNEL_END

  RAJAReduceMin<bool> passed{true};

  CARE_REDUCE_LOOP(i, 0, 1) {
     care::local_ptr<int> arr0 = ind0;
     care::local_ptr<int> arr1 = ind1;

     // max of entire array arr0
     int result = care::ArrayMax<int>(arr0, size, -1);

     if (result != 8) {
        passed.min(false);
     }

     // max of entire array arr1
     result = care::ArrayMax<int>(arr1, size, -1);

     if (result != 12) {
        passed.min(false);
     }

     // value max of arr1 with init val 99
     result = care::ArrayMax<int>(arr1, size, 99);

     if (result != 99) {
        passed.min(false);
     }
  } CARE_REDUCE_LOOP_END

  ASSERT_TRUE((bool) passed);

  ind1.free();
  ind0.free();
}

GPU_TEST(algorithm, min_max_notfound)
{
  double min = -1;
  double max = -1;

  // some minmax tests for empty arrays
  const int size1 = 0;
  care::host_device_ptr<int> nill;
 
  int result = care::ArrayMinMax<int>(nill, nill, size1, &min, &max);
  EXPECT_EQ(min, -DBL_MAX);
  EXPECT_EQ(max,  DBL_MAX);
  EXPECT_EQ(result, false);

  const int size2 = 4;
  care::host_device_ptr<int> mask(size2, "mask");

  CARE_GPU_LOOP(i, 0, size2) {
     mask[i] = 0;
  } CARE_GPU_LOOP_END

  min = -1;
  max = -1;

  result = care::ArrayMinMax<int>(nill, mask, size1, &min, &max);

  EXPECT_EQ(min, -DBL_MAX);
  EXPECT_EQ(max,  DBL_MAX);
  EXPECT_EQ(result, false);

  // the mask is set to off for the whole array, so we should still get the DBL_MAX base case here
  care::host_device_ptr<int> skippedvals(size2, "skipped");

  CARE_GPU_LOOP(i, 0, size2) {
     skippedvals[i] = i + 1;
  } CARE_GPU_LOOP_END

  min = -1;
  max = -1;

  result = care::ArrayMinMax<int>(skippedvals, mask, size2, &min, &max);

  EXPECT_EQ(min, -DBL_MAX);
  EXPECT_EQ(max,  DBL_MAX);
  EXPECT_EQ(result, false);

  skippedvals.free();
  mask.free();
  nill.free();
}

GPU_TEST(algorithm, min_max_general)
{ 
  double min = -1;
  double max = -1;

  const int size1 = 1;
  const int size2 = 7;

  care::host_device_ptr<int> a1(size1, "minmax1");
  care::host_device_ptr<int> a7(size2, "minmax7");
  care::host_device_ptr<int> mask(size2, "skippedvals");

  CARE_GPU_KERNEL {
     a1[0] = 2;

     a7[0] = 1;
     a7[1] = 5;
     a7[2] = 4;
     a7[3] = 3;
     a7[4] = -2;
     a7[5] = 9;
     a7[6] = 0;

     mask[0] = 1;
     mask[1] = 1;
     mask[2] = 1;
     mask[3] = 1;
     mask[4] = 0;
     mask[5] = 0;
     mask[6] = 1;
  } CARE_GPU_KERNEL_END
  
  // note that output min/max are double whereas input is int. I am not testing for casting failures because
  // I'm treating the output value as a given design decision
  care::ArrayMinMax<int>(a7, nullptr, size2, &min, &max);
  EXPECT_EQ(min, -2);
  EXPECT_EQ(max,  9);

  // test with a mask
  care::ArrayMinMax<int>(a7, mask, size2, &min, &max);
  EXPECT_EQ(min, 0);
  EXPECT_EQ(max, 5);

  // no mask, just find the min/max in the array
  care::ArrayMinMax<int>(a1, nullptr, size1, &min, &max);
  EXPECT_EQ(min, 2);
  EXPECT_EQ(max, 2);

  mask.free();
  a7.free();
  a1.free();
}

GPU_TEST(algorithm, min_max_innerloop)
{
  // tests for the local_ptr version of minmax
  const int size1 = 1;
  const int size2 = 7;

  care::host_device_ptr<int> a1(size1, "minmax1");
  care::host_device_ptr<int> a7(size2, "minmax7");
  care::host_device_ptr<int> mask(size2, "skippedvals");

  CARE_GPU_KERNEL {
     a1[0] = 2;

     a7[0] = 1;
     a7[1] = 5;
     a7[2] = 4;
     a7[3] = 3;
     a7[4] = -2;
     a7[5] = 9;
     a7[6] = 0;

     mask[0] = 1;
     mask[1] = 1;
     mask[2] = 1;
     mask[3] = 1;
     mask[4] = 0;
     mask[5] = 0;
     mask[6] = 1;
  } CARE_GPU_KERNEL_END

  RAJAReduceMin<bool> passed{true};

  CARE_REDUCE_LOOP(i, 0, 1) {
     double min = -1;
     double max = -1;

     care::local_ptr<int> arr1 = a1;
     care::local_ptr<int> arr7 = a7;
     care::local_ptr<int> mask7 = mask;

     care::ArrayMinMax<int>(arr7, nullptr, size2, &min, &max);

     if (min != -2 && max != 9) {
        passed.min(false);
     }

     care::ArrayMinMax<int>(arr7, mask7, size2, &min, &max);

     if (min != 0 && max != 5) {
        passed.min(false);
     }

     care::ArrayMinMax<int>(arr1, nullptr, size1, &min, &max);

     if (min != 2 && max != 2) {
        passed.min(false);
     }

  } CARE_REDUCE_LOOP_END

  ASSERT_TRUE((bool)passed);

  mask.free();
  a7.free();
  a1.free();
}

GPU_TEST(algorithm, minloc_empty)
{ 
  const int size = 0;
  care::host_device_ptr<int> a;
  int loc = 10;
  const int initVal = -1;

  const int result = care::ArrayMinLoc<int>(a, size, initVal, loc);
  EXPECT_EQ(result, initVal);
  EXPECT_EQ(loc, -1); // empty array, not found

  a.free();
}

GPU_TEST(algorithm, minloc_seven)
{
  const int size = 7;
  care::host_device_ptr<int> a(size, "minseven");

  CARE_GPU_KERNEL {
     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 3;
     a[5] = 5;
     a[6] = 7;
  } CARE_GPU_KERNEL_END

  // min of whole array
  int initVal = 99;
  int loc = -10;
  int result = care::ArrayMinLoc<int>(a, size, initVal, loc);
  EXPECT_EQ(result, 1);
  EXPECT_EQ(loc, 1);

  // test init val -1
  initVal = -1;
  result = care::ArrayMinLoc<int>(a, size, initVal, loc);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(loc, -1);

  a.free();
}

GPU_TEST(algorithm, maxloc_empty)
{
  const int size = 0;
  care::host_device_ptr<int> a;

  int loc = 7;
  const int initVal = -1;
  const int result = care::ArrayMaxLoc<int>(a, size, initVal, loc);
  EXPECT_EQ(result, initVal);
  EXPECT_EQ(loc, -1); // empty, not found

  a.free();
}

GPU_TEST(algorithm, maxloc_seven)
{
  const int size = 7;
  care::host_device_ptr<int> a(size, "minseven");
  care::host_device_ptr<double> b(size, "maxsevend");

  CARE_GPU_KERNEL {
     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 3;
     a[5] = 5;
     a[6] = 7;

     b[0] = 1.2;
     b[1] = 3.0/2.0;
     b[2] = 9.2;
     b[3] = 11.0/5.0;
     b[4] = 1/2;
     b[5] = 97.8;
     b[6] = -12.2;
  } CARE_GPU_KERNEL_END

  // max of whole array
  int loc = -1;
  int initVal = -1;
  int result = care::ArrayMaxLoc<int>(a, size, initVal, loc);
  EXPECT_EQ(result, 8);
  EXPECT_EQ(loc, 3);

  const double resultd = care::ArrayMaxLoc<double>(b, size, initVal, loc);
  EXPECT_EQ(resultd, 97.8);
  EXPECT_EQ(loc, 5);

  // test init val 99
  initVal = 99;
  result = care::ArrayMaxLoc<int>(a, size, initVal, loc);
  EXPECT_EQ(result, 99);
  EXPECT_EQ(loc, -1);

  b.free();
  a.free();
}

GPU_TEST(algorithm, arrayfind)
{ 
  int loc = 99;

  const int size1 = 1;
  care::host_device_ptr<int> a1(size1, "find1");

  const int size2 = 7;
  care::host_device_ptr<int> a7(size2, "find7");

  CARE_GPU_KERNEL {
     a1[0] = 10;

     a7[0] = 2;
     a7[1] = 1;
     a7[2] = 8;
     a7[3] = 3;
     a7[4] = 5;
     a7[5] = 7;
     a7[6] = 1;
  } CARE_GPU_KERNEL_END

  // empty array
  loc = care::ArrayFind<int>(nullptr, 0, 10, 0);
  EXPECT_EQ(loc, -1);

  loc = care::ArrayFind<int>(a1, size1, 10, 0);
  EXPECT_EQ(loc, 0);

  // not present
  loc = care::ArrayFind<int>(a1, size1, 77, 0);
  EXPECT_EQ(loc, -1);
 
  // start location past the end of the array 
  loc = 4;
  loc = care::ArrayFind<int>(a1, size1, 77, 99);
  EXPECT_EQ(loc, -1);

  loc = care::ArrayFind<int>(a7, size2, 1, 0);
  EXPECT_EQ(loc, 1);

  // start search at index 3
  loc = care::ArrayFind<int>(a7, size2, 1, 3);
  EXPECT_EQ(loc, 6);

  loc = care::ArrayFind<int>(a7, size2, 8, 0);
  EXPECT_EQ(loc, 2);

  a7.free();
  a1.free();
}

GPU_TEST(algorithm, findabovethreshold)
{
  const int size = 7;
  care::host_device_ptr<int> a(size, "array");
  care::host_device_ptr<double> threshold(size, "thresh");

  CARE_GPU_KERNEL {
     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 3;
     a[5] = 5;
     a[6] = 7;

     threshold[0] = 0;
     threshold[1] = 0;
     threshold[2] = 0;
     threshold[3] = 10;
     threshold[4] = 10;
     threshold[5] = 10;
     threshold[6] = 10;
  } CARE_GPU_KERNEL_END

  // null array
  double cutoff = -10;
  int threshIdx = 0;
  int result = care::FindIndexMinAboveThresholds<int>(nullptr, 0, threshold, cutoff, &threshIdx);
  EXPECT_EQ(result, -1);

  // null threshold
  result = care::FindIndexMinAboveThresholds<int>(a, size, nullptr, cutoff, &threshIdx);
  EXPECT_EQ(result, 1);

  // threshold always triggers
  result = care::FindIndexMinAboveThresholds<int>(a, size, threshold, cutoff, &threshIdx);
  EXPECT_EQ(result, 1);
  EXPECT_EQ(threshIdx, 1);

  // set threshold higher to alter result
  cutoff = 5;
  result = care::FindIndexMinAboveThresholds<int>(a, size, threshold, cutoff, &threshIdx);
  EXPECT_EQ(result, 4);
  EXPECT_EQ(threshIdx, 4);

  // threshold never triggers
  cutoff = 999;
  result = care::FindIndexMinAboveThresholds<int>(a, size, threshold, cutoff, &threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx, -1);

  threshold.free();
  a.free();
}

GPU_TEST(algorithm, minindexsubset)
{ 
  const int size1 = 1;
  care::host_device_ptr<int> subset1(size1, "sub1");

  const int size3 = 3;
  care::host_device_ptr<int> subset3(size3, "sub3");

  const int size7 = 7;
  care::host_device_ptr<int> a(size7, "arrseven");
  care::host_device_ptr<int> subsetrev(size7, "rev");

  CARE_GPU_KERNEL {
     subset1[0] = 3;

     subset3[0] = 5;
     subset3[1] = 0;
     subset3[2] = 6;

     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 3;
     a[5] = 5;
     a[6] = 7;

     subsetrev[0] = 6;
     subsetrev[1] = 5;
     subsetrev[2] = 4;
     subsetrev[3] = 3;
     subsetrev[4] = 2;
     subsetrev[5] = 1;
     subsetrev[6] = 0;
  } CARE_GPU_KERNEL_END

  // null subset
  int result = care::FindIndexMinSubset<int>(a, nullptr, 0);
  EXPECT_EQ(result, -1);
  
  // all elements but reverse order
  result = care::FindIndexMinSubset<int>(a, subsetrev, size7);
  // NOTE: Since we are going in reverse order, this is 2 NOT 1
  EXPECT_EQ(result, 2);

  // len 3 subset
  result = care::FindIndexMinSubset<int>(a, subset3, size3);
  EXPECT_EQ(result, 0);

  // len 1 subset
  result = care::FindIndexMinSubset<int>(a, subset1, size1);
  EXPECT_EQ(result, 3);  

  subsetrev.free();
  a.free();
  subset3.free();
  subset1.free();
}

GPU_TEST(algorithm, minindexsubsetabovethresh)
{
  const int size1 = 1;
  care::host_device_ptr<int> subset1(size1, "sub1");
  care::host_device_ptr<double> threshold1(size1, "thresh1");

  const int size3 = 3;
  care::host_device_ptr<int> subset3(size3, "sub3");
  care::host_device_ptr<double> threshold3(size3, "thresh3");

  const int size7 = 7;
  care::host_device_ptr<int> a(size7, "arrseven");
  care::host_device_ptr<int> subsetrev(size7, "rev");
  care::host_device_ptr<double> threshold7(size7, "thresh7");


  CARE_GPU_KERNEL {
     subset1[0] = 3;

     threshold1[0] = 10;

     subset3[0] = 5;
     subset3[1] = 0;
     subset3[2] = 6;

     threshold3[0] = 0;
     threshold3[1] = 10;
     threshold3[2] = 10;

     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 5;
     a[5] = 3;
     a[6] = 7;

     subsetrev[0] = 6;
     subsetrev[1] = 5;
     subsetrev[2] = 4;
     subsetrev[3] = 3;
     subsetrev[4] = 2;
     subsetrev[5] = 1;
     subsetrev[6] = 0;

     threshold7[0] = 10;
     threshold7[1] = 10;
     threshold7[2] = 10;
     threshold7[3] = 10;
     threshold7[4] = 0;
     threshold7[5] = 0;
     threshold7[6] = 10;
  } CARE_GPU_KERNEL_END

  // null subset
  double cutoff = -10;
  int threshIdx = -1;
  int result = care::FindIndexMinSubsetAboveThresholds<int>(a, nullptr, 0, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx, -1);

  // all elements but reverse order, no threshold
  result = care::FindIndexMinSubsetAboveThresholds<int>(a, subsetrev, size7, nullptr, cutoff, &threshIdx);
  // NOTE: Since we are going in reverse order, this is 2 NOT 1
  EXPECT_EQ(result, 2);

  // all elements in reverse order, cutoff not triggered
  result = care::FindIndexMinSubsetAboveThresholds<int>(a, subsetrev, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, 2); // this is the index in the original array
  EXPECT_EQ(threshIdx, 4); // this is the index in the subset

  // change the cutoff
  cutoff = 5.0;
  result = care::FindIndexMinSubsetAboveThresholds<int>(a, subsetrev, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, 0); // this is the index in the original array
  EXPECT_EQ(threshIdx, 6); // this is the index in the subset

  // len 3 subset
  cutoff = 5.0;
  result = care::FindIndexMinSubsetAboveThresholds<int>(a, subset3, size3, threshold3, cutoff, &threshIdx);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(threshIdx, 1);

  // len 1 subset
  cutoff = 5.0;
  result = care::FindIndexMinSubsetAboveThresholds<int>(a, subset1, size1, threshold1, cutoff, &threshIdx);
  EXPECT_EQ(result, 3);
  EXPECT_EQ(threshIdx, 0);

  // len 1 subset not found (cutoff too high)
  cutoff = 25.67;
  result = care::FindIndexMinSubsetAboveThresholds<int>(a, subset1, size1, threshold1, cutoff, &threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx, -1);

  threshold7.free();
  subsetrev.free();
  a.free();
  threshold3.free();
  subset3.free();
  threshold1.free();
  subset1.free();
}

GPU_TEST(algorithm, maxindexsubset)
{
  const int size1 = 1;
  care::host_device_ptr<int> subset1(size1, "sub1");

  const int size3 = 3;
  care::host_device_ptr<int> subset3(size3, "sub3");

  const int size7 = 7;
  care::host_device_ptr<int> a(size7, "arrseven");
  care::host_device_ptr<int> subsetrev(size7, "rev");

  CARE_GPU_KERNEL {
     subset1[0] = 2;

     subset3[0] = 5;
     subset3[1] = 0;
     subset3[2] = 6;

     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 3;
     a[5] = 5;
     a[6] = 7;

     subsetrev[0] = 6;
     subsetrev[1] = 5;
     subsetrev[2] = 4;
     subsetrev[3] = 3;
     subsetrev[4] = 2;
     subsetrev[5] = 1;
     subsetrev[6] = 0;
  } CARE_GPU_KERNEL_END

  // null subset
  int result = care::FindIndexMaxSubset<int>(a, nullptr, 0);
  EXPECT_EQ(result, -1);

  // all elements but reverse order
  result = care::FindIndexMaxSubset<int>(a, subsetrev, size7);
  EXPECT_EQ(result, 3);

  // len 3 subset
  result = care::FindIndexMaxSubset<int>(a, subset3, size3);
  EXPECT_EQ(result, 6);

  // len 1 subset
  result = care::FindIndexMaxSubset<int>(a, subset1, size1);
  EXPECT_EQ(result, 2);

  subsetrev.free();
  a.free();
  subset3.free();
  subset1.free();
}

GPU_TEST(algorithm, maxindexsubsetabovethresh)
{
  const int size1 = 1;
  care::host_device_ptr<int> subset1(size1, "sub1");
  care::host_device_ptr<double> threshold1(size1, "thresh1");

  const int size3 = 3;
  care::host_device_ptr<int> subset3(size3, "sub3");
  care::host_device_ptr<double> threshold3(size3, "thresh3");

  const int size7 = 7;
  care::host_device_ptr<int> a(size7, "arrseven");
  care::host_device_ptr<int> subsetrev(size7, "rev");
  care::host_device_ptr<double> threshold7(size7, "thresh7");

  CARE_GPU_KERNEL {
     subset1[0] = 2;

     threshold1[0] = 10;

     subset3[0] = 5;
     subset3[1] = 0;
     subset3[2] = 6;

     threshold3[0] = 0;
     threshold3[1] = 10;
     threshold3[2] = 10;

     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 5;
     a[5] = 3;
     a[6] = 7;

     subsetrev[0] = 6;
     subsetrev[1] = 5;
     subsetrev[2] = 4;
     subsetrev[3] = 3;
     subsetrev[4] = 2;
     subsetrev[5] = 1;
     subsetrev[6] = 0;

     threshold7[0] = 10;
     threshold7[1] = 10;
     threshold7[2] = 10;
     threshold7[3] = 0;
     threshold7[4] = 0;
     threshold7[5] = 0;
     threshold7[6] = 10;
  } CARE_GPU_KERNEL_END

  // null subset
  double cutoff = -10;
  int threshIdx = -1;
  int result = care::FindIndexMaxSubsetAboveThresholds<int>(a, nullptr, 0, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx, -1);

  // all elements but reverse order, no threshold
  result = care::FindIndexMaxSubsetAboveThresholds<int>(a, subsetrev, size7, nullptr, cutoff, &threshIdx);
  EXPECT_EQ(result, 3);

  // all elements in reverse order, cutoff not triggered
  result = care::FindIndexMaxSubsetAboveThresholds<int>(a, subsetrev, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, 3); // this is the index in the original array
  EXPECT_EQ(threshIdx, 3); // this is the index in the subset

  // change the cutoff
  cutoff = 5.0;
  result = care::FindIndexMaxSubsetAboveThresholds<int>(a, subsetrev, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, 6); // this is the index in the original array
  EXPECT_EQ(threshIdx, 0); // this is the index in the subset

  // len 3 subset
  cutoff = 5.0;
  result = care::FindIndexMaxSubsetAboveThresholds<int>(a, subset3, size3, threshold3, cutoff, &threshIdx);
  EXPECT_EQ(result, 6);
  EXPECT_EQ(threshIdx, 2);

  // len 1 subset
  cutoff = 5.0;
  result = care::FindIndexMaxSubsetAboveThresholds<int>(a, subset1, size1, threshold1, cutoff, &threshIdx);
  EXPECT_EQ(result, 2);
  EXPECT_EQ(threshIdx, 0);

  // len 1 subset not found (cutoff too high)
  cutoff = 25.67;
  result = care::FindIndexMaxSubsetAboveThresholds<int>(a, subset1, size1, threshold1, cutoff, &threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx, -1);

  threshold7.free();
  subsetrev.free();
  a.free();
  threshold3.free();
  subset3.free();
  threshold1.free();
  subset1.free();
}

GPU_TEST(algorithm, pickperformfindmax)
{
  const int size1 = 1;
  care::host_device_ptr<int> subset1(size1, "sub1");
  care::host_device_ptr<double> threshold1(size1, "thresh1");

  const int size3 = 3;
  care::host_device_ptr<int> subset3(size3, "sub3");
  care::host_device_ptr<double> threshold3(size3, "thresh3");

  const int size7 = 7;
  care::host_device_ptr<int> a(size7, "arrseven");
  care::host_device_ptr<int> subsetrev(size7, "rev");
  care::host_device_ptr<double> threshold7(size7, "thresh7");
  care::host_device_ptr<int> mask(size7, "mask");

  CARE_GPU_KERNEL {
     subset1[0] = 2;

     threshold1[0] = 10;

     subset3[0] = 5;
     subset3[1] = 0;
     subset3[2] = 6;

     threshold3[0] = 0;
     threshold3[1] = 10;
     threshold3[2] = 10;

     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 5;
     a[5] = 3;
     a[6] = 7;

     subsetrev[0] = 6;
     subsetrev[1] = 5;
     subsetrev[2] = 4;
     subsetrev[3] = 3;
     subsetrev[4] = 2;
     subsetrev[5] = 1;
     subsetrev[6] = 0;

     threshold7[0] = 10;
     threshold7[1] = 10;
     threshold7[2] = 10;
     threshold7[3] = 0;
     threshold7[4] = 0;
     threshold7[5] = 0;
     threshold7[6] = 10;

     mask[0] = 0;
     mask[1] = 0;
     mask[2] = 0;
     mask[3] = 1;
     mask[4] = 0;
     mask[5] = 0;
     mask[6] = 0;
  } CARE_GPU_KERNEL_END

  // all elements in reverse order, no mask or subset
  double cutoff = -10;
  int threshIdx = -1;
  int result = care::PickAndPerformFindMaxIndex<int>(a, nullptr, nullptr, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, 3);
  EXPECT_EQ(threshIdx, 3);

  // all elements in reverse order, no subset
  result = care::PickAndPerformFindMaxIndex<int>(a, mask, nullptr, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, -1); // This is -1 because it is masked off
  EXPECT_EQ(threshIdx, 3); // NOTE: even if a value is masked off, the threshold index still comes back

  // all elements but reverse order, no threshold
  result = care::PickAndPerformFindMaxIndex<int>(a, mask, subsetrev, size7, nullptr, cutoff, &threshIdx);
  EXPECT_EQ(result, -1); //masked off

  // all elements in reverse order, cutoff not triggered
  // NOTE: even though the element is masked off, threshIdx is valid!!! Just the return index is -1, not the threshIdx!!!!
  result = care::PickAndPerformFindMaxIndex<int>(a, mask, subsetrev, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, -1); // this is the index in the original array
  EXPECT_EQ(threshIdx, 3); // this is the index in the subset

  // change the cutoff
  cutoff = 5.0;
  result = care::PickAndPerformFindMaxIndex<int>(a, mask, subsetrev, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, 6); // this is the index in the original array

  // len 3 subset
  cutoff = 5.0;
  result = care::PickAndPerformFindMaxIndex<int>(a, mask, subset3, size3, threshold3, cutoff, &threshIdx);
  EXPECT_EQ(result, 6);
  EXPECT_EQ(threshIdx, 2);

  // len 1 subset
  cutoff = 5.0;
  result = care::PickAndPerformFindMaxIndex<int>(a, mask, subset1, size1, threshold1, cutoff, &threshIdx);
  EXPECT_EQ(result, 2);
  EXPECT_EQ(threshIdx, 0);

  // len 1 subset not found (cutoff too high)
  cutoff = 25.67;
  result = care::PickAndPerformFindMaxIndex<int>(a, mask, subset1, size1, threshold1, cutoff, &threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx, -1);

  mask.free();
  threshold7.free();
  subsetrev.free();
  a.free();
  threshold3.free();
  subset3.free();
  threshold1.free();
  subset1.free();
}

GPU_TEST(algorithm, pickperformfindmin)
{
  const int size1 = 1;
  care::host_device_ptr<int> subset1(size1, "sub1");
  care::host_device_ptr<double> threshold1(size1, "thresh1");

  const int size3 = 3;
  care::host_device_ptr<int> subset3(size3, "sub3");
  care::host_device_ptr<double> threshold3(size3, "thresh3");

  const int size7 = 7;
  care::host_device_ptr<int> a(size7, "arrseven");
  care::host_device_ptr<int> subsetrev(size7, "rev");
  care::host_device_ptr<double> threshold7(size7, "thresh7");
  care::host_device_ptr<int> mask(size7, "mask");

  CARE_GPU_KERNEL {
     subset1[0] = 2;

     threshold1[0] = 10;

     subset3[0] = 5;
     subset3[1] = 0;
     subset3[2] = 6;

     threshold3[0] = 0;
     threshold3[1] = 10;
     threshold3[2] = 10;

     a[0] = 2;
     a[1] = 1;
     a[2] = 1;
     a[3] = 8;
     a[4] = 5;
     a[5] = 3;
     a[6] = 7;

     subsetrev[0] = 6;
     subsetrev[1] = 5;
     subsetrev[2] = 4;
     subsetrev[3] = 3;
     subsetrev[4] = 2;
     subsetrev[5] = 1;
     subsetrev[6] = 0;

     threshold7[0] = 0;
     threshold7[1] = 0;
     threshold7[2] = 0;
     threshold7[3] = 0;
     threshold7[4] = 10;
     threshold7[5] = 10;
     threshold7[6] = 0;

     mask[0] = 0;
     mask[1] = 1;
     mask[2] = 1;
     mask[3] = 0;
     mask[4] = 0;
     mask[5] = 0;
     mask[6] = 0;
  } CARE_GPU_KERNEL_END

  // all elements in reverse order, no mask or subset
  double cutoff = -10;
  int threshIdx = -1;
  int result = care::PickAndPerformFindMinIndex<int>(a, nullptr, nullptr, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, 1);
  EXPECT_EQ(threshIdx, 1);

  // all elements in reverse order, no subset
  result = care::PickAndPerformFindMinIndex<int>(a, mask, nullptr, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, -1); // This is -1 because it is masked off
  EXPECT_EQ(threshIdx, 1); // NOTE: even if a value is masked off, the threshold index still comes back

  // all elements but reverse order, no threshold
  result = care::PickAndPerformFindMinIndex<int>(a, mask, subsetrev, size7, nullptr, cutoff, &threshIdx);
  EXPECT_EQ(result, -1); //masked off

  // all elements in reverse order, cutoff not triggered
  // NOTE: even though the element is masked off, threshIdx is valid!!! Just the return index is -1, not the threshIdx!!!!
  result = care::PickAndPerformFindMinIndex<int>(a, mask, subsetrev, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, -1); // this is the index in the original array
  EXPECT_EQ(threshIdx, 4); // this is the index in the subset

  // change the cutoff
  // CURRENTLY TEST GIVES DIFFERENT RESULTS WHEN COMPILED RELEASE VS DEBUG!!!
  cutoff = 5.0;
  result = care::PickAndPerformFindMinIndex<int>(a, nullptr, subsetrev, size7, threshold7, cutoff, &threshIdx);
  EXPECT_EQ(result, 2); // this is the index in the original array
  EXPECT_EQ(threshIdx, 4);

  // len 3 subset
  cutoff = 5.0;
  result = care::PickAndPerformFindMinIndex<int>(a, mask, subset3, size3, threshold3, cutoff, &threshIdx);
  EXPECT_EQ(result, 0);
  EXPECT_EQ(threshIdx, 1);

  // len 1 subset, masked off
  cutoff = 5.0;
  result = care::PickAndPerformFindMinIndex<int>(a, mask, subset1, size1, threshold1, cutoff, &threshIdx);
  EXPECT_EQ(result, -1);
  EXPECT_EQ(threshIdx, 0);

  mask.free();
  threshold7.free();
  subsetrev.free();
  a.free();
  threshold3.free();
  subset3.free();
  threshold1.free();
  subset1.free();
}

GPU_TEST(algorithm, arraycount)
{
  const int size = 7;
  care::host_device_ptr<int> a(size, "arrseven");
  care::host_device_ptr<int> b = nullptr;

  CARE_GPU_KERNEL {
     a[0] = 0;
     a[1] = 1;
     a[2] = 0;
     a[3] = 3;
     a[4] = 4;
     a[5] = 0;
     a[6] = 6;
  } CARE_GPU_KERNEL_END
  
  // count element in array
  int result = care::ArrayCount<int>(a, size, 0);
  EXPECT_EQ(result, 3);

  result = care::ArrayCount<int>(a, size, 6);
  EXPECT_EQ(result, 1);

  // not in the array
  result = care::ArrayCount<int>(a, size, -1);
  EXPECT_EQ(result, 0);

  // the null array test
  result = care::ArrayCount<int>(b, 0, 0);
  EXPECT_EQ(result, 0);

  b.free();
  a.free();
}

GPU_TEST(algorithm, arraysum)
{
  const int size1 = 1;
  const int size7 = 7;
  care::host_device_ptr<int> a(size7, "arrseven");
  care::host_device_ptr<int> b(size1, "arrone");
  care::host_device_ptr<int> nil = nullptr;

  CARE_GPU_KERNEL {
     a[0] = 0;
     a[1] = 1;
     a[2] = 0;
     a[3] = 3;
     a[4] = 4;
     a[5] = 0;
     a[6] = 6;

     b[0] = 99;
  } CARE_GPU_KERNEL_END

  // sum everything
  int result = care::ArraySum<int>(a, size7, 0);
  EXPECT_EQ(result, 14);

  // sum everything with initval -12
  result = care::ArraySum<int>(a, size7, -12);
  EXPECT_EQ(result, 2);

  // sum first 4 elems of length 7 array
  result = care::ArraySum<int>(a, 4, 0);
  EXPECT_EQ(result, 4);

  // length 1 array
  result = care::ArraySum<int>(b, size1, 0);
  EXPECT_EQ(result, 99);

  // the null array test
  result = care::ArraySum<int>(nil, 0, 0);
  EXPECT_EQ(result, 0);
 
  // null array, different init val
  result = care::ArraySum<int>(nil, 0, -5);
  EXPECT_EQ(result, -5);

  nil.free();
  b.free();
  a.free();
}

GPU_TEST(algorithm, arraysumsubset)
{
  const int size1 = 1;
  care::host_device_ptr<int> b(size1, "arrone");
  care::host_device_ptr<int> sub1(size1, "subb");

  const int size3 = 3;
  care::host_device_ptr<int> sub3(size3, "suba");

  const int size7 = 7;
  care::host_device_ptr<int> a(size7, "arrseven");

  care::host_device_ptr<int> nil = nullptr;

  CARE_GPU_KERNEL {
     a[0] = 0;
     a[1] = 1;
     a[2] = 0;
     a[3] = 3;
     a[4] = 4;
     a[5] = 0;
     a[6] = 6;

     b[0] = 99;

     sub3[0] = 3;
     sub3[1] = 1;
     sub3[2] = 4;

     sub1[0] = 0;
  } CARE_GPU_KERNEL_END

  // sum subset
  int result = care::ArraySumSubset<int, int>(a, sub3, size3, 0);
  EXPECT_EQ(result, 8);

  // sum everything with initval -12
  result = care::ArraySumSubset<int, int>(a, sub3, size3, -12);
  EXPECT_EQ(result, -4);

  // sum first 2 elements from subset subset
  result = care::ArraySumSubset<int, int>(a, sub3, 2, 0);
  EXPECT_EQ(result, 4);

  // length 1 array
  result = care::ArraySumSubset<int, int>(b, sub1, size1, 0);
  EXPECT_EQ(result, 99);

  // the null array test
  result = care::ArraySumSubset<int, int>(nil, nil, 0, 0);
  EXPECT_EQ(result, 0);

  // null array, different init val
  result = care::ArraySumSubset<int, int>(nil, nil, 0, -5);
  EXPECT_EQ(result, -5);

  nil.free();
  a.free();
  sub3.free();
  sub1.free();
  b.free();
}

GPU_TEST(algorithm, arraysummaskedsubset)
{ 
  const int size0 = 0;
  care::host_device_ptr<int> nil = nullptr;

  const int size1 = 1;
  care::host_device_ptr<int> b(size1, "arrone");
  care::host_device_ptr<int> sub1(size1, "subb");

  const int size3 = 3;
  care::host_device_ptr<int> sub3(size3, "suba");

  const int size7 = 7;
  care::host_device_ptr<int> a(size7, "arrseven");
  care::host_device_ptr<int> mask(size7, "mask");

  CARE_GPU_KERNEL {
     a[0] = 0;
     a[1] = 1;
     a[2] = 0;
     a[3] = 3;
     a[4] = 4;
     a[5] = 0;
     a[6] = 6;

     b[0] = 99;

     sub3[0] = 3;
     sub3[1] = 1;
     sub3[2] = 4;

     sub1[0] = 0;

     mask[0] = 1;
     mask[1] = 1;
     mask[2] = 0;
     mask[3] = 1;
     mask[4] = 0;
     mask[5] = 0;
     mask[6] = 0;
  } CARE_GPU_KERNEL_END

  // sum subset
  int result = care::ArrayMaskedSumSubset<int, int>(a, mask, sub3, size3, 0);
  EXPECT_EQ(result, 4);
  
  // sum first 2 elements from subset subset
  result = care::ArrayMaskedSumSubset<int, int>(a, mask, sub3, 2, 0);
  EXPECT_EQ(result, 0);
  
  // length 1 array
  result = care::ArrayMaskedSumSubset<int, int>(b, mask, sub1, size1, 0);
  EXPECT_EQ(result, 0);
  
  // the null array test
  result = care::ArrayMaskedSumSubset<int, int>(nil, mask, nil, size0, 0);
  EXPECT_EQ(result, 0);

  mask.free();
  a.free();
  sub3.free();
  sub1.free();
  b.free();
  nil.free();
}

GPU_TEST(algorithm, findindexgt)
{
  const int size = 7;
  care::host_device_ptr<int> a(size, "arrseven");

  CARE_GPU_KERNEL {
     a[0] = 0;
     a[1] = 1;
     a[2] = 0;
     a[3] = 3;
     a[4] = 4;
     a[5] = 0;
     a[6] = 6;
  } CARE_GPU_KERNEL_END

  // in practice it finds the element that is the highest above the limit, so 6
  // But in theory a different implementation could give the first element above the limit.
  int result = care::FindIndexGT<int>(a, size, 3);
  ASSERT_TRUE(result==4 || result==6);

  result = care::FindIndexGT<int>(a, 7, 0);
  ASSERT_TRUE(result==1 || result==3 || result==4 || result==6);

  // limit is higher than all elements
  result = care::FindIndexGT<int>(a, 7, 99);
  EXPECT_EQ(result, -1);

  a.free();
}

GPU_TEST(algorithm, findindexmax)
{
  care::host_device_ptr<int> a(7, "arrseven");
  care::host_device_ptr<int> b(3, "arrthree");
  care::host_device_ptr<int> c(2, "arrtwo");

  CARE_GPU_KERNEL {
     a[0] = 0;
     a[1] = 1;
     a[2] = 0;
     a[3] = 3;
     a[4] = 99;
     a[5] = 0;
     a[6] = 6;

     b[0] = 9;
     b[1] = 10;
     b[2] = -2;

     c[0] = 5;
     c[1] = 5;
  } CARE_GPU_KERNEL_END

  int result = care::FindIndexMax<int>(a, 7);
  EXPECT_EQ(result, 4);

  result = care::FindIndexMax<int>(b, 3);
  EXPECT_EQ(result, 1);

  result = care::FindIndexMax<int>(c, 2);
  EXPECT_EQ(result, 0);

  // nil test
  result = care::FindIndexMax<int>(nullptr, 0);
  EXPECT_EQ(result, -1);

  c.free();
  b.free();
  a.free();
}

// duplicating and copying arrays
// NOTE: no test for when to and from are the same array or aliased. I'm assuming that is not allowed.
GPU_TEST(algorithm, dup_and_copy) {
  const int size = 7;
  care::host_device_ptr<int> to1(size, "zeros1");
  care::host_device_ptr<int> to2(size, "zeros2");
  care::host_device_ptr<int> to3(size, "zeros3");
  care::host_device_ptr<int> from(size, "from7");
  care::host_device_ptr<int> nil = nullptr;

  CARE_GPU_KERNEL {
     from[0] = 9;
     from[1] = 10;
     from[2] = -2;
     from[3] = 67;
     from[4] = 9;
     from[5] = 45;
     from[6] = -314;
  } CARE_GPU_KERNEL_END

  CARE_GPU_LOOP(i, 0, size) {
     to1[i] = 0;
     to2[i] = 0;
     to3[i] = 0;
  } CARE_GPU_LOOP_END

  // duplicate and check that elements are the same
  care::host_device_ptr<int> dup = care::ArrayDup<int>(from, size);
  RAJAReduceMin<bool> passeddup{true};

  CARE_REDUCE_LOOP(i, 0, size) {
    if (dup[i] != from[i]) {
      passeddup.min(false);
    }
  } CARE_REDUCE_LOOP_END

  ASSERT_TRUE((bool) passeddup);

  // duplicating nullptr should give nullptr
  care::host_device_ptr<int> dupnil = care::ArrayDup<int>(nil, 0);
  EXPECT_EQ(dupnil, nullptr);

  // copy and check that elements are the same
  care::ArrayCopy<int>(to1, from, size);
  RAJAReduceMin<bool> passed1{true};

  CARE_REDUCE_LOOP(i, 0, size) {
    if (to1[i] != from[i]) {
      passed1.min(false);
    }
  } CARE_REDUCE_LOOP_END

  ASSERT_TRUE((bool) passed1);

  // copy 2 elements, testing different starting points
  care::ArrayCopy<int>(to2, from, 2, 3, 4);
  RAJAReduceMin<bool> passed2{true};

  CARE_REDUCE_LOOP(i, 0, 1) {
    if (to2[0] != 0 || to2[1] != 0 || to2[2] != 0 || to2[5] != 0 || to2[6] != 0) {
      passed2.min(false);
    }

    if (to2[3] != 9) {
      passed2.min(false);
    }

    if (to2[4] != 45) {
      passed2.min(false);
    }
  } CARE_REDUCE_LOOP_END

  ASSERT_TRUE((bool) passed2);

  // copy 2 elements, testing different starting points
  care::ArrayCopy<int>(to3, from, 2, 4, 3);
  RAJAReduceMin<bool> passed3{true};

  CARE_REDUCE_LOOP(i, 0, 1) {
    if (to3[0] != 0 || to3[1] != 0 || to3[2] != 0 || to3[3] != 0 || to3[6] != 0) {
      passed3.min(false);
    }

    if (to3[4] != 67) {
      passed3.min(false);
    }

    if (to3[5] != 9) {
      passed3.min(false);
    }
  } CARE_REDUCE_LOOP_END

  ASSERT_TRUE((bool) passed3);
  dupnil.free();
  dup.free();
  nil.free();
  from.free();
  to3.free();
  to2.free();
  to1.free();
}

GPU_TEST(algorithm, intersectarrays) {
   care::host_device_ptr<int> a(3, "a");
   care::host_device_ptr<int> b(5, "b");
   care::host_device_ptr<int> c(7, "c");
   care::host_device_ptr<int> d(9, "d");

  CARE_GPU_KERNEL {
     a[0] = 1;
     a[1] = 2;
     a[2] = 5;

     b[0] = 2;
     b[1] = 3;
     b[2] = 4;
     b[3] = 5;
     b[4] = 6;

     c[0] = -1;
     c[1] = 0;
     c[2] = 2;
     c[3] = 3;
     c[4] = 6;
     c[5] = 120;
     c[6] = 360;

     d[0] = 1001;
     d[1] = 1002;
     d[2] = 2003;
     d[3] = 3004;
     d[4] = 4005;
     d[5] = 5006;
     d[6] = 6007;
     d[7] = 7008;
     d[8] = 8009;
  } CARE_GPU_KERNEL_END

   care::host_device_ptr<int> matches1, matches2;
   int numMatches[1] = {77};

   // nil test
   care::IntersectArrays<int>(RAJAExec(), c, 7, 0, nullptr, 0, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0);

   matches2.free();
   matches1.free();

   // intersect c and b
   care::IntersectArrays<int>(RAJAExec(), c, 7, 0, b, 5, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 3);
   EXPECT_EQ(matches1.pick(0), 2);
   EXPECT_EQ(matches1.pick(1), 3);
   EXPECT_EQ(matches1.pick(2), 4);
   EXPECT_EQ(matches2.pick(0), 0);
   EXPECT_EQ(matches2.pick(1), 1);
   EXPECT_EQ(matches2.pick(2), 4);

   matches2.free();
   matches1.free();

   // introduce non-zero starting locations. In this case, matches are given as offsets from those starting locations
   // and not the zero position of the arrays.
   care::IntersectArrays<int>(RAJAExec(), c, 4, 3, b, 4, 1, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 2);
   EXPECT_EQ(matches1.pick(0), 0);
   EXPECT_EQ(matches1.pick(1), 1);
   EXPECT_EQ(matches2.pick(0), 0);
   EXPECT_EQ(matches2.pick(1), 3);

   matches2.free();
   matches1.free();

   // intersect a and b
   care::IntersectArrays<int>(RAJAExec(), a, 3, 0, b, 5, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 2);
   EXPECT_EQ(matches1.pick(0), 1);
   EXPECT_EQ(matches1.pick(1), 2);
   EXPECT_EQ(matches2.pick(0), 0);
   EXPECT_EQ(matches2.pick(1), 3);

   matches2.free();
   matches1.free();

   // no matches
   care::IntersectArrays<int>(RAJAExec(), a, 3, 0, d, 9, 0, matches1, matches2, numMatches);
   EXPECT_EQ(numMatches[0], 0);

   matches2.free();
   matches1.free();

   d.free();
   c.free();
   b.free();
   a.free();
}

GPU_TEST(algorithm, binarsearchhostdev) {
   const int size = 7;
   care::host_device_ptr<int> aptr(size, "asorted");

   CARE_GPU_KERNEL {
      aptr[0] = -9;
      aptr[1] = 0;
      aptr[2] = 3;
      aptr[3] = 7;
      aptr[4] = 77;
      aptr[5] = 500;
      aptr[6] = 999;
   } CARE_GPU_KERNEL_END

   const int result = care::BinarySearch<int>(aptr, 0, size, 77, false);
   EXPECT_EQ(result, 4);

   aptr.free();
}

// test insertion sort and also sortLocal (which currently uses InsertionSort),
// and then unique the result
GPU_TEST(algorithm, localsortunique) {
   // set up arrays
   const int size = 4;
   care::host_device_ptr<int> aptr(size, "unsorta");
   care::host_device_ptr<int> bptr(size, "unsortb");

   CARE_GPU_KERNEL {
      aptr[0] = 4;
      aptr[1] = 0;
      aptr[2] = 2;
      aptr[3] = 0;

      bptr[0] = 4;
      bptr[1] = 0;
      bptr[2] = 2;
      bptr[3] = 0;
   } CARE_GPU_KERNEL_END
 
   // sort on local ptrs  
   CARE_STREAM_LOOP(i, 0, 1) {
     care::local_ptr<int> aloc = aptr;
     care::local_ptr<int> bloc = bptr;

     care::sortLocal(aloc, size);
     care::InsertionSort(bloc, size);
   } CARE_STREAM_LOOP_END

   // check results of sortLocal
   EXPECT_EQ(aptr.pick(0), 0);
   EXPECT_EQ(aptr.pick(1), 0);
   EXPECT_EQ(aptr.pick(2), 2);
   EXPECT_EQ(aptr.pick(3), 4);

   // test results for InsertionSort
   EXPECT_EQ(bptr.pick(0), 0);
   EXPECT_EQ(bptr.pick(1), 0);
   EXPECT_EQ(bptr.pick(2), 2);
   EXPECT_EQ(bptr.pick(3), 4);

   // perform unique. Should eliminate the extra 0, give a length
   // of one less (for the deleted duplicate number)
   RAJAReduceMin<int> newlen{size};

   CARE_REDUCE_LOOP(i, 0, 1) {
     int len = size;
     care::local_ptr<int> aloc = aptr;
     care::uniqLocal(aloc, len);
     newlen.min(len);
   } CARE_REDUCE_LOOP_END

   EXPECT_EQ((int)newlen, 3);
   EXPECT_EQ(aptr.pick(0), 0);
   EXPECT_EQ(aptr.pick(1), 2);
   EXPECT_EQ(aptr.pick(2), 4);

   bptr.free();
   aptr.free();
}

#endif // CARE_GPUCC

