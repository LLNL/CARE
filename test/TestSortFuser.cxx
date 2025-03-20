//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

#include "gtest/gtest.h"

#include "care/Setup.h"
#include "care/SortFuser.h"
#include "care/detail/test_utils.h"

using namespace care;

GPU_TEST(TestPacker, gpu_initialization) {
   printf("Initializing\n");
   init_care_for_testing();
   printf("Initialized... Testing care::SortFuser\n");
}

GPU_TEST(TestPacker, testFuseSort) {
   int N = 5; 
   int_ptr arr1(N); 
   int_ptr arr2(N);
   CARE_STREAM_LOOP(i,0,N) {
      arr1[i] = N-1-i;
      arr2[i] = N+3-i;
   } CARE_STREAM_LOOP_END

   SortFuser<int> sorter = SortFuser<int>();
   sorter.reset();
   sorter.fusibleSortArray(arr1,N, 5); 
   sorter.fusibleSortArray(arr2,N, 9); 
   sorter.sort();
   int_ptr concatanated = sorter.getConcatenatedResult();
   
   CARE_SEQUENTIAL_LOOP(i,0,N) {
      EXPECT_EQ(arr1[i],concatanated[i]);
      EXPECT_EQ(arr2[i],concatanated[i+N]);
      EXPECT_EQ(arr1[i],i);
      EXPECT_EQ(arr2[i],i+4);
   } CARE_SEQUENTIAL_LOOP_END

}

GPU_TEST(TestPacker, testFuseUniq) {
   int N = 5; 
   int_ptr arr1(N); 
   int_ptr arr2(N);
   CARE_STREAM_LOOP(i,0,N) {
      arr1[i] = i - i%2;
      arr2[i] = i + N/2- i%2; 
   } CARE_STREAM_LOOP_END

   int_ptr out1,out2;
   int len1, len2;
   SortFuser<int> sorter = SortFuser<int>();
   sorter.reset();
   sorter.fusibleUniqArray(arr1,N,10,out1,len1); 
   sorter.fusibleUniqArray(arr2,N,10,out2,len2); 
   sorter.uniq();
   int_ptr concatanated = sorter.getConcatenatedResult();
   int_ptr concatanated_lengths = sorter.getConcatenatedLengths();
   
   EXPECT_EQ(len1,3); 
   CARE_SEQUENTIAL_LOOP(i,0,len1) {
      EXPECT_EQ(out1[i],concatanated[i]);
      EXPECT_EQ(out1[i],i*2);
   } CARE_SEQUENTIAL_LOOP_END
   
   EXPECT_EQ(len2,3); 
   CARE_SEQUENTIAL_LOOP(i,0,len2) {
      EXPECT_EQ(out2[i],concatanated[i+len1]);
      EXPECT_EQ(out2[i],i*2+N/2);
   } CARE_SEQUENTIAL_LOOP_END

   EXPECT_EQ(concatanated_lengths.pick(0), len1);
   EXPECT_EQ(concatanated_lengths.pick(1), len2);
}

GPU_TEST(TestPacker, testFuseSortUniq) {
   int N = 5; 
   int_ptr arr1(N); 
   int_ptr arr2(N);
   CARE_STREAM_LOOP(j,0,N) {
      int i = N-1 -j;
      arr1[j] = i - i%2;
      arr2[j] = i + N/2- i%2; 
   } CARE_STREAM_LOOP_END

   int_ptr out1,out2;
   int len1, len2;
   SortFuser<int> sorter = SortFuser<int>();
   sorter.reset();
   sorter.fusibleSortUniqArray(arr1,N,10,out1,len1); 
   sorter.fusibleSortUniqArray(arr2,N,10,out2,len2); 
   sorter.sortUniq();
   int_ptr concatanated = sorter.getConcatenatedResult();
   int_ptr concatanated_lengths = sorter.getConcatenatedLengths();
   
   EXPECT_EQ(len1,3); 
   CARE_SEQUENTIAL_LOOP(i,0,len1) {
      EXPECT_EQ(out1[i],concatanated[i]);
      EXPECT_EQ(out1[i],i*2);
   } CARE_SEQUENTIAL_LOOP_END
   
   EXPECT_EQ(len2,3); 
   CARE_SEQUENTIAL_LOOP(i,0,len2) {
      EXPECT_EQ(out2[i],concatanated[i+len1]);
      EXPECT_EQ(out2[i],i*2+N/2);
   } CARE_SEQUENTIAL_LOOP_END

   EXPECT_EQ(concatanated_lengths.pick(0), len1);
   EXPECT_EQ(concatanated_lengths.pick(1), len2);
}


GPU_TEST(TestPacker, testFuseSortUniqMissingArrays) {
   int a0[3] = {15,16,16};
   int a1[18] = {5,6,6,7,7,8,8,10,11,11,12,12,13,13,17,17,18,18}; 
   int a2[3] = {15,16,16};
   int a3[18] = {5,6,6,7,7,8,8,10,11,11,12,12,13,13,17,17,18,18};
   /* int a4 */
   int a5[6] = {1,2,4,5,7,8};
   /* int a6; */
   int a7[6] = {1,2,4,5,6,8};

   int_ptr a0_ptr = int_ptr(a0,3,"a0");
   int_ptr a1_ptr = int_ptr(a1,18,"a1");
   int_ptr a2_ptr = int_ptr(a2,3,"a2");
   int_ptr a3_ptr = int_ptr(a3,18,"a3");
   int_ptr a4_ptr = nullptr;
   int_ptr a5_ptr = int_ptr(a5,6,"a5");
   int_ptr a6_ptr = nullptr;
   int_ptr a7_ptr = int_ptr(a7,6,"a6");
   int_ptr a8_ptr = nullptr;
   int_ptr a9_ptr = nullptr;
   
   SortFuser<int> sorter = SortFuser<int>();
   sorter.reset();
   int_ptr a0_out, a1_out,a2_out,a3_out,a4_out,a5_out,a6_out,a7_out,a8_out,a9_out;
   int a0_len, a1_len,a2_len,a3_len,a4_len,a5_len,a6_len,a7_len,a8_len,a9_len;
   sorter.fusibleSortUniqArray(a0_ptr,3,19,a0_out,a0_len);
   sorter.fusibleSortUniqArray(a1_ptr,18,19,a1_out,a1_len);
   sorter.fusibleSortUniqArray(a2_ptr,3,19,a2_out,a2_len);
   sorter.fusibleSortUniqArray(a3_ptr,18,19,a3_out,a3_len);
   sorter.fusibleSortUniqArray(a4_ptr,0,19,a4_out,a4_len);
   sorter.fusibleSortUniqArray(a5_ptr,6,19,a5_out,a5_len);
   sorter.fusibleSortUniqArray(a6_ptr,0,19,a6_out,a6_len);
   sorter.fusibleSortUniqArray(a7_ptr,6,19,a7_out,a7_len);
   sorter.fusibleSortUniqArray(a8_ptr,0,19,a8_out,a8_len);
   sorter.fusibleSortUniqArray(a9_ptr,0,19,a9_out,a9_len);
   sorter.sortUniq();

   EXPECT_EQ(a0_len,2);
   EXPECT_EQ(a1_len,10);
   EXPECT_EQ(a2_len,2);
   EXPECT_EQ(a3_len,10);
   EXPECT_EQ(a4_len,0);
   EXPECT_EQ(a5_len,6);
   EXPECT_EQ(a6_len,0);
   EXPECT_EQ(a7_len,6);
   EXPECT_EQ(a8_len,0);
   EXPECT_EQ(a9_len,0);
}

