//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

#define GPU_ACTIVE
#include "gtest/gtest.h"

#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/scan.h"
#include "care/Setup.h"

// This makes it so we can use device lambdas from within a GPU_TEST
#define GPU_TEST(X, Y) static void gpu_test_ ## X_ ## Y(); \
   TEST(X, Y) { gpu_test_ ## X_ ## Y(); } \
   static void gpu_test_ ## X_ ## Y()

using int_ptr = chai::ManagedArray<int>;

GPU_TEST(Scan, test_scan_offset) {
#if defined(CARE_GPUCC)
   int poolSize = 128*1024*1024; // 128 MB
   care::initialize_pool("PINNED", "PINNED_POOL", chai::PINNED, poolSize, poolSize ,true);
   care::initialize_pool("DEVICE", "DEVICE_POOL", chai::GPU, poolSize, poolSize, true);
#endif

   const int starting_offset = 17;
   int offset = starting_offset;
   int length = 20;
   int_ptr scan_Result(length);

   // test offset
   SCAN_LOOP(i,0,length,pos,offset,true) {
      scan_Result[i] = pos;
   } SCAN_LOOP_END(length,pos,offset)

   EXPECT_EQ(offset,starting_offset+length);

   CARE_SEQUENTIAL_LOOP(i,0,length) {
      EXPECT_EQ(scan_Result[i],starting_offset+i);
   } CARE_SEQUENTIAL_LOOP_END
}

GPU_TEST(Scan, test_scan_zero_length) {
   const int starting_offset = 17;
   int offset = starting_offset;
   int length = 0;
   int_ptr scan_Result(length);

   // test offset
   SCAN_LOOP(i,0,length,pos,offset,true) {
      
      scan_Result[i] = pos;
   } SCAN_LOOP_END(length,pos,offset)

   EXPECT_EQ(offset,starting_offset+length);

   CARE_SEQUENTIAL_LOOP(i,0,length) {
      EXPECT_EQ(scan_Result[i],starting_offset+i);
   } CARE_SEQUENTIAL_LOOP_END
}

GPU_TEST(Scan, test_scan_offset_index) {
   const int starting_offset = 17;
   int offset = starting_offset;
   int length = 20;
   int start = 5;
   int end = 25;
   int_ptr scan_Result(length);

   // test offset
   SCAN_LOOP(i,start,end,pos,offset,true) {
      scan_Result[i-start] = pos;
   } SCAN_LOOP_END(length,pos,offset)

   EXPECT_EQ(offset,starting_offset+length);

   CARE_SEQUENTIAL_LOOP(i,0,length) {
      EXPECT_EQ(scan_Result[i],starting_offset+i);
   } CARE_SEQUENTIAL_LOOP_END
}

GPU_TEST(Scan, test_scan_offset_index_half) {
   const int starting_offset = 17;
   int offset = starting_offset;
   int length = 20;
   int start = 5;
   int end = 25;
   int_ptr scan_Result(length);

   // test offset
   SCAN_LOOP(i,start,end,pos,offset,i%2 == 0) {
      scan_Result[i-start] = pos;
   } SCAN_LOOP_END(length,pos,offset)

   EXPECT_EQ(offset,starting_offset+length/2);

   CARE_SEQUENTIAL_LOOP(i,0,length) {
      if ((i+start)%2 == 0) {
         EXPECT_EQ(scan_Result[i],starting_offset+i/2);
      }
   } CARE_SEQUENTIAL_LOOP_END
}

GPU_TEST(Scan, test_scan_everywhere) {
   const int starting_offset = 17;
   int offset = starting_offset;
   int length = 20;
   int start = 5;
   int end = 25;
   int_ptr scan_Result(length);

   // test offset
   SCAN_EVERYWHERE_LOOP(i,start,end,pos,offset,true) {
      scan_Result[i-start] = pos;
   } SCAN_EVERYWHERE_LOOP_END(length,pos,offset)

   EXPECT_EQ(offset,starting_offset+length);

   CARE_SEQUENTIAL_LOOP(i,0,length) {
      EXPECT_EQ(scan_Result[i],starting_offset+i);
   } CARE_SEQUENTIAL_LOOP_END
}

#if CARE_HAVE_LLNL_GLOBALID

using globalID_ptr = chai::ManagedArray<globalID>;

GPU_TEST(Scan, test_scan_offset_index_gid) {
   const int starting_offset = 13;
   globalID offset(starting_offset);
   int length = 20;
   int start = 5;
   int end = 25;
   globalID_ptr scan_Result(length);

   // test offset
   SCAN_LOOP_GID(i,start,end,pos,offset,true) {
      scan_Result[i-start] = pos;
   } SCAN_LOOP_GID_END(length,pos,offset)

   EXPECT_EQ(offset.Ref(),starting_offset+length);

   CARE_SEQUENTIAL_LOOP(i,0,length) {
      EXPECT_EQ(scan_Result[i].Ref(),starting_offset+i);
   } CARE_SEQUENTIAL_LOOP_END
}

#endif
