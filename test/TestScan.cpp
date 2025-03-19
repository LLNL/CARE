//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

#include "gtest/gtest.h"

#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/scan.h"
#include "care/Setup.h"
#include "care/detail/test_utils.h"



GPU_TEST(forall, Initialization) {
   printf("Initializing\n");
   init_care_for_testing();
   printf("Initialized... Testing Scan\n");
}

GPU_TEST(Scan, test_scan_offset) {
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
