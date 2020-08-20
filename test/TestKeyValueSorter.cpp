//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

// Makes LOOP_REDUCE run on the device
#if defined(__GPUCC__)
#define GPU_ACTIVE
#endif

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/KeyValueSorter.h"
#include "care/care.h"

TEST(KeyValueSorter, SizeConstructor)
{
   int length = 5;
   int data[5] = {4, 1, 2, 0, 3};
   care::KeyValueSorter<int, RAJA::seq_exec> sorter(length);

   LOOP_SEQUENTIAL(i, 0, length) {
      sorter.setKey(i, i);
      sorter.setValue(i, data[i]);
   } LOOP_SEQUENTIAL_END

   LOOP_SEQUENTIAL(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), data[i]);
   } LOOP_SEQUENTIAL_END

   sorter.sort();

   CARE_HOST_KERNEL {
      EXPECT_EQ(sorter.key(0), 3);
      EXPECT_EQ(sorter.key(1), 1);
      EXPECT_EQ(sorter.key(2), 2);
      EXPECT_EQ(sorter.key(3), 4);
      EXPECT_EQ(sorter.key(4), 0);

      EXPECT_EQ(sorter.value(0), 0);
      EXPECT_EQ(sorter.value(1), 1);
      EXPECT_EQ(sorter.value(2), 2);
      EXPECT_EQ(sorter.value(3), 3);
      EXPECT_EQ(sorter.value(4), 4);
   } CARE_HOST_KERNEL_END
}

TEST(KeyValueSorter, RawArrayConstructor)
{
   int length = 5;
   int data[5] = {4, 1, 2, 0, 3};
   care::KeyValueSorter<int, RAJA::seq_exec> sorter(length, data);

   LOOP_SEQUENTIAL(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), data[i]);
   } LOOP_SEQUENTIAL_END

   sorter.sort();

   CARE_HOST_KERNEL {
      EXPECT_EQ(sorter.key(0), 3);
      EXPECT_EQ(sorter.key(1), 1);
      EXPECT_EQ(sorter.key(2), 2);
      EXPECT_EQ(sorter.key(3), 4);
      EXPECT_EQ(sorter.key(4), 0);

      EXPECT_EQ(sorter.value(0), 0);
      EXPECT_EQ(sorter.value(1), 1);
      EXPECT_EQ(sorter.value(2), 2);
      EXPECT_EQ(sorter.value(3), 3);
      EXPECT_EQ(sorter.value(4), 4);
   } CARE_HOST_KERNEL_END
}

TEST(KeyValueSorter, host_device_ptr_Constructor)
{
   int length = 5;
   care::host_device_ptr<int> data(length);

   CARE_HOST_KERNEL {
      data[0] = 4;
      data[1] = 1;
      data[2] = 2;
      data[3] = 0;
      data[4] = 3;
   } CARE_HOST_KERNEL_END

   care::KeyValueSorter<int, RAJA::seq_exec> sorter(length, data);

   LOOP_SEQUENTIAL(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), data[i]);
   } LOOP_SEQUENTIAL_END

   sorter.sort();

   CARE_HOST_KERNEL {
      EXPECT_EQ(sorter.key(0), 3);
      EXPECT_EQ(sorter.key(1), 1);
      EXPECT_EQ(sorter.key(2), 2);
      EXPECT_EQ(sorter.key(3), 4);
      EXPECT_EQ(sorter.key(4), 0);

      EXPECT_EQ(sorter.value(0), 0);
      EXPECT_EQ(sorter.value(1), 1);
      EXPECT_EQ(sorter.value(2), 2);
      EXPECT_EQ(sorter.value(3), 3);
      EXPECT_EQ(sorter.value(4), 4);
   } CARE_HOST_KERNEL_END
}

#if defined(__GPUCC__)

// Adapted from CHAI
#define GPU_TEST(X, Y) \
   static void gpu_test_##X##Y(); \
   TEST(X, gpu_test_##Y) { gpu_test_##X##Y(); } \
   static void gpu_test_##X##Y()


#endif // __GPUCC__

