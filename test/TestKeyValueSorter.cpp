//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/array.h"
#include "care/KeyValueSorter.h"
#include "care/detail/test_utils.h"

#if defined(CARE_GPUCC)
GPU_TEST(forall, Initialization) {
   printf("Initializing\n");
   init_care_for_testing();
   printf("Initialized... Testing KeyValueSorter\n");
}
#endif

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that the size constructor and manually
///        filling in the keys and values behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
TEST(KeyValueSorter, SizeConstructor)
{
   int length = 5;
   int data[5] = {4, 1, 2, 0, 3};
   care::KeyValueSorter<size_t, int, RAJA::seq_exec> sorter(length);

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      sorter.setKey(i, i);
      sorter.setValue(i, data[i]);
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), data[i]);
   } CARE_SEQUENTIAL_LOOP_END

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

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that the c-style array constructor
///        behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
TEST(KeyValueSorter, RawArrayConstructor)
{
   int length = 5;
   int data[5] = {4, 1, 2, 0, 3};
   care::KeyValueSorter<size_t, int, RAJA::seq_exec> sorter(length, data);

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), data[i]);
   } CARE_SEQUENTIAL_LOOP_END

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

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that the host_device_ptr constructor
///        behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
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

   care::KeyValueSorter<size_t, int, RAJA::seq_exec> sorter(length, data);

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), data[i]);
   } CARE_SEQUENTIAL_LOOP_END

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

#if defined(CARE_GPUCC)

/////////////////////////////////////////////////////////////////////////
///
/// @brief GPU test case that checks that the size constructor and
///        manually filling in the keys and values behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
GPU_TEST(KeyValueSorter, SizeConstructor)
{
   int length = 5;
   care::array<int, 5> data{{4, 1, 2, 0, 3}};
   care::KeyValueSorter<size_t, int, RAJAExec> sorter(length);

   CARE_STREAM_LOOP(i, 0, length) {
      sorter.setKey(i, i);
      sorter.setValue(i, data[i]);
   } CARE_STREAM_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), data[i]);
   } CARE_SEQUENTIAL_LOOP_END

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

/////////////////////////////////////////////////////////////////////////
///
/// @brief GPU test case that checks that the c-style array constructor
///        behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
GPU_TEST(KeyValueSorter, RawArrayConstructor)
{
   int length = 5;
   care::array<int, 5> data{{4, 1, 2, 0, 3}};
   care::KeyValueSorter<size_t, int, RAJAExec> sorter(length, data.data());

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), data[i]);
   } CARE_SEQUENTIAL_LOOP_END

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

/////////////////////////////////////////////////////////////////////////
///
/// @brief GPU test case that checks that the host_device_ptr constructor
///        behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
GPU_TEST(KeyValueSorter, host_device_ptr_Constructor)
{
   int length = 5;
   care::host_device_ptr<int> data(length);

   CARE_GPU_KERNEL {
      data[0] = 4;
      data[1] = 1;
      data[2] = 2;
      data[3] = 0;
      data[4] = 3;
   } CARE_GPU_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJAExec> sorter(length, data);

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), data[i]);
   } CARE_SEQUENTIAL_LOOP_END

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

#endif // CARE_GPUCC

