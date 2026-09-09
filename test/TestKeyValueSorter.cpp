//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

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

   care::KeyValueSorter<size_t, int, RAJA::seq_exec> sorter(length, care::host_device_ptr<const int>(data));

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

   care::KeyValueSorter<size_t, int, RAJAExec> sorter(length, care::host_device_ptr<const int>(data));

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

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks the new constructor that takes ownership
///        of keys and values arrays.
///
/////////////////////////////////////////////////////////////////////////
TEST(KeyValueSorter, OwnershipConstructor)
{
   int length = 5;
   care::host_device_ptr<size_t> keys(length, "keys");
   care::host_device_ptr<int> values(length, "values");

   CARE_HOST_KERNEL {
      keys[0] = 0;
      keys[1] = 1;
      keys[2] = 2;
      keys[3] = 3;
      keys[4] = 4;

      values[0] = 4;
      values[1] = 1;
      values[2] = 2;
      values[3] = 0;
      values[4] = 3;
   } CARE_HOST_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJA::seq_exec> sorter(length, std::move(keys), std::move(values));

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), values[i]);
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
/// @brief Test case that checks shallow copies share key and value storage
///        for the sequential KeyValueSorter specialization.
///
/////////////////////////////////////////////////////////////////////////
TEST(KeyValueSorter, CopySharesStorage)
{
   int length = 5;
   care::host_device_ptr<size_t> keys(length, "keys");
   care::host_device_ptr<int> values(length, "values");

   CARE_HOST_KERNEL {
      keys[0] = 3;
      keys[1] = 1;
      keys[2] = 3;
      keys[3] = 2;
      keys[4] = 1;

      values[0] = 7;
      values[1] = 5;
      values[2] = 6;
      values[3] = 3;
      values[4] = 4;
   } CARE_HOST_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJA::seq_exec> owner(length, std::move(keys), std::move(values));
   care::KeyValueSorter<size_t, int, RAJA::seq_exec> alias(owner);

   const size_t* sharedKeys = alias.keys().cdata();
   const int* sharedValues = alias.values().cdata();

   EXPECT_EQ(owner.keys().cdata(), sharedKeys);
   EXPECT_EQ(owner.values().cdata(), sharedValues);

   alias.sortByKeyThenValue();

   const size_t* sortedKeys = owner.keys().cdata();
   const int* sortedValues = owner.values().cdata();

   EXPECT_EQ(alias.keys().cdata(), sortedKeys);
   EXPECT_EQ(alias.values().cdata(), sortedValues);

   CARE_HOST_KERNEL {
      EXPECT_EQ(owner.key(0), 1);
      EXPECT_EQ(owner.key(1), 1);
      EXPECT_EQ(owner.key(2), 2);
      EXPECT_EQ(owner.key(3), 3);
      EXPECT_EQ(owner.key(4), 3);

      EXPECT_EQ(owner.value(0), 4);
      EXPECT_EQ(owner.value(1), 5);
      EXPECT_EQ(owner.value(2), 3);
      EXPECT_EQ(owner.value(3), 6);
      EXPECT_EQ(owner.value(4), 7);
   } CARE_HOST_KERNEL_END
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks a shallow copy remains valid after the copy
///        is move-assigned and then mutates the shared key-value storage.
///
/////////////////////////////////////////////////////////////////////////
TEST(KeyValueSorter, CopyMoveSharesStorage)
{
   int length = 4;
   care::host_device_ptr<size_t> keys(length, "keys");
   care::host_device_ptr<int> values(length, "values");

   CARE_HOST_KERNEL {
      keys[0] = 2;
      keys[1] = 1;
      keys[2] = 2;
      keys[3] = 1;

      values[0] = 4;
      values[1] = 3;
      values[2] = 2;
      values[3] = 1;
   } CARE_HOST_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJA::seq_exec> owner(length, std::move(keys), std::move(values));

   const size_t* cachedKeys = owner.keys().cdata();
   const int* cachedValues = owner.values().cdata();

   care::KeyValueSorter<size_t, int, RAJA::seq_exec> alias(owner);
   care::KeyValueSorter<size_t, int, RAJA::seq_exec> moved;

   moved = std::move(alias);

   EXPECT_EQ(moved.keys().cdata(), cachedKeys);
   EXPECT_EQ(moved.values().cdata(), cachedValues);

   moved.sortByKeyThenValue();

   CARE_HOST_KERNEL {
      EXPECT_EQ(owner.key(0), 1);
      EXPECT_EQ(owner.key(1), 1);
      EXPECT_EQ(owner.key(2), 2);
      EXPECT_EQ(owner.key(3), 2);

      EXPECT_EQ(owner.value(0), 1);
      EXPECT_EQ(owner.value(1), 3);
      EXPECT_EQ(owner.value(2), 2);
      EXPECT_EQ(owner.value(3), 4);
   } CARE_HOST_KERNEL_END
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks the sortByKeyThenValue method.
///
/////////////////////////////////////////////////////////////////////////
TEST(KeyValueSorter, SortByKeyThenValue)
{
   int length = 8;
   care::host_device_ptr<size_t> keys(length, "keys");
   care::host_device_ptr<int> values(length, "values");

   CARE_HOST_KERNEL {
      // Create data with duplicate keys
      keys[0] = 3;
      keys[1] = 1;
      keys[2] = 3;
      keys[3] = 2;
      keys[4] = 1;
      keys[5] = 2;
      keys[6] = 3;
      keys[7] = 1;

      // Values are in reverse order within each key group
      values[0] = 7;
      values[1] = 5;
      values[2] = 6;
      values[3] = 3;
      values[4] = 4;
      values[5] = 2;
      values[6] = 8;
      values[7] = 1;
   } CARE_HOST_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJA::seq_exec> sorter(length, std::move(keys), std::move(values));

   // Sort by key then by value
   sorter.sortByKeyThenValue();

   // Check that keys are sorted
   CARE_HOST_KERNEL {
      // Keys should be in ascending order
      EXPECT_EQ(sorter.key(0), 1);
      EXPECT_EQ(sorter.key(1), 1);
      EXPECT_EQ(sorter.key(2), 1);
      EXPECT_EQ(sorter.key(3), 2);
      EXPECT_EQ(sorter.key(4), 2);
      EXPECT_EQ(sorter.key(5), 3);
      EXPECT_EQ(sorter.key(6), 3);
      EXPECT_EQ(sorter.key(7), 3);

      // Values should be in ascending order within each key group
      EXPECT_EQ(sorter.value(0), 1);
      EXPECT_EQ(sorter.value(1), 4);
      EXPECT_EQ(sorter.value(2), 5);
      EXPECT_EQ(sorter.value(3), 2);
      EXPECT_EQ(sorter.value(4), 3);
      EXPECT_EQ(sorter.value(5), 6);
      EXPECT_EQ(sorter.value(6), 7);
      EXPECT_EQ(sorter.value(7), 8);
   } CARE_HOST_KERNEL_END

   int boundaryLength = 3;
   care::host_device_ptr<size_t> boundaryKeys(boundaryLength, "boundaryKeys");
   care::host_device_ptr<int> boundaryValues(boundaryLength, "boundaryValues");

   CARE_HOST_KERNEL {
      boundaryKeys[0] = 1;
      boundaryKeys[1] = 1;
      boundaryKeys[2] = 2;

      boundaryValues[0] = 30;
      boundaryValues[1] = 20;
      boundaryValues[2] = 10;
   } CARE_HOST_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJA::seq_exec> boundarySorter(boundaryLength,
                                                                    std::move(boundaryKeys),
                                                                    std::move(boundaryValues));

   boundarySorter.sortByKeyThenValue();

   CARE_HOST_KERNEL {
      EXPECT_EQ(boundarySorter.key(0), 1);
      EXPECT_EQ(boundarySorter.value(0), 20);

      EXPECT_EQ(boundarySorter.key(1), 1);
      EXPECT_EQ(boundarySorter.value(1), 30);

      EXPECT_EQ(boundarySorter.key(2), 2);
      EXPECT_EQ(boundarySorter.value(2), 10);
   } CARE_HOST_KERNEL_END
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks the eliminateDuplicatePairs method.
///
/////////////////////////////////////////////////////////////////////////
TEST(KeyValueSorter, EliminateDuplicatePairs)
{
   int length = 10;
   care::host_device_ptr<size_t> keys(length, "keys");
   care::host_device_ptr<int> values(length, "values");

   CARE_HOST_KERNEL {
      // Create data with duplicate key-value pairs
      keys[0] = 1;
      keys[1] = 2;
      keys[2] = 3;
      keys[3] = 1;  // Duplicate of pair at index 0
      keys[4] = 2;  // Duplicate of pair at index 1
      keys[5] = 4;
      keys[6] = 5;
      keys[7] = 3;  // Duplicate of pair at index 2
      keys[8] = 4;  // Duplicate of pair at index 5
      keys[9] = 6;

      values[0] = 10;
      values[1] = 20;
      values[2] = 30;
      values[3] = 10;  // Duplicate of pair at index 0
      values[4] = 20;  // Duplicate of pair at index 1
      values[5] = 40;
      values[6] = 50;
      values[7] = 30;  // Duplicate of pair at index 2
      values[8] = 40;  // Duplicate of pair at index 5
      values[9] = 60;
   } CARE_HOST_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJA::seq_exec> sorter(length, std::move(keys), std::move(values));

   // Eliminate duplicate pairs
   sorter.eliminateDuplicatePairs();

   // Check that duplicates were removed
   CARE_HOST_KERNEL {
      // Should have 6 unique pairs
      EXPECT_EQ(sorter.len(), 6);

      // Check the unique pairs
      EXPECT_EQ(sorter.key(0), 1);
      EXPECT_EQ(sorter.value(0), 10);

      EXPECT_EQ(sorter.key(1), 2);
      EXPECT_EQ(sorter.value(1), 20);

      EXPECT_EQ(sorter.key(2), 3);
      EXPECT_EQ(sorter.value(2), 30);

      EXPECT_EQ(sorter.key(3), 4);
      EXPECT_EQ(sorter.value(3), 40);

      EXPECT_EQ(sorter.key(4), 5);
      EXPECT_EQ(sorter.value(4), 50);

      EXPECT_EQ(sorter.key(5), 6);
      EXPECT_EQ(sorter.value(5), 60);
   } CARE_HOST_KERNEL_END
}

#if defined(CARE_GPUCC)

/////////////////////////////////////////////////////////////////////////
///
/// @brief GPU test case that checks the new constructor that takes ownership
///        of keys and values arrays.
///
/////////////////////////////////////////////////////////////////////////
GPU_TEST(KeyValueSorter, OwnershipConstructor)
{
   int length = 5;
   care::host_device_ptr<size_t> keys(length, "keys");
   care::host_device_ptr<int> values(length, "values");

   CARE_GPU_KERNEL {
      keys[0] = 0;
      keys[1] = 1;
      keys[2] = 2;
      keys[3] = 3;
      keys[4] = 4;

      values[0] = 4;
      values[1] = 1;
      values[2] = 2;
      values[3] = 0;
      values[4] = 3;
   } CARE_GPU_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJAExec> sorter(length, std::move(keys), std::move(values));

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(sorter.key(i), i);
      EXPECT_EQ(sorter.value(i), values[i]);
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
/// @brief GPU test case that checks the sortByKeyThenValue method.
///
/////////////////////////////////////////////////////////////////////////
GPU_TEST(KeyValueSorter, SortByKeyThenValue)
{
   int length = 8;
   care::host_device_ptr<size_t> keys(length, "keys");
   care::host_device_ptr<int> values(length, "values");

   CARE_GPU_KERNEL {
      // Create data with duplicate keys
      keys[0] = 3;
      keys[1] = 1;
      keys[2] = 3;
      keys[3] = 2;
      keys[4] = 1;
      keys[5] = 2;
      keys[6] = 3;
      keys[7] = 1;

      // Values are in reverse order within each key group
      values[0] = 7;
      values[1] = 5;
      values[2] = 6;
      values[3] = 3;
      values[4] = 4;
      values[5] = 2;
      values[6] = 8;
      values[7] = 1;
   } CARE_GPU_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJAExec> sorter(length, std::move(keys), std::move(values));

   // Sort by key then by value
   sorter.sortByKeyThenValue();

   // Check that keys are sorted
   CARE_HOST_KERNEL {
      // Keys should be in ascending order
      EXPECT_EQ(sorter.key(0), 1);
      EXPECT_EQ(sorter.key(1), 1);
      EXPECT_EQ(sorter.key(2), 1);
      EXPECT_EQ(sorter.key(3), 2);
      EXPECT_EQ(sorter.key(4), 2);
      EXPECT_EQ(sorter.key(5), 3);
      EXPECT_EQ(sorter.key(6), 3);
      EXPECT_EQ(sorter.key(7), 3);

      // Values should be in ascending order within each key group
      EXPECT_EQ(sorter.value(0), 1);
      EXPECT_EQ(sorter.value(1), 4);
      EXPECT_EQ(sorter.value(2), 5);
      EXPECT_EQ(sorter.value(3), 2);
      EXPECT_EQ(sorter.value(4), 3);
      EXPECT_EQ(sorter.value(5), 6);
      EXPECT_EQ(sorter.value(6), 7);
      EXPECT_EQ(sorter.value(7), 8);
   } CARE_HOST_KERNEL_END

   int boundaryLength = 3;
   care::host_device_ptr<size_t> boundaryKeys(boundaryLength, "boundaryKeys");
   care::host_device_ptr<int> boundaryValues(boundaryLength, "boundaryValues");

   CARE_GPU_KERNEL {
      boundaryKeys[0] = 1;
      boundaryKeys[1] = 1;
      boundaryKeys[2] = 2;

      boundaryValues[0] = 30;
      boundaryValues[1] = 20;
      boundaryValues[2] = 10;
   } CARE_GPU_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJAExec> boundarySorter(boundaryLength,
                                                              std::move(boundaryKeys),
                                                              std::move(boundaryValues));

   boundarySorter.sortByKeyThenValue();

   CARE_HOST_KERNEL {
      EXPECT_EQ(boundarySorter.key(0), 1);
      EXPECT_EQ(boundarySorter.value(0), 20);

      EXPECT_EQ(boundarySorter.key(1), 1);
      EXPECT_EQ(boundarySorter.value(1), 30);

      EXPECT_EQ(boundarySorter.key(2), 2);
      EXPECT_EQ(boundarySorter.value(2), 10);
   } CARE_HOST_KERNEL_END
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief GPU test case that checks the eliminateDuplicatePairs method.
///
/////////////////////////////////////////////////////////////////////////
GPU_TEST(KeyValueSorter, EliminateDuplicatePairs)
{
   int length = 10;
   care::host_device_ptr<size_t> keys(length, "keys");
   care::host_device_ptr<int> values(length, "values");

   CARE_GPU_KERNEL {
      // Create data with duplicate key-value pairs
      keys[0] = 1;
      keys[1] = 2;
      keys[2] = 3;
      keys[3] = 1;  // Duplicate of pair at index 0
      keys[4] = 2;  // Duplicate of pair at index 1
      keys[5] = 4;
      keys[6] = 5;
      keys[7] = 3;  // Duplicate of pair at index 2
      keys[8] = 4;  // Duplicate of pair at index 5
      keys[9] = 6;

      values[0] = 10;
      values[1] = 20;
      values[2] = 30;
      values[3] = 10;  // Duplicate of pair at index 0
      values[4] = 20;  // Duplicate of pair at index 1
      values[5] = 40;
      values[6] = 50;
      values[7] = 30;  // Duplicate of pair at index 2
      values[8] = 40;  // Duplicate of pair at index 5
      values[9] = 60;
   } CARE_GPU_KERNEL_END

   care::KeyValueSorter<size_t, int, RAJAExec> sorter(length, std::move(keys), std::move(values));

   // Eliminate duplicate pairs
   sorter.eliminateDuplicatePairs();

   // Check that duplicates were removed
   CARE_HOST_KERNEL {
      // Should have 6 unique pairs
      EXPECT_EQ(sorter.len(), 6);

      // Check the unique pairs
      EXPECT_EQ(sorter.key(0), 1);
      EXPECT_EQ(sorter.value(0), 10);

      EXPECT_EQ(sorter.key(1), 2);
      EXPECT_EQ(sorter.value(1), 20);

      EXPECT_EQ(sorter.key(2), 3);
      EXPECT_EQ(sorter.value(2), 30);

      EXPECT_EQ(sorter.key(3), 4);
      EXPECT_EQ(sorter.value(3), 40);

      EXPECT_EQ(sorter.key(4), 5);
      EXPECT_EQ(sorter.value(4), 50);

      EXPECT_EQ(sorter.key(5), 6);
      EXPECT_EQ(sorter.value(5), 60);
   } CARE_HOST_KERNEL_END
}

#endif
