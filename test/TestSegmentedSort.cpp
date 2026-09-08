//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/sort.h"
#include "care/detail/test_utils.h"

#include "gtest/gtest.h"

// Verify that each segment is sorted independently and empty segments are
// accepted without affecting adjacent segments.
TEST(segmented_sort, segment_local_and_empty)
{
   care::host_device_ptr<int> keys(8);
   care::host_device_ptr<int> offsets(5);

   const int input[] = {
      5, 1, 4,
      // empty segment
      9, 3, 8,
      7, 2
   };
   const int segmentOffsets[] = {0, 3, 3, 6, 8};
   const int expected[] = {
      1, 4, 5,
      // empty segment
      3, 8, 9,
      2, 7
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort(keys, offsets);

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      EXPECT_EQ(keys[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   keys.free();
}

// Verify that sorting an empty key array is a no-op.
TEST(segmented_sort, empty_input)
{
   care::host_device_ptr<int> keys;
   care::host_device_ptr<int> offsets(1);

   CARE_SEQUENTIAL_LOOP(i, 0, 1) {
      offsets[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort(keys, offsets);
   offsets.free();

   EXPECT_EQ(keys.size(), 0);
   EXPECT_EQ(keys.data(), nullptr);
}

// Verify that sorting a slice updates its backing storage without replacing
// the slice or modifying values outside it.
TEST(segmented_sort, preserves_slice)
{
   care::host_device_ptr<int> storage(6);
   care::host_device_ptr<int> keys = storage.slice(1, 4);
   care::host_device_ptr<int> offsets(3);

   const int input[] = {
      -1, // before slice
      4, 2,
      5, 3,
      -2 // after slice
   };
   const int segmentOffsets[] = {0, 2, 4};
   const int expected[] = {
      -1, // before slice
      2, 4,
      3, 5,
      -2 // after slice
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      storage[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort(keys, offsets);

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(storage[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   storage.free();
}

// Verify that sorting and uniquing compacts each segment independently,
// preserves empty segments, and leaves the input arrays unchanged.
TEST(segmented_sort_and_unique, out_of_place)
{
   care::host_device_ptr<int> keys(10);
   care::host_device_ptr<int> offsets(5);
   care::host_device_ptr<int> uniqueKeys(1);
   care::host_device_ptr<int> uniqueOffsets(1);

   const int input[] = {
      4, 1, 4,
      // empty segment
      8, 7, 8, 7,
      5, 5, 4
   };
   const int segmentOffsets[] = {0, 3, 3, 7, 10};
   const int expectedKeys[] = {1, 4, 7, 8, 4, 5};
   const int expectedOffsets[] = {0, 2, 2, 4, 6};

   CARE_SEQUENTIAL_LOOP(i, 0, 10) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort_and_unique(
      keys, offsets, uniqueKeys, uniqueOffsets);

   ASSERT_EQ(uniqueKeys.size(), 6);
   ASSERT_EQ(uniqueOffsets.size(), 5);
   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(uniqueKeys[i], expectedKeys[i]);
   } CARE_SEQUENTIAL_LOOP_END
   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      EXPECT_EQ(uniqueOffsets[i], expectedOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 10) {
      EXPECT_EQ(keys[i], input[i]);
   } CARE_SEQUENTIAL_LOOP_END
   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      EXPECT_EQ(offsets[i], segmentOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   uniqueOffsets.free();
   uniqueKeys.free();
   offsets.free();
   keys.free();
}

// Verify that the replacement overload updates both arrays and their sizes.
TEST(segmented_sort_and_unique, replaces_inputs)
{
   care::host_device_ptr<int> keys(9);
   care::host_device_ptr<int> offsets(4);

   const int input[] = {
      3, 2, 3,
      1, 1, 2, 1,
      2, 2
   };
   const int segmentOffsets[] = {0, 3, 7, 9};
   const int expectedKeys[] = {2, 3, 1, 2, 2};
   const int expectedOffsets[] = {0, 2, 4, 5};

   CARE_SEQUENTIAL_LOOP(i, 0, 9) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort_and_unique(keys, offsets);

   ASSERT_EQ(keys.size(), 5);
   ASSERT_EQ(offsets.size(), 4);
   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      EXPECT_EQ(keys[i], expectedKeys[i]);
   } CARE_SEQUENTIAL_LOOP_END
   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      EXPECT_EQ(offsets[i], expectedOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   keys.free();
}

// Verify that empty input and multiple empty segments produce empty output
// while retaining every segment boundary.
TEST(segmented_sort_and_unique, empty_input)
{
   care::host_device_ptr<int> keys;
   care::host_device_ptr<int> offsets(4);
   care::host_device_ptr<int> uniqueKeys;
   care::host_device_ptr<int> uniqueOffsets;

   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      offsets[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort_and_unique(
      keys, offsets, uniqueKeys, uniqueOffsets);

   EXPECT_EQ(uniqueKeys.size(), 0);
   ASSERT_EQ(uniqueOffsets.size(), 4);
   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      EXPECT_EQ(uniqueOffsets[i], 0);
   } CARE_SEQUENTIAL_LOOP_END

   uniqueOffsets.free();
   offsets.free();
}

int main(int argc, char** argv)
{
   testing::InitGoogleTest(&argc, argv);

#ifdef CARE_GPUCC
   init_care_for_testing();
#endif

   return RUN_ALL_TESTS();
}
