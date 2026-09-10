//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/unique.h"
#include "care/sort.h"
#include "care/detail/test_utils.h"

#include "gtest/gtest.h"

TEST(unique, compacts_sorted_input)
{
   care::host_device_ptr<int> keys(8);
   const int input[] = {1, 1, 2, 3, 3, 3, 4, 4};
   const int expected[] = {1, 2, 3, 4};

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   const size_t numUnique = care::unique(keys);

   EXPECT_EQ(numUnique, 4);
   EXPECT_EQ(keys.size(), 8);
   CARE_SEQUENTIAL_LOOP(i, 0, numUnique) {
      EXPECT_EQ(keys[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   keys.free();
}

GPU_TEST(unique, nonsegmented_custom_equivalence_predicate)
{
   care::host_device_ptr<int> keys(6);
   const int input[] = {21, 22, 22, 31, 33, 33};
   const int expected[] = {21, 31};

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   const size_t numUnique = care::unique(
      keys,
      [] CARE_HOST_DEVICE (int left, int right) {
         return left / 10 == right / 10;
      });

   EXPECT_EQ(numUnique, 2);
   CARE_SEQUENTIAL_LOOP(i, 0, numUnique) {
      EXPECT_EQ(keys[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   keys.free();
}

TEST(segmented_unique, segment_local_and_empty)
{
   care::host_device_ptr<int> keys(10);
   care::host_device_ptr<int> offsets(5);

   const int input[] = {
      4, 4, 5,
      // empty segment
      2, 2, 3, 3,
      1, 2, 3
   };

   const int segmentOffsets[] = {0, 3, 3, 7, 10};

   const int expectedKeys[] = {
      4, 5,
      // empty segment
      2, 3,
      1, 2, 3
   };

   const int expectedOffsets[] = {0, 2, 2, 4, 7};

   CARE_SEQUENTIAL_LOOP(i, 0, 10) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_unique(keys, offsets);

   ASSERT_EQ(keys.size(), 10);
   ASSERT_EQ(offsets.size(), 5);

   CARE_SEQUENTIAL_LOOP(i, 0, 7) {
      EXPECT_EQ(keys[i], expectedKeys[i]);
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      EXPECT_EQ(offsets[i], expectedOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   keys.free();
}

TEST(segmented_unique, unique_after_sort)
{
   care::host_device_ptr<int> keys(9);
   care::host_device_ptr<int> offsets(4);

   const int input[] = {
      4, 1, 4,
      8, 7, 8, 7,
      5, 5
   };

   const int segmentOffsets[] = {0, 3, 7, 9};

   const int expectedKeys[] = {
      1, 4,
      7, 8,
      5
   };

   const int expectedOffsets[] = {0, 2, 4, 5};

   CARE_SEQUENTIAL_LOOP(i, 0, 9) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_sort(keys, offsets);
   care::segmented_unique(keys, offsets);

   ASSERT_EQ(keys.size(), 9);
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

GPU_TEST(segmented_unique, custom_equivalence_predicate)
{
   care::host_device_ptr<int> keys(6);
   care::host_device_ptr<int> offsets(3);

   const int input[] = {
      21, 22, 22,
      11, 13, 13
   };
   const int segmentOffsets[] = {0, 3, 6};

   const int expectedKeys[] = {
      21,
      11
   };

   const int expectedOffsets[] = {0, 1, 2};

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      keys[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_unique(
      keys,
      offsets,
      [] CARE_HOST_DEVICE (int left, int right) {
         return left / 10 == right / 10;
      });

   ASSERT_EQ(keys.size(), 6);
   ASSERT_EQ(offsets.size(), 3);

   CARE_SEQUENTIAL_LOOP(i, 0, 2) {
      EXPECT_EQ(keys[i], expectedKeys[i]);
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      EXPECT_EQ(offsets[i], expectedOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   keys.free();
}

TEST(segmented_unique, empty_input_and_segments)
{
   care::host_device_ptr<int> keys;
   care::host_device_ptr<int> offsets(4);

   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      offsets[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_unique(keys, offsets);

   EXPECT_EQ(keys.size(), 0);
   ASSERT_EQ(offsets.size(), 4);

   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      EXPECT_EQ(offsets[i], 0);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
}

TEST(segmented_unique, compacts_input_slices)
{
   care::host_device_ptr<int> keyStorage(8);
   care::host_device_ptr<int> offsetStorage(5);
   care::host_device_ptr<int> keys = keyStorage.slice(1, 6);
   care::host_device_ptr<int> offsets = offsetStorage.slice(1, 3);

   const int input[] = {
      -1,
      1, 1, 2,
      0, 0, 1,
      -2
   };
   const int segmentOffsets[] = {-1, 0, 3, 6, -2};

   const int expectedKeys[] = {
      1, 2,
      0, 1
   };

   const int expectedOffsets[] = {0, 2, 4};

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      keyStorage[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsetStorage[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_unique(keys, offsets);

   ASSERT_EQ(keys.size(), 6);
   ASSERT_EQ(offsets.size(), 3);

   CARE_SEQUENTIAL_LOOP(i, 0, 4) {
      EXPECT_EQ(keys[i], expectedKeys[i]);
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      EXPECT_EQ(offsets[i], expectedOffsets[i]);
   } CARE_SEQUENTIAL_LOOP_END

   EXPECT_EQ(keyStorage[0], -1);
   EXPECT_EQ(keyStorage[7], -2);
   EXPECT_EQ(offsetStorage[0], -1);
   EXPECT_EQ(offsetStorage[4], -2);

   offsetStorage.free();
   keyStorage.free();
}

int main(int argc, char** argv)
{
   testing::InitGoogleTest(&argc, argv);

#ifdef CARE_GPUCC
   init_care_for_testing();
#endif

   return RUN_ALL_TESTS();
}
