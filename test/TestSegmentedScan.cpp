//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/segmented_scan.h"
#include "care/detail/test_utils.h"

#include "gtest/gtest.h"

struct Multiply {
   CARE_HOST_DEVICE int operator()(int lhs, int rhs) const
   {
      return lhs * rhs;
   }
};

// Verify that the two-argument overload uses zero as the initial value and
// addition as the operation independently within each segment.
TEST(segmented_exclusive_scan, defaults_to_zero_and_addition)
{
   care::host_device_ptr<int> values(5);
   care::host_device_ptr<int> offsets(3);

   const int input[] = {
      3, 4, 5,
      6, 7
   };
   const int segmentOffsets[] = {0, 3, 5};
   const int expected[] = {
      0, 3, 7,
      0, 6
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      values[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_exclusive_scan(values, offsets);

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      EXPECT_EQ(values[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   values.free();
}

// Verify that a nonzero initial value is applied independently to each
// segment and that empty segments do not affect adjacent segments.
TEST(segmented_exclusive_scan, segment_local_empty_and_nonzero_initial_value)
{
   care::host_device_ptr<int> values(8);
   care::host_device_ptr<int> offsets(5);

   const int input[] = {
      5, 1, 4,
      // empty segment
      9, 3, 8,
      7, 2
   };
   const int segmentOffsets[] = {0, 3, 3, 6, 8};
   const int expected[] = {
      10, 15, 16,
      // empty segment
      10, 19, 22,
      10, 17
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      values[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_exclusive_scan(values, offsets, 10);

   CARE_SEQUENTIAL_LOOP(i, 0, 8) {
      EXPECT_EQ(values[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   values.free();
}

// Verify that the four-argument overload applies a custom binary operation
// and resets its initial value at each segment boundary.
TEST(segmented_exclusive_scan, generic_binary_operation)
{
   care::host_device_ptr<int> values(5);
   care::host_device_ptr<int> offsets(3);

   const int input[] = {
      3, 4, 5,
      6, 7
   };
   const int segmentOffsets[] = {0, 3, 5};
   const int expected[] = {
      2, 6, 24,
      2, 12
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      values[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_exclusive_scan(values, offsets, 2, Multiply {});

   CARE_SEQUENTIAL_LOOP(i, 0, 5) {
      EXPECT_EQ(values[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   values.free();
}

// Verify that scanning an empty value array is a no-op.
TEST(segmented_exclusive_scan, empty_input)
{
   care::host_device_ptr<int> values;
   care::host_device_ptr<int> offsets(1);

   CARE_SEQUENTIAL_LOOP(i, 0, 1) {
      offsets[i] = 0;
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_exclusive_scan(values, offsets, 7);
   offsets.free();

   EXPECT_EQ(values.size(), 0);
   EXPECT_EQ(values.data(), nullptr);
}

// Verify that scanning a slice updates its backing storage without replacing
// the slice or modifying values outside it.
TEST(segmented_exclusive_scan, preserves_slice)
{
   care::host_device_ptr<int> storage(6);
   care::host_device_ptr<int> values = storage.slice(1, 4);
   care::host_device_ptr<int> offsets(3);

   const int input[] = {
      -1, // before slice
      2, 3,
      4, 5,
      -2 // after slice
   };
   const int segmentOffsets[] = {0, 2, 4};
   const int expected[] = {
      -1, // before slice
      1, 3,
      1, 5,
      -2 // after slice
   };

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      storage[i] = input[i];
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, 3) {
      offsets[i] = segmentOffsets[i];
   } CARE_SEQUENTIAL_LOOP_END

   care::segmented_exclusive_scan(values, offsets, 1);

   CARE_SEQUENTIAL_LOOP(i, 0, 6) {
      EXPECT_EQ(storage[i], expected[i]);
   } CARE_SEQUENTIAL_LOOP_END

   offsets.free();
   storage.free();
}

int main(int argc, char** argv)
{
   testing::InitGoogleTest(&argc, argv);

#ifdef CARE_GPUCC
   init_care_for_testing();
#endif

   return RUN_ALL_TESTS();
}
