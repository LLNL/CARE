//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

#include "gtest/gtest.h"

#include "care/DefaultMacros.h"
#include "care/host_device_map.h"
#include "care/host_device_ptr.h"
#include "care/detail/test_utils.h"

#include <vector>

#if defined(CARE_GPUCC)
GPU_TEST(HostDeviceMap, InsertSortLookup)
{
   init_care_for_testing();

   constexpr int num_entries = 4;
   constexpr int missing_key = 77;
   constexpr int miss_signal = -999;
   care::host_device_map<int, int, RAJAExec> map(num_entries, miss_signal);

   CARE_STREAM_LOOP(i, 0, num_entries) {
      map.emplace((num_entries - 1) - i, 100 + i);
   } CARE_STREAM_LOOP_END

   map.sort();
   EXPECT_EQ(map.size(), num_entries);

   care::host_device_ptr<int> values(num_entries + 1, "map_values");

   CARE_STREAM_LOOP(i, 0, num_entries) {
      values[i] = map.at((num_entries - 1) - i);
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(i, num_entries, num_entries + 1) {
      values[i] = map.at(missing_key);
   } CARE_STREAM_LOOP_END

   EXPECT_EQ(values.pick(0), 100);
   EXPECT_EQ(values.pick(1), 101);
   EXPECT_EQ(values.pick(2), 102);
   EXPECT_EQ(values.pick(3), 103);
   EXPECT_EQ(values.pick(4), miss_signal);

   map.free();
}

GPU_TEST(HostDeviceMap, VectorDefaultConstruction)
{
   init_care_for_testing();

   constexpr int num_maps = 2;
   constexpr int num_entries = 2;
   constexpr int miss_signal = -1;
   std::vector<care::host_device_map<int, int, RAJAExec>> maps(num_maps);

   maps[0] = care::host_device_map<int, int, RAJAExec>(num_entries, miss_signal);
   maps[1] = care::host_device_map<int, int, RAJAExec>(num_entries, miss_signal);

   auto map0_insert = maps[0];
   auto map1_insert = maps[1];

   CARE_STREAM_LOOP(i, 0, num_entries) {
      map0_insert.emplace(i, 10 + i);
      map1_insert.emplace(10 + i, 20 + i);
   } CARE_STREAM_LOOP_END

   maps[0].sort();
   maps[1].sort();

   const auto map0 = maps[0];
   const auto map1 = maps[1];
   care::host_device_ptr<int> values(2 * num_entries, "vector_map_values");

   CARE_STREAM_LOOP(i, 0, num_entries) {
      values[i] = map0.at(i);
      values[num_entries + i] = map1.at(10 + i);
   } CARE_STREAM_LOOP_END

   EXPECT_EQ(values.pick(0), 10);
   EXPECT_EQ(values.pick(1), 11);
   EXPECT_EQ(values.pick(2), 20);
   EXPECT_EQ(values.pick(3), 21);

   maps[0].free();
   maps[1].free();
}
#endif // CARE_GPUCC
