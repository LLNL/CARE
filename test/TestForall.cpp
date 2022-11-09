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
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/detail/test_utils.h"


#if defined(CARE_GPUCC)
GPU_TEST(forall, Initialization) {
   printf("Initializing\n");
   init_care_for_testing();
   printf("Initialized... Testing care::forall\n");
}
#endif

CPU_TEST(forall, static_policy)
{
   const int length = 10;
   care::host_device_ptr<int> temp(length, "temp");

   CARE_LOOP(care::sequential{}, i, 0, length) {
      temp[i] = i;
   } CARE_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(temp[i], i);
   } CARE_SEQUENTIAL_LOOP_END

   temp.free();
}

CPU_TEST(forall, dynamic_policy)
{
   const int length = 10;
   care::host_device_ptr<int> temp(length, "temp");

   CARE_LOOP(care::Policy::sequential, i, 0, length) {
      temp[i] = i;
   } CARE_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(temp[i], i);
   } CARE_SEQUENTIAL_LOOP_END

   temp.free();
}

#if defined(CARE_ENABLE_RACE_DETECTION)
CPU_TEST(forall, race_condition_detection)
{
   printf("in cpu_test race_condition_detection\n");
   const int length = 10;
   care::host_device_ptr<int, care::RaceConditionAccessor> temp(length, "temp");

   CARE_LOOP(care::parallel{}, i, 0, length) {
      temp[i] = i;
   } CARE_LOOP_END
   
   // expect race condition detection output here 
   CARE_PARALLEL_LOOP(i, 0, length-1) {
      temp[i+1] = temp[i];
   } CARE_PARALLEL_LOOP_END
   
   CARE_LOOP(care::parallel{}, i, 0, length) {
      temp[i] = i;
   } CARE_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(temp[i], i);
   } CARE_SEQUENTIAL_LOOP_END

   temp.free();
}
#endif

#if defined(CARE_GPUCC)

GPU_TEST(forall, static_policy)
{
   const int length = 10;
   care::host_device_ptr<int> temp(length, "temp");

   CARE_LOOP(care::gpu{}, i, 0, length) {
      temp[i] = i;
   } CARE_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(temp[i], i);
   } CARE_SEQUENTIAL_LOOP_END

   temp.free();
}

GPU_TEST(forall, dynamic_policy)
{
   const int length = 10;
   care::host_device_ptr<int> temp(length, "temp");

   CARE_LOOP(care::Policy::gpu, i, 0, length) {
      temp[i] = i;
   } CARE_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(temp[i], i);
   } CARE_SEQUENTIAL_LOOP_END

   temp.free();
}

#endif // CARE_GPUCC

