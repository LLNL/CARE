//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

// Makes CARE_REDUCE_LOOP run on the device
#if defined(CARE_GPUCC)
#define GPU_ACTIVE
#endif

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"

// Adapted from CHAI
#define CPU_TEST(X, Y) \
   static void cpu_test_##X##Y(); \
   TEST(X, cpu_test_##Y) { cpu_test_##X##Y(); } \
   static void cpu_test_##X##Y()

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

#if defined(CARE_GPUCC)

// Adapted from CHAI
#define GPU_TEST(X, Y) \
   static void gpu_test_##X##Y(); \
   TEST(X, gpu_test_##Y) { gpu_test_##X##Y(); } \
   static void gpu_test_##X##Y()

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

