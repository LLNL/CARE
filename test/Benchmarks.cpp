//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

// always have DEBUG on to force the packer to be on for CPU builds.
#ifndef CARE_DEBUG
#define CARE_DEBUG
#endif
#include "gtest/gtest.h"

#include "care/Setup.h"
#include "care/host_device_ptr.h"
#include "care/LoopFuser.h"
#include "care/detail/test_utils.h"

#if CARE_ENABLE_LOOP_FUSER

using namespace care;

GPU_TEST(TestFuser, Initialization) {
   printf("Initializing\n");
   init_care_for_testing();
   printf("Initialized... Benchmarking Loop Fusion\n");
}

GPU_TEST(TestFuser, OneMillionSmallKernels) {
   int numLoops = 1000000;
   int loopLength = 32;
   int_ptr a(loopLength,"a");
   int_ptr b(loopLength,"b");
   
   for (int i = 0; i < numLoops; ++i) {
      CARE_STREAM_LOOP(j,0,loopLength) {
         a[j] = i;
         b[j] = i/2;
      } CARE_STREAM_LOOP_END
   }

   a.free();
   b.free();
   care::syncIfNeeded();
}


GPU_TEST(TestFuser, OneMillionSmallFusedKernels) {
   int numLoops = 1000000;
   int loopLength = 32;
   int_ptr a(loopLength,"a");
   int_ptr b(loopLength,"b");
   LOOPFUSER(CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT)::getInstance()->setVerbose(true);
   
   FUSIBLE_LOOPS_START 
   for (int i = 0; i < numLoops; ++i) {
      FUSIBLE_LOOP_STREAM(j,0,loopLength) {
         a[j] = i;
         b[j] = i/2;
      } FUSIBLE_LOOP_STREAM_END
   }
   FUSIBLE_LOOPS_STOP

   a.free();
   b.free();
   care::syncIfNeeded();
}


GPU_TEST(TestFuser, OneThousandLargeKernels) {
   int numLoops = 1000;
   int loopLength = 7000000;
   int_ptr a(loopLength,"a");
   int_ptr b(loopLength,"b");
   
   for (int i = 0; i < numLoops; ++i) {
      CARE_STREAM_LOOP(j,0,loopLength) {
         a[j] = i;
         b[j] = i/2;
      } CARE_STREAM_LOOP_END
   }

   a.free();
   b.free();
   care::syncIfNeeded();
}

GPU_TEST(TestFuser, OneThousandLargeFusedKernels) {
   int numLoops = 1000;
   int loopLength = 7000000;
   int_ptr a(loopLength,"a");
   int_ptr b(loopLength,"b");
   
   FUSIBLE_LOOPS_START 
   for (int i = 0; i < numLoops; ++i) {
      FUSIBLE_LOOP_STREAM(j,0,loopLength) {
         a[j] = i;
         b[j] = i/2;
      } FUSIBLE_LOOP_STREAM_END
   }
   FUSIBLE_LOOPS_STOP

   a.free();
   b.free();
   care::syncIfNeeded();
   care::syncIfNeeded();
}



static int medium_length = 32000;

GPU_TEST(TestFuser, TenThousandMediumKernels) {
   int numLoops = 10000;
   int loopLength = medium_length;
   int_ptr a(loopLength,"a");
   int_ptr b(loopLength,"b");
   
   for (int i = 0; i < numLoops; ++i) {
      CARE_STREAM_LOOP(j,0,loopLength) {
         a[j] = i;
         b[j] = i/2;
      } CARE_STREAM_LOOP_END
   }

   a.free();
   b.free();
   care::syncIfNeeded();
}



GPU_TEST(TestFuser, TenThousandMediumFusedKernels) {
   int numLoops = 10000;
   int loopLength = medium_length;
   int_ptr a(loopLength,"a");
   int_ptr b(loopLength,"b");
   
   FUSIBLE_LOOPS_START 
   for (int i = 0; i < numLoops; ++i) {
      FUSIBLE_LOOP_STREAM(j,0,loopLength) {
         a[j] = i;
         b[j] = i/2;
      } FUSIBLE_LOOP_STREAM_END
   }
   FUSIBLE_LOOPS_STOP

   a.free();
   b.free();
   care::syncIfNeeded();
}

GPU_TEST(TestFuser, OneThousandFusedLaunches) {
   int numLoops = 1000;
   int loopLength = medium_length;
   int_ptr a(loopLength,"a");
   int_ptr b(loopLength,"b");
   
   for (int i = 0; i < numLoops; ++i) {
      FUSIBLE_LOOPS_START 
      FUSIBLE_LOOP_STREAM(j,0,loopLength) {
         a[j] = i;
         b[j] = i/2;
      } FUSIBLE_LOOP_STREAM_END
      FUSIBLE_LOOPS_STOP
   }

   a.free();
   b.free();
   care::syncIfNeeded();
}

#endif
