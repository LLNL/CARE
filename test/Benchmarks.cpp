//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

#define GPU_ACTIVE
// always have DEBUG on to force the packer to be on for CPU builds.
#ifndef CARE_DEBUG
#define CARE_DEBUG
#endif
#include "gtest/gtest.h"

#include "care/Setup.h"
#include "care/host_device_ptr.h"
#include "care/LoopFuser.h"

#if CARE_ENABLE_LOOP_FUSER
/* CUDA profiling macros */
#ifdef __CUDACC__
#include "nvToolsExt.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);
static unsigned int currentColor = 0;
#define PUSH_RANGE(name) { \
      int color_id = currentColor++; \
      color_id = color_id%num_colors; \
      nvtxEventAttributes_t eventAttrib = { 0 }; \
      eventAttrib.version = NVTX_VERSION; \
      eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
      eventAttrib.colorType = NVTX_COLOR_ARGB; \
      eventAttrib.color = colors[color_id]; \
      eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
      eventAttrib.message.ascii = name; \
      nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name)
#define POP_RANGE
#endif

#define str(X) #X
// This makes it so we can use device lambdas from within a CUDA_TEST
#define CUDA_TEST(X, Y) static void cuda_test_ ## X_ ## Y(); \
   TEST(X, Y) { PUSH_RANGE(str(Y)); cuda_test_ ## X_ ## Y(); POP_RANGE ;} \
   static void cuda_test_ ## X_ ## Y()

using namespace care;
using int_ptr = host_device_ptr<int>;

CUDA_TEST(TestFuser, Initialization) {
   printf("Initializing\n");
   // initializing and allocate memory pools
   care::initialize_pool("PINNED","PINNED_POOL",chai::PINNED,128*1024*1024,128*1024*1024,true);
   care::initialize_pool("DEVICE","DEVICE_POOL",chai::GPU,128*1024*1024,128*1024*1024,true);
   int_ptr trigger_device_allocation(1,"trigger_device");
   int_ptr trigger_pinned_allocation = chai::ManagedArray<int>(1, chai::PINNED);
   trigger_device_allocation.free();
   trigger_pinned_allocation.free();
   // initialize loop fuser
   LoopFuser<CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT>::getInstance();

   care::syncIfNeeded();
   printf("Initialized... Benchmarking Loop Fusion\n");
}


CUDA_TEST(TestFuser, OneMillionSmallKernels) {
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


CUDA_TEST(TestFuser, OneMillionSmallFusedKernels) {
   int numLoops = 1000000;
   int loopLength = 32;
   int_ptr a(loopLength,"a");
   int_ptr b(loopLength,"b");
   LoopFuser<CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT>::getInstance()->setVerbose(true);
   
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


CUDA_TEST(TestFuser, OneThousandLargeKernels) {
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

CUDA_TEST(TestFuser, OneThousandLargeFusedKernels) {
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

CUDA_TEST(TestFuser, TenThousandMediumKernels) {
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



CUDA_TEST(TestFuser, TenThousandMediumFusedKernels) {
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

CUDA_TEST(TestFuser, OneThousandFusedLaunches) {
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
