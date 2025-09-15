//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_TEST_UTILS_H
#define CARE_TEST_UTILS_H

#include "care/config.h"
#include "care/host_device_ptr.h"
#include "care/LoopFuser.h"
#include "care/Setup.h"

/* CUDA profiling macros */
#ifdef __CUDACC__
#include "nvtx3/nvToolsExt.h"
#ifdef CARE_TEST_PUSH_VERBOSE
#define PUSH_PRINT(NAME) printf("%s\n",name);
#else
#define PUSH_PRINT(NAME)
#endif

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);
static unsigned int currentColor = 0;
#define PUSH_RANGE(name) { \
      PUSH_PRINT(name); \
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
#define POP_RANGE cudaDeviceSynchronize(); nvtxRangePop();
#else
#define PUSH_RANGE(name)
#define POP_RANGE
#endif

/////////////////////////////////////////////////////////////////////////
///
/// @brief Macro that allows extended __host__ __device__ lambdas (i.e.
///        CARE_STREAM_LOOP) to be used in google tests. Essentially,
///        extended __host__ __device__ lambdas cannot be used in
///        private or protected members, and the TEST macro creates a
///        protected member function. We get around this by creating a
///        function that the TEST macro then calls.
///
/// @note  Adapted from CHAI
///
/////////////////////////////////////////////////////////////////////////
#define str(X) #X
#define GPU_TEST(X, Y) static void cuda_test_ ## X_ ## Y(); \
   TEST(X, gpu_test_##Y) { PUSH_RANGE(str(Y)); cuda_test_ ## X_ ## Y(); POP_RANGE ;} \
   static void cuda_test_ ## X_ ## Y()


// Adapted from CHAI
#define CPU_TEST(X, Y) \
   static void cpu_test_##X##Y(); \
   TEST(X, cpu_test_##Y) { cpu_test_##X##Y(); } \
   static void cpu_test_##X##Y()

// ptr types 
using int_ptr = care::host_device_ptr<int>;

// memory pool initialization

void init_care_for_testing() {
#ifdef CARE_GPUCC
   // initializing and allocate memory pools
   care::initialize_pool("PINNED","PINNED_POOL",chai::PINNED,128*1024*1024,128*1024*1024,true);
   care::initialize_pool("DEVICE","DEVICE_POOL",chai::GPU,1024*1024*1024,1024*1024*1024,true);
   int_ptr trigger_device_allocation(1,"trigger_device");
   int_ptr trigger_pinned_allocation = chai::ManagedArray<int>(1, chai::PINNED);
   trigger_device_allocation.free();
   trigger_pinned_allocation.free();
#endif

   // initialize loop fuser
#if CARE_ENABLE_LOOP_FUSER
   LOOPFUSER(CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT)::getInstance();
#endif

   care::syncIfNeeded();
}


#if defined(CARE_GPUCC) && GTEST_HAS_DEATH_TEST
// This asserts a crash on the GPU, but does not mark gtest as passing.
#define GPU_FAIL(code) ASSERT_DEATH(code, "")
#else
#define GPU_FAIL(code) code
#endif


#endif
