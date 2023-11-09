//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

#if defined(CARE_GPUCC)

// other library headers
#include "gtest/gtest.h"
#include "umpire/strategy/QuickPool.hpp"

// care headers
#include "care/CudaUmpireResource.h"
#include "care/detail/test_utils.h"

GPU_TEST(CudaUmpireResource, gpu_initialization) {
   init_care_for_testing();
}

GPU_TEST(CudaUmpireResource, DefaultConstructor)
{
   care::CudaUmpireResource resource;
}

GPU_TEST(CudaUmpireResource, AllocatorConstructor)
{
   auto& rm = umpire::ResourceManager::getInstance();

   // Device allocator
   auto deviceAllocator = rm.getAllocator("DEVICE_POOL"); // Initialized above
   auto customDeviceAllocator =
      rm.makeAllocator<umpire::strategy::QuickPool>("CUSTOM_DEVICE_POOL",
                                                    deviceAllocator,
                                                    64*1024*1024,
                                                    16*1024*1024);

   // Pinned allocator
   auto pinnedAllocator = rm.getAllocator("PINNED_POOL"); // Initialized above
   auto customPinnedAllocator =
      rm.makeAllocator<umpire::strategy::QuickPool>("CUSTOM_PINNED_POOL",
                                                    pinnedAllocator,
                                                    8*1024*1024,
                                                    2*1024*1024);

   // Managed allocator
   auto managedAllocator = rm.getAllocator("UM"); // Umpire default

   // Make a unified memory pool to draw from (not done in init_care_for_testing())
   auto managedPoolAllocator =
      rm.makeAllocator<umpire::strategy::QuickPool>("UM_POOL",
                                                    managedAllocator,
                                                    128*1024*1024,
                                                    8*1024*1024);

   auto customManagedAllocator =
      rm.makeAllocator<umpire::strategy::QuickPool>("CUSTOM_UM_POOL",
                                                    managedPoolAllocator,
                                                    8*1024*1024,
                                                    2*1024*1024);

   care::CudaUmpireResource resource(customDeviceAllocator,
                                     customPinnedAllocator,
                                     customManagedAllocator);
}

#endif // CARE_GPUCC

