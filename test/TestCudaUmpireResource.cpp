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

// care headers
#include "care/CudaUmpireResource.h"
#include "care/DefaultMacros.h"
#include "care/policies.h"
#include "care/detail/test_utils.h"

GPU_TEST(CudaUmpireResource, gpu_initialization) {
   printf("Initializing\n");
   init_care_for_testing();
   printf("Initialized... Testing care::array\n");
}

GPU_TEST(CudaUmpireResource, DefaultConstructor)
{
   care::CudaUmpireResource resource;
}

#endif // CARE_GPUCC

