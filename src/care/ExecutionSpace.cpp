//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/config.h"
#include "care/ExecutionSpace.h"

// Explicit use of CHAI spaces that can be configured to not exist is not portable,
// we define a ZERO_COPY and a PAGEABLE memory space that falls back to something
// that's guaranteed to exist if the optimal solution is not present.

namespace chai {
#if defined(CHAI_ENABLE_PINNED)
   chai::ExecutionSpace ZERO_COPY = chai::PINNED;
#elif defined(CHAI_ENABLE_UM)
   chai::ExecutionSpace ZERO_COPY = chai::UM;
#elif defined(CARE_GPU_MEMORY_IS_ACCESSIBLE_ON_CPU)
   chai::ExecutionSpace ZERO_COPY = chai::GPU;
#else
   chai::ExecutionSpace ZERO_COPY = chai::CPU;
#endif

#if defined(CHAI_ENABLE_UM)
   chai::ExecutionSpace PAGEABLE = chai::UM;
#elif defined(CARE_GPU_MEMORY_IS_ACCESSIBLE_ON_CPU)
   chai::ExecutionSpace PAGEABLE = chai::GPU;
#else
   chai::ExecutionSpace PAGEABLE = chai::CPU;
#endif

#if defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE
   chai::ExecutionSpace DEFAULT = chai::GPU;
#else
   chai::ExecutionSpace DEFAULT = chai::CPU;
#endif

}

namespace care {
#if defined(CHAI_ENABLE_PINNED)
   care::ExecutionSpace ZERO_COPY = care::PINNED;
#elif defined(CHAI_ENABLE_UM)
   care::ExecutionSpace ZERO_COPY = care::UM;
#elif defined(CARE_GPU_MEMORY_IS_ACCESSIBLE_ON_CPU)
   care::ExecutionSpace ZERO_COPY = care::GPU;
#else
   care::ExecutionSpace ZERO_COPY = care::CPU;
#endif

#if defined(CHAI_ENABLE_UM)
   care::ExecutionSpace PAGEABLE = care::UM;
#elif defined(CARE_GPU_MEMORY_IS_ACCESSIBLE_ON_CPU)
   care::ExecutionSpace PAGEABLE = care::GPU;
#else
   care::ExecutionSpace PAGEABLE = care::CPU;
#endif

#if defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE
   care::ExecutionSpace DEFAULT = care::GPU;
#else
   care::ExecutionSpace DEFAULT = care::CPU;
#endif
}

