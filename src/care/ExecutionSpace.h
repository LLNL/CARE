//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_EXECUTION_SPACE_H_
#define _CARE_EXECUTION_SPACE_H_

// CHAI headers
#include "chai/ExecutionSpaces.hpp"

namespace care {
   enum ExecutionSpace {
     NONE = chai::NONE,
     CPU = chai::CPU,
     GPU = chai::GPU,
     UM = chai::UM,
     PINNED = chai::PINNED,
     NUM_EXECUTION_SPACES = chai::NUM_EXECUTION_SPACES
   };

// Explicit use of CHAI spaces that can be configured to not exist is not portable, we define a ZERO_COPY and a PAGEABLE memory // space that falls back to something that's guaranteed to exist if the optimal solution is not present.

#if definied(CHAI_ENABLE_PINNED)
   ExecutionSpace ZERO_COPY = PINNED;
#elif defined(CHAI_ENABLE_UM)
   ExecutionSpace ZERO_COPY = UM;
#elif defined(CARE_GPU_MEMORY_IS_ACCESSIBLE_ON_CPU)
   ExecutionSpace ZERO_COPY = GPU;
#else
   ExecutionSpace ZERO_COPY = CPU;
#endif

#if defined(CHAI_ENABLE_UM)
   ExecutionSpace PAGEABLE = UM;
#elif defined(CARE_GPU_MEMORY_IS_ACCESSIBLE_ON_CPU)
   ExecutionSpace PAGEABLE = GPU;
#else
   ExecutionSpace PAGEABLE = CPU;
#endif

} // namespace care

#endif // !defined(_CARE_EXECUTION_SPACE_H_)

