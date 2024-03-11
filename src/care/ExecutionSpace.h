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
} // namespace care

#endif // !defined(_CARE_EXECUTION_SPACE_H_)

