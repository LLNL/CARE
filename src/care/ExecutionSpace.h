//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

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
   
   // the ZERO_COPY memory space. Typically PINNED memory, but may be a different space depending on
   // how CHAI and CARE are configured.
   extern CARE_DLL_API care::ExecutionSpace ZERO_COPY;
   // the  PAGEABLE memory space. Typically UM, but may be a different space depending on how
   // CHAI and CARE are configured.
   extern CARE_DLL_API care::ExecutionSpace PAGEABLE;
   // the  DEFAULT memory space. Typically GPU for GPU platforms and CPU for CPU platforms, but may be a different space depending on how
   // CHAI and CARE are configured.
   extern CARE_DLL_API care::ExecutionSpace DEFAULT;
} // namespace care

namespace chai {
   // the ZERO_COPY memory space. Typically PINNED memory, but may be a different space depending on
   // how CHAI and CARE are configured.
   extern CARE_DLL_API chai::ExecutionSpace ZERO_COPY;
   // the  PAGEABLE memory space. Typically UM, but may be a different space depending on how
   // CHAI and CARE are configured.
   extern CARE_DLL_API chai::ExecutionSpace PAGEABLE;
   // the  DEFAULT memory space. Typically GPU for GPU platforms and CPU for CPU platforms, but may be a different space depending on how
   // CHAI and CARE are configured.
   extern CARE_DLL_API chai::ExecutionSpace DEFAULT;
} // namespace chai

#endif // !defined(_CARE_EXECUTION_SPACE_H_)

