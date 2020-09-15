//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_CARE_H_
#define _CARE_CARE_H_

// CARE config header
#include "care/config.h"

// ******** Whether RAJA HAS DETECTED GPU ACTIVE ****
#ifdef CARE_GPUCC
#ifdef GPU_ACTIVE
#define RAJA_GPU_ACTIVE
#endif
#endif

// Other CARE headers
#include "care/CHAICallback.h"
#include "care/CUDAWatchpoint.h"
#include "care/DefaultMacros.h"
#include "care/FOREACHMACRO.h"
#include "care/PointerTypes.h"
#include "care/Setup.h"
#include "care/atomic.h"
#include "care/openmp.h"
#include "care/policies.h"
#include "care/forall.h"
#include "care/scan.h"

#endif // !defined(_CARE_CARE_H_)

