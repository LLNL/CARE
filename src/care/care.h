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

// Other CARE headers
#include "care/atomic.h"
#include "care/CHAICallback.h"
#include "care/CUDAWatchpoint.h"
#include "care/FOREACHMACRO.h"
#include "care/Setup.h"

// Other library headers
#if defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP)
   #include <omp.h>
#endif

// take a look at RAJA/RAJA.hpp for more platform options
#include "RAJA/RAJA.hpp"



#if 0
#define ANNOTATE_J2(X, Y) X "_" #Y
#define ANNOTATE_J1(X, Y) ANNOTATE_J2(X, Y)
#define showloc() printf("[CARE] %s\n", ANNOTATE_J1(__FILE__, __LINE__) )
#endif

#if defined(_OPENMP) && defined(THREAD_CARE_LOOPS)

// #define OMP_FOR_BEGIN showloc() ; CARE_PRAGMA(omp parallel) { /* printf("[CARE] %d\n", omp_get_num_threads()) ; */ CARE_PRAGMA(omp for schedule(static))
// #define OMP_FOR_END }
#define OMP_FOR_BEGIN CARE_PRAGMA(omp parallel for schedule(static))
#define OMP_FOR_END

#else

#define OMP_FOR_BEGIN
#define OMP_FOR_END

#endif

typedef RAJA::TypedIndexSet<RAJA::RangeSegment, RAJA::ListSegment, RAJA::RangeStrideSegment> CAREIndexSet ;

#include "care/policies.h"
#include "care/forall.h"
#include "care/DefaultMacros.h"


// ******** Whether RAJA HAS DETECTED GPU ACTIVE ****
#ifdef CARE_GPUCC
#ifdef GPU_ACTIVE
#define RAJA_GPU_ACTIVE
#endif
#endif
// ************ DEFAULT MACRO SELECTION **************

// Define default behavior for work loops
#if defined(CARE_GPUCC)
// As of 30 July 2018, cycle and lagrange run faster with FISSION_LOOPS turned off
//#define FISSION_LOOPS 1
#define USE_PERMUTED_CONNECTIVITY 1
#else // CARE_GPUCC is False, CHAI_GPU_SIM_MODE is 0

#ifndef CARE_LEGACY_COMPATIBILITY_MODE
// As of 30 July 2018, cycle and lagrange run faster with FISSION_LOOPS turned off
//#define FISSION_LOOPS 1
#define USE_PERMUTED_CONNECTIVITY 1
#else
#define FISSION_LOOPS 0
#define USE_PERMUTED_CONNECTIVITY 0
#endif

#endif // CARE_GPUCC

#include "care/scan.h"
#include "care/PointerTypes.h"

// Priority phase value for the default loop fuser
const double CARE_DEFAULT_PHASE = -FLT_MAX/2.0;

#endif // !defined(_CARE_CARE_H_)

