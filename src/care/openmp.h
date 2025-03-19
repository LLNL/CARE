//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_OPENMP_H
#define CARE_OPENMP_H

#include "care/config.h"
#include "RAJA/RAJA.hpp"

// Other library headers
#if defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP)
   #include <omp.h>
#endif

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

#endif // CARE_OPENMP_H
