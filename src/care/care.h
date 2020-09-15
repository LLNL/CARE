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
#include "care/CHAICallback.h"
#include "care/CUDAWatchpoint.h"
#include "care/FOREACHMACRO.h"
#include "care/Setup.h"

// Other library headers
#include "chai/ManagedArray.hpp"

#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

#if defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP)
   #include <omp.h>
#endif

#if defined __CUDACC__ && defined GPU_ACTIVE

// Our NVCC build is NVCC wrapping gcc
#if CHAI_GPU_SIM_MODE
#define RAJA_COMPILER_GNU
#else
#define RAJA_COMPILER_GNU
#define RAJA_USE_CUDA
#endif

#else // defined __CUDACC__ and defined GPU_ACTIVE

// These currently only do things like set macros that RAJA users can use for cross platform
// alignment and things like that. We don't use them. Not setting this produces scary
// build messages.
#if defined(__INTEL_COMPILER)
#define RAJA_COMPILER_INTEL
#define RAJA_COMPILER_ICC
#elif defined(_WIN32)
#define RAJA_COMPILER_MSVC
#endif

#define CARE_STRINGIFY(x) CARE_DO_STRINGIFY(x)
#define CARE_DO_STRINGIFY(x) #x
#ifdef _WIN32
#define CARE_PRAGMA(x) __pragma(x)
#else
#define CARE_PRAGMA(x) _Pragma(CARE_STRINGIFY(x))
#endif

#endif // defined __CUDACC__ and defined GPU_ACTIVE

// take a look at RAJA/RAJA.hpp for more platform options
#include "RAJA/RAJA.hpp"

using RAJAAtomic = RAJA::auto_atomic;

#define ATOMIC_ADD(ref, inc) RAJA::atomicAdd<RAJAAtomic>(&(ref), inc)
#define ATOMIC_MIN(ref, val) RAJA::atomicMin<RAJAAtomic>(&(ref), val)
#define ATOMIC_MAX(ref, val) RAJA::atomicMax<RAJAAtomic>(&(ref), val)
#define ATOMIC_OR(ref, val)  RAJA::atomicOr<RAJAAtomic>(&(ref), val)
#define ATOMIC_AND(ref, val) RAJA::atomicAnd<RAJAAtomic>(&(ref), val)
#define ATOMIC_XOR(ref, val) RAJA::atomicXor<RAJAAtomic>(&(ref), val)

// RAJADeviceExec is the device execution policy
// on this platform, irrespective of whether GPU_ACTIVE is set.
#if defined (CARE_GPUCC)

#define CARE_CUDA_BLOCK_SIZE 256
#define CARE_CUDA_ASYNC true

#if CHAI_GPU_SIM_MODE
using RAJADeviceExec = RAJA::seq_exec;
#else // CHAI_GPU_SIM_MODE
#if defined (__CUDACC__)
using RAJADeviceExec = RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC> ;
#elif defined(__HIPCC__)
using RAJADeviceExec = RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC> ;
#endif // __CUDACC__
#endif // CHAI_GPU_SIM_MDOE

#elif defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP) // CARE_GPUCC

using RAJADeviceExec = RAJA::omp_parallel_for_exec ;

#else // CARE_GPUCC

using RAJADeviceExec = RAJA::seq_exec;

#endif // CARE_GPUCC

#if defined (CARE_GPUCC) && defined (GPU_ACTIVE)

#if CHAI_GPU_SIM_MODE

template <class T>
using RAJAReduceMax = RAJA::ReduceMax< RAJA::seq_reduce, T>  ;
template<class T>
using RAJAReduceMin = RAJA::ReduceMin< RAJA::seq_reduce, T>  ;
template<class T>
using RAJAReduceMinLoc = RAJA::ReduceMinLoc< RAJA::seq_reduce, T>  ;
template<class T>
using RAJAReduceMaxLoc = RAJA::ReduceMaxLoc< RAJA::seq_reduce, T>  ;
template<class T>
using RAJAReduceSum = RAJA::ReduceSum< RAJA::seq_reduce, T>  ;
using RAJAExec = RAJADeviceExec ;
#define thrustExec thrust::seq
#define RAJA_PARALLEL_ACTIVE

#elif defined(__CUDACC__) // CHAI_GPU_SIM_MODE

using RAJACudaReduce = RAJA::cuda_reduce ;

template <class T>
using RAJAReduceMax = RAJA::ReduceMax<RAJACudaReduce, T> ;
template<class T>
using RAJAReduceMin = RAJA::ReduceMin<RAJACudaReduce, T> ;
template<class T>
using RAJAReduceMinLoc = RAJA::ReduceMinLoc<RAJACudaReduce, T> ;
template<class T>
using RAJAReduceMaxLoc = RAJA::ReduceMaxLoc<RAJACudaReduce, T> ;
template<class T>
using RAJAReduceSum = RAJA::ReduceSum<RAJACudaReduce, T> ;
using RAJACudaExec = RAJADeviceExec ;
using RAJAExec = RAJADeviceExec ;

#define thrustExec thrust::device
#define RAJA_PARALLEL_ACTIVE

#else // CHAI_GPU_SIM_MODE

// The defined(__HIPCC__) case is here: 
using RAJAHipReduce = RAJA::hip_reduce ;

template <class T>
using RAJAReduceMax = RAJA::ReduceMax<RAJAHipReduce, T> ;
template<class T>
using RAJAReduceMin = RAJA::ReduceMin<RAJAHipReduce, T> ;
template<class T>
using RAJAReduceMinLoc = RAJA::ReduceMinLoc<RAJAHipReduce, T> ;
template<class T>
using RAJAReduceMaxLoc = RAJA::ReduceMaxLoc<RAJAHipReduce, T> ;
template<class T>
using RAJAReduceSum = RAJA::ReduceSum<RAJAHipReduce, T> ;
using RAJAHipExec = RAJADeviceExec ;
using RAJAExec = RAJADeviceExec ;

#define thrustExec thrust::device
#define RAJA_PARALLEL_ACTIVE

#endif // CHAI_GPU_SIM_MODE

#elif defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP) // CARE_GPUCC and GPU_ACTIVE
template <class T>
using RAJAReduceMax = RAJA::ReduceMax< RAJA::omp_reduce, T>  ;
template<class T>
using RAJAReduceMin = RAJA::ReduceMin< RAJA::omp_reduce, T>  ;
template<class T>
using RAJAReduceMinLoc = RAJA::ReduceMinLoc< RAJA::omp_reduce, T>  ;
template<class T>
using RAJAReduceMaxLoc = RAJA::ReduceMaxLoc< RAJA::omp_reduce, T>  ;
template<class T>
using RAJAReduceSum = RAJA::ReduceSum< RAJA::omp_reduce, T>  ;
using RAJAExec = RAJADeviceExec ;
#define thrustExec thrust::host
#define RAJA_PARALLEL_ACTIVE

#else // CARE_GPUCC and GPU_ACTIVE

template <class T>
using RAJAReduceMax = RAJA::ReduceMax< RAJA::seq_reduce, T>  ;
template<class T>
using RAJAReduceMin = RAJA::ReduceMin< RAJA::seq_reduce, T>  ;
template<class T>
using RAJAReduceMinLoc = RAJA::ReduceMinLoc< RAJA::seq_reduce, T>  ;
template<class T>
using RAJAReduceMaxLoc = RAJA::ReduceMaxLoc< RAJA::seq_reduce, T>  ;
template<class T>
using RAJAReduceSum = RAJA::ReduceSum< RAJA::seq_reduce, T>  ;
using RAJAExec = RAJA::seq_exec ;
#define thrustExec thrust::seq

#endif // CARE_GPUCC

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

