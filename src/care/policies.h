//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_POLICIES_H_
#define _CARE_POLICIES_H_

#include "RAJA/RAJA.hpp"

namespace care {
   struct sequential {};
   struct openmp {};
   struct gpu {};
   struct parallel {};
   struct raja_fusible {};
   struct raja_fusible_seq {};
   struct raja_chai_everywhere {};
   struct gpu_simulation {};

   enum class Policy {
      none = -1,
      sequential,
      openmp,
      gpu,
      parallel,
      raja_chai_everywhere
   };
} // namespace care




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




#endif // !defined(_CARE_POLICIES_H_)
