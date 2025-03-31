//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_POLICIES_H_
#define _CARE_POLICIES_H_

#include "RAJA/RAJA.hpp"

namespace care {
   struct sequential {};
   struct openmp {};
   struct gpu {};
   struct gpu_reduce {};
   struct parallel {};
   struct raja_fusible {};
   struct raja_fusible_seq {};
   struct managed_ptr_read {};
   struct managed_ptr_write {};
   struct gpu_simulation {};

   enum class Policy {
      none = -1,
      sequential,
      openmp,
      gpu,
      gpu_reduce,
      parallel,
      managed_ptr_read,
      managed_ptr_write
   };
} // namespace care




// RAJADeviceExec is the "device" execution policy
// on this platform, irrespective of whether a GPU is available.
#if defined (CARE_GPUCC)

#define CARE_CUDA_BLOCK_SIZE 256
#define CARE_CUDA_ASYNC true

#if CARE_ENABLE_GPU_SIMULATION_MODE
using RAJADeviceExec = RAJA::seq_exec;
#else // CARE_ENABLE_GPU_SIMULATION_MODE
#if defined (__CUDACC__)
using RAJADeviceExec = RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC> ;
#define CARE_PARALLEL_DEVICE
#elif defined(__HIPCC__)
using RAJADeviceExec = RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC> ;
#define CARE_PARALLEL_DEVICE
#endif // __CUDACC__
#endif // CARE_ENABLE_GPU_SIMULATION_MODE

#elif defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP) // CARE_GPUCC

using RAJADeviceExec = RAJA::omp_parallel_for_exec ;
#define CARE_PARALLEL_DEVICE

#else // CARE_GPUCC

using RAJADeviceExec = RAJA::seq_exec;

#endif // CARE_GPUCC

//  reduction kernel policy
#if defined(__HIPCC__)
using RAJAReductionExec = RAJA::hip_exec_with_reduce<256, CARE_CUDA_ASYNC>;
#elif defined(__CUDACC__)
using RAJAReductionExec = RAJA::cuda_exec_with_reduce<256, CARE_CUDA_ASYNC>;
#elif defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP) // CARE_GPUCC
using RAJAReductionExec = RAJA::omp_parallel_for_exec;
#else
using RAJAReductionExec = RAJADeviceExec;
#endif



#if defined (CARE_GPUCC)

#if CARE_ENABLE_GPU_SIMULATION_MODE

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
template<class T>
using RAJAMultiReduceMax = RAJA::MultiReduceMax< RAJA::seq_multi_reduce, T>  ;
template<class T>
using RAJAMultiReduceMin = RAJA::MultiReduceMin< RAJA::seq_multi_reduce, T>  ;
template<class T>
using RAJAMultiReduceSum = RAJA::MultiReduceSum< RAJA::seq_multi_reduce, T>  ;
using RAJAExec = RAJADeviceExec ;
#define RAJA_PARALLEL_ACTIVE

#elif defined(__CUDACC__) // CARE_ENABLE_GPU_SIMULATION_MODE

using RAJACudaReduce = RAJA::cuda_reduce_atomic ;
using RAJACudaMultiReduce = RAJA::cuda_multi_reduce_atomic ;

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
template<class T>
using RAJAMultiReduceMax = RAJA::MultiReduceMax< RAJACudaMultiReduce, T>  ;
template<class T>
using RAJAMultiReduceMin = RAJA::MultiReduceMin< RAJACudaMultiReduce, T>  ;
template<class T>
using RAJAMultiReduceSum = RAJA::MultiReduceSum< RAJACudaMultiReduce, T>  ;
using RAJACudaExec = RAJADeviceExec ;
using RAJAExec = RAJADeviceExec ;

#define RAJA_PARALLEL_ACTIVE

#else // CARE_ENABLE_GPU_SIMULATION_MODE

// The defined(__HIPCC__) case is here:
using RAJAHipReduce = RAJA::hip_reduce_atomic ;
using RAJAHipMultiReduce = RAJA::hip_multi_reduce_atomic ;

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
template<class T>
using RAJAMultiReduceMax = RAJA::MultiReduceMax< RAJAHipMultiReduce, T>  ;
template<class T>
using RAJAMultiReduceMin = RAJA::MultiReduceMin< RAJAHipMultiReduce, T>  ;
template<class T>
using RAJAMultiReduceSum = RAJA::MultiReduceSum< RAJAHipMultiReduce, T>  ;
using RAJAHipExec = RAJADeviceExec ;
using RAJAExec = RAJADeviceExec ;

#define RAJA_PARALLEL_ACTIVE

#endif // CARE_ENABLE_GPU_SIMULATION_MODE

#elif defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP) // CARE_GPUCC
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
template<class T>
using RAJAMultiReduceMax = RAJA::MultiReduceMax< RAJA::omp_multi_reduce, T>  ;
template<class T>
using RAJAMultiReduceMin = RAJA::MultiReduceMin< RAJA::omp_multi_reduce, T>  ;
template<class T>
using RAJAMultiReduceSum = RAJA::MultiReduceSum< RAJA::omp_multi_reduce, T>  ;
using RAJAExec = RAJADeviceExec ;
#define RAJA_PARALLEL_ACTIVE

#else // CARE_GPUCC

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
template<class T>
using RAJAMultiReduceMax = RAJA::MultiReduceMax< RAJA::seq_multi_reduce, T>  ;
template<class T>
using RAJAMultiReduceMin = RAJA::MultiReduceMin< RAJA::seq_multi_reduce, T>  ;
template<class T>
using RAJAMultiReduceSum = RAJA::MultiReduceSum< RAJA::seq_multi_reduce, T>  ;
using RAJAExec = RAJA::seq_exec ;

#endif // CARE_GPUCC




#endif // !defined(_CARE_POLICIES_H_)
