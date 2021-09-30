//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_FORALL_H_
#define _CARE_FORALL_H_

/////////////////////////////////////////////////////////////////////////////////
///
/// This file provides forall methods that are extensions to RAJA and CHAI.
/// Each forall method takes a tag struct representing a policy. The list of
/// available tag structs is in policies.h.
///
/////////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/policies.h"
#include "care/RAJAPlugin.h"
#include "care/util.h"

// other library headers
#include "chai/ArrayManager.hpp"
#include "RAJA/RAJA.hpp"

namespace care {
#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
   static bool s_reverseLoopOrder = false;
#endif

   template <typename T>
   struct ExecutionPolicyToSpace {
      static constexpr const chai::ExecutionSpace value = chai::CPU;
   };

#if defined(__CUDACC__)
   template <>
   struct ExecutionPolicyToSpace<RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>> {
      static constexpr const chai::ExecutionSpace value = chai::GPU;
   };
#elif defined (__HIPCC__)
   template <>
   struct ExecutionPolicyToSpace<RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>> {
      static constexpr const chai::ExecutionSpace value = chai::GPU;
   };
#endif

#if CARE_ENABLE_GPU_SIMULATION_MODE
   template <>
   struct ExecutionPolicyToSpace<gpu_simulation> {
      static constexpr const chai::ExecutionSpace value = chai::GPU;
   };
#endif

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson, Alan Dayton
   ///
   /// @brief Loops over the given indices and calls the loop body with each index.
   ///        This overload is CHAI and RAJA aware and sets the execution space accordingly.
   ///
   /// @arg[in] policy Used to choose this overload of forall
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename ExecutionPolicy, typename LB>
   void forall(ExecutionPolicy /* policy */, const char * fileName, const int lineNumber,
               const int start, const int end, LB&& body) {
      const int length = end - start;

      if (length != 0) {
#ifndef CARE_DISABLE_RAJAPLUGIN
         RAJAPlugin::pre_forall_hook(ExecutionPolicyToSpace<ExecutionPolicy>::value, fileName, lineNumber);
#endif

#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
         RAJA::RangeStrideSegment rangeSegment(start, end, 1);

         if (s_reverseLoopOrder) {
            rangeSegment = RAJA::RangeStrideSegment(end - 1, start - 1, -1);
         }
         else {
            rangeSegment = RAJA::RangeStrideSegment(start, end, 1);
         }
#else
         RAJA::RangeSegment rangeSegment = RAJA::RangeSegment(start, end);
#endif

#if CARE_ENABLE_GPU_SIMULATION_MODE
         RAJA::forall<RAJA::seq_exec>(rangeSegment, body);
#else
         RAJA::forall<ExecutionPolicy>(rangeSegment, body);
#endif

#ifndef CARE_DISABLE_RAJAPLUGIN
         RAJAPlugin::post_forall_hook(ExecutionPolicyToSpace<ExecutionPolicy>::value, fileName, lineNumber);
#endif
      }
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief Execute on the host. This specialization is needed for clang-query.
   ///
   /// @arg[in] sequential Used to choose this overload of forall
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(sequential, const char * fileName, const int lineNumber,
               const int start, const int end, LB&& body) {
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, body);
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief If openmp is available, execute using an openmp policy. Otherwise,
   ///        execute sequentially. This specialization is needed for clang-query.
   ///
   /// @arg[in] openmp Used to choose this overload of forall
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(openmp, const char * fileName, const int lineNumber,
               const int start, const int end, LB&& body) {
#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = true;
#endif

#if defined(_OPENMP) && defined(OPENMP_ACTIVE)
      forall(RAJA::omp_parallel_for_exec{}, fileName, lineNumber, start, end, body);
#else
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, body);
#endif

#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = false;
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief If GPU is available, execute on the device. Otherwise, execute on
   ///        the host. This specialization is needed for clang-query.
   ///
   /// @arg[in] gpu Used to choose this overload of forall
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(gpu, const char * fileName, const int lineNumber,
               const int start, const int end, LB&& body) {
#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = true;
#endif

#if defined(GPU_ACTIVE) && CARE_ENABLE_GPU_SIMULATION_MODE
      forall(gpu_simulation{}, fileName, lineNumber, start, end, body);
#elif defined(GPU_ACTIVE) && defined(__CUDACC__)
      forall(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, body);
#elif defined(GPU_ACTIVE) && defined(__HIPCC__)
      forall(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, body);
#else
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, body);
#endif

#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = false;
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief If GPU is available, execute on the device. If GPU is not available
   ///        but openmp is, execute an openmp policy. Otherwise, execute on the host.
   ///        This specialization is needed for clang-query.
   ///
   /// @arg[in] parallel Used to choose this overload of forall
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(parallel, const char * fileName, const int lineNumber,
               const int start, const int end, LB&& body) {
#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = true;
#endif

#if defined(GPU_ACTIVE) && CARE_ENABLE_GPU_SIMULATION_MODE
      forall(gpu_simulation{}, fileName, lineNumber, start, end, body);
#elif defined(GPU_ACTIVE) && defined(__CUDACC__)
      forall(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, body);
#elif defined(GPU_ACTIVE) && defined(__HIPCC__)
      forall(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, body);
#elif defined(_OPENMP) && defined(OPENMP_ACTIVE)
      forall(RAJA::omp_parallel_for_exec{}, fileName, lineNumber, start, end, body);
#else
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, body);
#endif

#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = false;
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief Executes a group of fused loops. This overload is CHAI aware and sets
   ///        the execution space accordingly.
   ///
   /// @arg[in] raja_fusible_seq Used to choose this overload of forall
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] fused Whether or not the loops have been fused
   /// @arg[in] action The index of the current loop to fuse/execute
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB, typename ...XARGS>
   void forall(raja_fusible_seq, int start, int end, LB && body, XARGS ...xargs) {
      const int length = end - start;

      if (length != 0) {
         chai::ArrayManager* threadRM = chai::ArrayManager::getInstance();
         threadRM->setExecutionSpace(chai::CPU);

         /* trigger the chai copy constructors in captured variables */
         LB my_body = body;

         for (int i = 0; i < length; ++i) {
            my_body(i,nullptr,xargs...);
         }

         threadRM->setExecutionSpace(chai::NONE);
      }
   }

#if defined(CARE_GPUCC)

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief Executes a group of fused loops. This overload is CHAI aware and sets
   ///        the execution space accordingly.
   ///
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] fused Whether or not the loops have been fused
   /// @arg[in] action The index of the current loop to fuse/execute
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB, typename ...XARGS>
   __global__ void forall_fusible_kernel(LB body, int start, int end, XARGS... xargs){
      int i = blockDim.x * blockIdx.x + threadIdx.x;
      int length =  end - start;

      if (i < length) {
         body(i, nullptr, xargs...);
      }
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief Executes a group of fused loops. This overload is CHAI aware and sets
   ///        the execution space accordingly.
   ///
   /// @arg[in] raja_fusible Used to choose this overload of forall
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] fused Whether or not the loops have been fused
   /// @arg[in] action The index of the current loop to fuse/execute
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB, typename ...XARGS>
   void forall(raja_fusible, int start, int end, LB && body, const char * fileName, int lineNumber, XARGS ...xargs){
      const int length = end - start;

      if (length != 0) {
         chai::ArrayManager* threadRM = chai::ArrayManager::getInstance();
         threadRM->setExecutionSpace(chai::GPU);

#if CARE_ENABLE_GPU_SIMULATION_MODE
         /* trigger the chai copy constructors in captured variables */
         LB my_body = body;

         for (int i = 0; i < length; ++i) {
            my_body(i,nullptr , xargs...);
         }
#else
         size_t blockSize = CARE_CUDA_BLOCK_SIZE;
         size_t gridSize = length / blockSize + 1;
         forall_fusible_kernel<<<gridSize, blockSize>>>(body, start, end, xargs...);

#if FORCE_SYNC
         care::gpuDeviceSynchronize(fileName, lineNumber);
#endif
#endif

         threadRM->setExecutionSpace(chai::NONE);
      }
   }

#endif

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief Loops over the given indices and calls the loop body with each index.
   ///        This overload is CHAI and RAJA aware. It sets the execution space
   ///        accordingly and calls RAJA::forall. First executes on the host, and
   ///        then on the device.
   /// @note In GPU_SIM mode, this does not simulate the call on the device because
   ///       there is only a single pointer for the managed_ptr in GPU_SIM mode.
   ///       If managed_ptr is ever updated to use a separate space in GPU_SIM mode,
   ///       an additional sequential RAJA::forall in the GPU space must be added
   ///       for GPU_SIM mode.
   ///
   /// @arg[in] managed_ptr_update Used to choose this overload of forall
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(managed_ptr_update, const char * fileName, int lineNumber,
               int start, const int end, LB body) {
      // preLoopPrint and postLoopPrint are handled in this call.
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, body);

#if defined(CARE_GPUCC)
      const int length = end - start;

      if (length != 0) {
         chai::ArrayManager* threadRM = chai::ArrayManager::getInstance();
         threadRM->setExecutionSpace(chai::GPU);

#if defined(__CUDACC__)
         RAJA::forall< RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>>(RAJA::RangeSegment(start, end), body);
#elif defined(__HIPCC__)
         RAJA::forall< RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>>(RAJA::RangeSegment(start, end), body);
#endif

#if FORCE_SYNC && defined(CARE_GPUCC)
         care::gpuDeviceSynchronize(fileName, lineNumber);
#endif

         threadRM->setExecutionSpace(chai::NONE);
      }
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief Loops over the given indices and calls the loop body with each index.
   ///        This overload takes a run time selectable policy.
   ///
   /// @arg[in] policy Run time policy used to select the backend to execute on.
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(Policy&& policy, const char * fileName, const int lineNumber,
               const int start, const int end, LB&& body) {
      switch (policy) {
         case Policy::sequential:
            forall(sequential{}, fileName, lineNumber, start, end, body);
            break;
         case Policy::openmp:
            forall(openmp{}, fileName, lineNumber, start, end, body);
            break;
         case Policy::gpu:
            forall(gpu{}, fileName, lineNumber, start, end, body);
            break;
         case Policy::parallel:
            forall(parallel{}, fileName, lineNumber, start, end, body);
            break;
         case Policy::managed_ptr_update:
            forall(managed_ptr_update{}, fileName, lineNumber, start, end, body);
            break;
         default:
            std::cout << "[CARE] Error: Invalid policy!" << std::endl;
            std::abort();
            break;
      }
   }
} // namespace care

#endif // !defined(_CARE_FORALL_H_)

