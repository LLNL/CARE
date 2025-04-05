//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

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
#include "care/util.h"
#include "care/PluginData.h"

// other library headers
#include "chai/ArrayManager.hpp"
#include "chai/ExecutionSpaces.hpp"
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
   template <>
   struct ExecutionPolicyToSpace<RAJAReductionExec> {
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
   /// @arg[in] batch_size Maximum length of each kernel (0 for no limit)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename ExecutionPolicy, typename LB>
   void forall(ExecutionPolicy /* policy */, const char * fileName, const int lineNumber,
               const int start, const int end, const int batch_size, LB&& body) {
      const int length = end - start;

      if (length != 0) {
         PluginData::setFileName(fileName);
         PluginData::setLineNumber(lineNumber);

         int index = start ;
         int chunk_size = batch_size > 0 ? batch_size : length ;

         while (index < end) {
            int chunk_start = index ;
            int chunk_end = (index + chunk_size < end) ? index + chunk_size : end ;

#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
            RAJA::RangeStrideSegment rangeSegment =
               s_reverseLoopOrder ?
               RAJA::RangeStrideSegment(chunk_end - 1, chunk_start - 1, -1) :
               RAJA::RangeStrideSegment(chunk_start, chunk_end, 1);
#else
            RAJA::RangeSegment rangeSegment = RAJA::RangeSegment(chunk_start, chunk_end);
#endif

#if CARE_ENABLE_GPU_SIMULATION_MODE
            chai::ArrayManager* threadRM = chai::ArrayManager::getInstance();
            if (ExecutionPolicyToSpace<ExecutionPolicy>::value == chai::GPU) {
               threadRM->setGPUSimMode(true);
            }
            else {
               threadRM->setGPUSimMode(false);
            }
            RAJA::forall<RAJA::seq_exec>(rangeSegment, std::forward<LB>(body));
            threadRM->setGPUSimMode(false);
#else
            RAJA::forall<ExecutionPolicy>(rangeSegment, std::forward<LB>(body));
#endif

            index += chunk_size ;
         }
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
   /// @arg[in] batch_size Maximum length of each kernel (0 for no limit)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(sequential, const char * fileName, const int lineNumber,
               const int start, const int end, const int batch_size, LB&& body) {
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
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
   /// @arg[in] batch_size Maximum length of each kernel (0 for no limit)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(openmp, const char * fileName, const int lineNumber,
               const int start, const int end, const int batch_size, LB&& body) {
#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = true;
#endif

#if defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP)
      forall(RAJA::omp_parallel_for_exec{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#else
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
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
   /// @arg[in] batch_size Maximum length of each kernel (0 for no limit)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(gpu, const char * fileName, const int lineNumber,
               const int start, const int end, const int batch_size, LB&& body) {
#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = true;
#endif

#if CARE_ENABLE_GPU_SIMULATION_MODE
      forall(gpu_simulation{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#elif defined(__CUDACC__)
      forall(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#elif defined(__HIPCC__)
      forall(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#else
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
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
   /// @arg[in] batch_size Maximum length of each kernel (0 for no limit)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(parallel, const char * fileName, const int lineNumber,
               const int start, const int end, const int batch_size, LB&& body) {
#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = true;
#endif
      PluginData::setParallelContext(true);
      
#if CARE_ENABLE_GPU_SIMULATION_MODE
      forall(gpu_simulation{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#elif defined(__CUDACC__)
      forall(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#elif defined(__HIPCC__)
      forall(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#elif defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP)
      forall(RAJA::omp_parallel_for_exec{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#else
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#endif
      PluginData::setParallelContext(false);

#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = false;
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief Execute using the care::RAJAReductionExec policy
   ///
   /// @arg[in] gpu_reduce Used to choose this overload of forall
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] batch_size Maximum length of each kernel (0 for no limit)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(gpu_reduce, const char * fileName, const int lineNumber,
               const int start, const int end, const int batch_size, LB&& body) {
#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = true;
#endif
      PluginData::setParallelContext(true);

      forall(RAJAReductionExec{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));

      PluginData::setParallelContext(false);

#if CARE_ENABLE_PARALLEL_LOOP_BACKWARDS
      s_reverseLoopOrder = false;
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief If GPU is available and managed_ptr is available on the device,
   ///        execute on the device. If GPU is not available but openmp is,
   ///        execute an openmp policy. Otherwise, execute on the host.
   ///        This specialization is needed for clang-query.
   ///
   /// @arg[in] managed_ptr_read Used to choose this overload of forall
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] batch_size Maximum length of each kernel (0 for no limit)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(managed_ptr_read, const char * fileName, const int lineNumber,
               const int start, const int end, const int batch_size, LB&& body) {
#if CARE_ENABLE_GPU_SIMULATION_MODE && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      forall(gpu_simulation{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#elif defined(__CUDACC__) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      forall(RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#elif defined(__HIPCC__) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      forall(RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>{},
             fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#elif defined(_OPENMP) && defined(RAJA_ENABLE_OPENMP)
      forall(RAJA::omp_parallel_for_exec{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#else
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#endif
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Benjamin Liu
   ///
   /// @brief Helper function to execute loop body without elision of the
   ///        copy constructor in captured variables.
   ///
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB, typename ...XARGS>
   void execute_body_fusible_seq(const int length, LB body, XARGS ...xargs) {
      for (int i = 0; i < length; ++i) {
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

         execute_body_fusible_seq(length, body, xargs...);

         threadRM->setExecutionSpace(chai::NONE);
      }
   }

#if defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE

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

#endif

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
         threadRM->setGPUSimMode(true);
         execute_body_fusible_seq(length, body, xargs...);
         threadRM->setGPUSimMode(false);
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

#endif /* defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE */

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief Loops over the given indices and calls the loop body with each index.
   ///        This overload is CHAI and RAJA aware. It sets the execution space
   ///        accordingly and calls RAJA::forall. First executes on the host, and
   ///        then on the device.
   ///
   /// @arg[in] managed_ptr_write Used to choose this overload of forall
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] start The starting index (inclusive)
   /// @arg[in] end The ending index (exclusive)
   /// @arg[in] batch_size Maximum length of each kernel (0 for no limit)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(managed_ptr_write, const char * fileName, int lineNumber,
               const int start, const int end, const int batch_size, LB&& body) {
      // preLoopPrint and postLoopPrint are handled in this call.
      forall(RAJA::seq_exec{}, fileName, lineNumber, start, end, batch_size, body);

#if (defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU)
      const int length = end - start;

      if (length != 0) {
         chai::ArrayManager* threadRM = chai::ArrayManager::getInstance();
         threadRM->setExecutionSpace(chai::GPU);

#if CARE_ENABLE_GPU_SIMULATION_MODE
         forall(gpu_simulation{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
#else
         int index = start ;
         int chunk_size = batch_size > 0 ? batch_size : length ;

         while (index < end) {
            int chunk_start = index ;
            int chunk_end = (index + chunk_size < end) ? index + chunk_size : end ;
            RAJA::RangeSegment rangeSegment = RAJA::RangeSegment(chunk_start, chunk_end);

#if defined(__CUDACC__)
            RAJA::forall< RAJA::cuda_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>>(rangeSegment, body);
#elif defined(__HIPCC__)
            RAJA::forall< RAJA::hip_exec<CARE_CUDA_BLOCK_SIZE, CARE_CUDA_ASYNC>>(rangeSegment, body);
#endif

            index += chunk_size ;
         }
#endif

#if FORCE_SYNC && defined(CARE_GPUCC)
         care::gpuDeviceSynchronize(fileName, lineNumber);
#endif

         threadRM->setExecutionSpace(chai::NONE);
      }
#endif /* (defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE) && defined(CHAI_ENABLE_MANAGED_PTR_ON_GPU) */
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
   /// @arg[in] batch_size Maximum length of each kernel (0 for no limit)
   /// @arg[in] body The loop body to execute at each index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void forall(Policy&& policy, const char * fileName, const int lineNumber,
               const int start, const int end, const int batch_size, LB&& body) {
      switch (policy) {
         case Policy::sequential:
            forall(sequential{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
            break;
         case Policy::openmp:
            forall(openmp{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
            break;
         case Policy::gpu:
            forall(gpu{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
            break;
         case Policy::parallel:
            forall(parallel{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
            break;
         case Policy::managed_ptr_read:
            forall(managed_ptr_read{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
            break;
         case Policy::managed_ptr_write:
            forall(managed_ptr_write{}, fileName, lineNumber, start, end, batch_size, std::forward<LB>(body));
            break;
         default:
            std::cout << "[CARE] Error: Invalid policy!" << std::endl;
            std::abort();
            break;
      }
   }

   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief Loops over a 2 dimensional index space with varying lengths in the second dimension. 
   ///
   /// @arg[in] policy Compile time execution space to select the backend to execute on.
   /// @arg[in] xstart X dimension starting index (inclusive)
   /// @arg[in] xend  X dimension upper bound of ending index (exclusive)
   /// @arg[in] host_lengths ending index in x dimension at each y index from ystart (inclusive) to ylength (exclusive). Raw pointer should be in an appropriate memory
   ///          space for the Exec type
   /// @arg[in] ystart The starting index in the y dimension (inclusive)
   /// @arg[in] ylength The ending index in the y dimension (exclusive)
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] body The loop body to execute at each (x,y) index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB, typename Exec>
   void launch_2D_jagged(Exec /*policy*/, int xstart, int /*xend*/, int const * host_lengths, int ystart, int ylength, const char * /* fileName */, int /* lineNumber */, LB && body) {
      chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();       
      arrayManager->setExecutionSpace(ExecutionPolicyToSpace<Exec>::value);

      // intentional trigger of copy constructor for CHAI correctness
      LB body_to_call{body};
      for (int y = ystart; y < ylength; ++y) {
         for (int x = xstart ; x < host_lengths[y]; ++x) {
            body_to_call(x, y);
         }
      }
      arrayManager->setExecutionSpace(chai::ExecutionSpace::NONE);
   }

#ifdef CARE_GPUCC
   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief the GPU kernel to call from a care::gpu specialization of launch_2D_jagged
   ///
   /// @arg[in] loopBody The loop body to execute at each (x,y) index
   /// @arg[in] lengths ending index in x dimension at each y index from ystart (inclusive) to ylength (exclusive). Raw pointer should be in an appropriate memory
   ///          space for executing on the GPU, recommend PINNED memory so long as bulk of data is in the x dimension (that is sum(lengths) >> ylength)
   /// @arg[in] ylength The ending index in the y dimension (exclusive)
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   CARE_GLOBAL void care_kernel_2D(LB loopBody, int const * lengths, int ylength) {
     int x = threadIdx.x + blockIdx.x * blockDim.x; 
     int y = threadIdx.y + blockIdx.y * blockDim.y; 
     if (x < lengths[y] && y < ylength) {
        loopBody(x,y);
     }
   }
   ////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson
   ///
   /// @brief Loops over a 2 dimensional index space with varying lengths in the second dimension. 
   ///
   /// @arg[in] policy Compile time execution space to select the backend to execute on.
   /// @arg[in] xstart X dimension starting index (inclusive)
   /// @arg[in] xend  X dimension upper bound of ending index (exclusive)
   /// @arg[in] host_lengths ending index in x dimension at each y index from ystart (inclusive) to ylength (exclusive). Raw pointer should be in an appropriate memory
   ///          space for the Exec type
   /// @arg[in] ystart The starting index in the y dimension (inclusive)
   /// @arg[in] ylength The ending index in the y dimension (exclusive)
   /// @arg[in] fileName The name of the file where this function is called
   /// @arg[in] lineNumber The line number in the file where this function is called
   /// @arg[in] body The loop body to execute at each (x,y) index
   ///
   ////////////////////////////////////////////////////////////////////////////////
   template <typename LB>
   void launch_2D_jagged(care::gpu, int xstart, int xend, int const * gpu_lengths, int ystart, int ylength, const char * fileName, int lineNumber , LB && body) {
       if (xend > 0 && ylength > 0) {
          // TODO launch this kernel in the camp or RAJA default stream - not sure how to do this - for now this is a synchronous call on the CUDA/HIP default stream
          chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
          arrayManager->setExecutionSpace(chai::GPU);

          dim3 dimBlock(CARE_CUDA_BLOCK_SIZE, 1);
          dim3 dimGrid;
          dimGrid.x  = (xend/CARE_CUDA_BLOCK_SIZE)+(xend%CARE_CUDA_BLOCK_SIZE==0?0:1);
          dimGrid.y = ylength;
          care_kernel_2D<<<dimGrid, dimBlock>>>( body, gpu_lengths, ylength);
          
          arrayManager->setExecutionSpace(chai::ExecutionSpace::NONE);
       }
   }

   template <typename LB>
   void launch_2D_jagged(care::gpu_reduce, int xstart, int xend, int const * gpu_lengths, int ystart, int ylength, const char * fileName, int lineNumber , LB && body) {
      launch_2D_jagged(care::gpu{}, xstart, xend, gpu_lengths, ystart, ylength, fileName, lineNumber, body) ;
   }
#endif
} // namespace care

#endif // !defined(_CARE_FORALL_H_)

