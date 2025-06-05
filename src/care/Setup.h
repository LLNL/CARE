//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_SETUP_H_
#define _CARE_SETUP_H_

// CARE headers
#include "care/config.h"
#include "care/CHAICallback.h"
#include "care/PluginData.h"

// Other library headers
#include "chai/ExecutionSpaces.hpp"
#include "chai/ArrayManager.hpp"
#include "umpire/Umpire.hpp"

#ifdef UMPIRE_ENABLE_MPI
#include "mpi.h"
#endif

// Std library headers
#include <string>

namespace care {
#ifdef UMPIRE_ENABLE_MPI
   inline void initialize(MPI_Comm communicator) {
      umpire::initialize(communicator);
   }

#else
   inline void initialize(int /*communicator*/) {
      umpire::initialize();
   }

#endif

   void initialize_pool(const std::string& resource,
                        const std::string& poolname,
                        chai::ExecutionSpace space,
                        std::size_t initial_size,
                        std::size_t min_block_size,
                        bool grows = true,
                        size_t alignment = 16);

   void initialize_pool_block_heuristic(const std::string& resource,
                                        const std::string& poolname,
                                        chai::ExecutionSpace space,
                                        std::size_t initial_size,
                                        std::size_t min_block_size,
                                        std::size_t block_coalesce_heuristic = 3,
                                        bool grows = true,
                                        size_t alignment = 16);

   void initialize_pool_percent_heuristic(const std::string& resource,
                                          const std::string& poolname,
                                          chai::ExecutionSpace space,
                                          std::size_t initial_size,
                                          std::size_t min_block_size,
                                          std::size_t percent_coalesce_heuristic = 100,
                                          bool grows = true,
                                          size_t alignment = 16);

   void dump_memory_statistics();

   inline void report_leaks() {
#if !defined(CHAI_DISABLE_RM)
      return chai::ArrayManager::getInstance()->reportLeaks();
#endif
   }

   inline void evict_memory(chai::ExecutionSpace space,
                            chai::ExecutionSpace destinationSpace) {
#if !defined(CHAI_DISABLE_RM)
      return chai::ArrayManager::getInstance()->evict(space, destinationSpace);
#endif
   }

   inline void evict_device_memory() {
#ifdef CARE_GPUCC
      evict_memory(chai::ExecutionSpace::GPU, chai::ExecutionSpace::CPU);
#endif

      return;
   }

   // Debugging output controls.
   inline void enable_logging() {
      CHAICallback::enableLogging();
   }

   inline void disable_logging() {
      CHAICallback::disableLogging();
   }

   inline void set_callback_diff_friendly(bool diffFriendly) {
      CHAICallback::setDiffFriendly(diffFriendly);
   }

   inline void set_callback_output_file(const char * fileName) {
      CHAICallback::setLogFile(fileName);
   }

   inline void set_log_chai_allocations(int log_chai_allocations) {
      CHAICallback::setLogAllocations(log_chai_allocations);
   }

   inline void set_log_chai_deallocations(int log_chai_deallocations) {
      CHAICallback::setLogDeallocations(log_chai_deallocations);
   }

   inline void set_log_chai_moves(int log_chai_moves) {
      CHAICallback::setLogMoves(log_chai_moves);
   }

   inline void set_log_chai_captures(int log_chai_captures) {
      CHAICallback::setLogCaptures(log_chai_captures);
   }

   inline void set_log_chai_abandoned(int log_chai_abandoned) {
      CHAICallback::setLogAbandoned(log_chai_abandoned);
   }

   inline void set_log_chai_leaks(int log_chai_leaks) {
      CHAICallback::setLogLeaks(log_chai_leaks);
   }

   inline void set_log_chai_data(int log_chai_data) {
      CHAICallback::setLogData(log_chai_data);
   }

   // does a GPU device synchronize if there has been a kernel launch through care
   // since the last time this was called.
   CARE_DLL_API bool syncIfNeeded();
} // namespace care

#endif // !defined(_CARE_SETUP_H_)

