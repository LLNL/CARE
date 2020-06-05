//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/CHAICallback.h"
#include "care/RAJAPlugin.h"
#include "care/Setup.h"
#include "care/util.h"

// Other library headers
#include "chai/ArrayManager.hpp"

/* CUDA profiling macros */
#if defined(__CUDACC__) && CARE_HAVE_NVTOOLSEXT
#include "nvToolsExt.h"
#endif

// Std library headers
#if defined(CARE_DEBUG) && !defined(_WIN32)
#include <execinfo.h>
#endif // defined(CARE_DEBUG) && !defined(_WIN32)

#include <unordered_set>

namespace care {
   bool RAJAPlugin::s_update_chai_execution_space = true;
   bool RAJAPlugin::s_debug_chai_data = true;
   bool RAJAPlugin::s_profile_host_loops = true;
   bool RAJAPlugin::s_synchronize_before = false;
   bool RAJAPlugin::s_synchronize_after = false;

   uint32_t RAJAPlugin::s_colors[7] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
   int RAJAPlugin::s_num_colors = sizeof(s_colors) / sizeof(uint32_t);
   unsigned int RAJAPlugin::s_current_color = 0;

   std::string RAJAPlugin::s_current_loop_file_name = "N/A";
   int RAJAPlugin::s_current_loop_line_number = -1;

   std::vector<const chai::PointerRecord*> RAJAPlugin::s_active_pointers_in_loop = std::vector<const chai::PointerRecord*>{};

   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @brief Set up to be done before executing a RAJA loop.
   ///
   /// @arg[in] space The execution space
   /// @arg[in] name The name of the loop
   ///
   /////////////////////////////////////////////////////////////////////////////////
   void RAJAPlugin::pre_forall_hook(chai::ExecutionSpace space, const char* fileName, int lineNumber) {
#if !defined(CHAI_DISABLE_RM)
      // Update the CHAI execution space
      if (s_update_chai_execution_space) {
         chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
         arrayManager->setExecutionSpace(space);
      }

      // Prepare to record CHAI data
      if (CHAICallback::isActive()) {
         s_current_loop_file_name = fileName;
         s_current_loop_line_number = lineNumber;
         s_active_pointers_in_loop.clear();

#if defined(__CUDACC__) && defined(CARE_DEBUG)
         CUDAWatchpoint::setOrCheckWatchpoint<int>();
#endif // defined(__CUDACC__) && defined(CARE_DEBUG)
      }
#endif // !defined(CHAI_DISABLE_RM)

#if defined(__CUDACC__)
      // Synchronize
      if (s_synchronize_before) {
         if (space == chai::GPU) {
            care::gpuAssert(::cudaDeviceSynchronize(), fileName, lineNumber, true);
         }
      }

#if CARE_HAVE_NVTOOLSEXT
      // Profile the host loops
      if (s_profile_host_loops) {
         if (space == chai::CPU) {
            std::string name = fileName + std::to_string(lineNumber);

            int color_id = s_current_color++;
            color_id = color_id % s_num_colors;

            // TODO: Add error checking
            nvtxEventAttributes_t eventAttrib = { 0 };
            eventAttrib.version = NVTX_VERSION;
            eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib.colorType = NVTX_COLOR_ARGB;
            eventAttrib.color = s_colors[color_id];
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib.message.ascii = name.c_str();
            nvtxRangePushEx(&eventAttrib);
         }
      }
#endif // CARE_HAVE_NVTOOLSEXT
#endif // defined(__CUDACC__)
   }

   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @brief Writes out debugging information after a loop is executed.
   ///
   /// @arg[in] space The execution space
   /// @arg[in] fileName The file where the loop macro was called
   /// @arg[in] lineNumber The line number where the loop macro was called
   ///
   /////////////////////////////////////////////////////////////////////////////////
   void RAJAPlugin::writeLoopData(chai::ExecutionSpace space,
                                  const char * fileName,
                                  int lineNumber) {
      if (CHAICallback::loggingIsEnabled()) {
         const int s_log_data = CHAICallback::getLogData();

         if (s_log_data == (int) space ||
             s_log_data == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
            // Get the output file
            FILE* callback_output_file = CHAICallback::getLogFile();

            // Write the loop header
            int numArrays = 0;
            std::unordered_set<const chai::PointerRecord*> usedRecords;

            for (const chai::PointerRecord* record : s_active_pointers_in_loop) {
               if (usedRecords.count(record) > 0) {
                  continue;
               }
               else {
                  usedRecords.emplace(record);
                  ++numArrays;
               }
            }

            fprintf(callback_output_file, "[CARE] AFTER LOOP EXECUTION %s:%i (%i arrays)\n", fileName, lineNumber, numArrays);

#if defined(CARE_DEBUG) && !defined(_WIN32)
            // Write the stack trace
            const int stackDepth = 16;
            void *stackArray[stackDepth];
            size_t stackSize = backtrace(stackArray, stackDepth);
            char **stackStrings = backtrace_symbols(stackArray, stackSize);

            // Skip the first two contexts
            for (size_t i = 2 ; i < stackSize ; ++i) {
               fprintf(callback_output_file, "[CARE] [STACK] %lu: %s\n", i, stackStrings[i]);
            }

            if (CHAICallback::isDiffFriendly()) {
               // For diff-friendly output, keep the number of lines constant
               for (size_t i = stackSize ; i < stackDepth ; ++i) {
                  fprintf(callback_output_file, "[CARE] [STACK] %lu: <empty>\n", i);
               }
            }

            free(stackStrings);
#endif // defined(CARE_DEBUG) && !defined(_WIN32)

            // Flush to the output file
            fflush(callback_output_file);

#ifdef CARE_DEBUG
            // Write the arrays captured in the loop
            usedRecords.clear();

            for (const chai::PointerRecord* record : s_active_pointers_in_loop) {
               if (record && usedRecords.find(record) == usedRecords.end()) {
                  usedRecords.emplace(record);
                  CHAICallback::writeArray(record, space);
                  fflush(callback_output_file);
               }
            }
#else // CARE_DEBUG
            // Write message that -logchaidata is not supported
            fprintf(callback_output_file, "[CARE] [CHAI] -logchaidata is not supported in this build\n");
            fflush(callback_output_file);
#endif // CARE_DEBUG
         }
      }
   }

   void RAJAPlugin::post_forall_hook(chai::ExecutionSpace space, const char* fileName, int lineNumber) {
#if defined(__CUDACC__)
#if CARE_HAVE_NVTOOLSEXT
      if (s_profile_host_loops) {
         if (space == chai::CPU) {
            // TODO: Add error checking
            nvtxRangePop();
         }
      }
#endif // CARE_HAVE_NVTOOLSEXT

      if (s_synchronize_after) {
         if (space == chai::GPU) {
            care::gpuAssert(::cudaDeviceSynchronize(), fileName, lineNumber, true);
         }
      }
#endif // defined(__CUDACC__)

#if !defined(CHAI_DISABLE_RM)
      if (CHAICallback::isActive()) {
         writeLoopData(space, fileName, lineNumber);

         // Clear out the captured arrays
         s_active_pointers_in_loop.clear();

#if defined(__CUDACC__) && defined(CARE_DEBUG)
         CUDAWatchpoint::setOrCheckWatchpoint<int>();
#endif // defined(__CUDACC__) && defined(CARE_DEBUG)
      }

      if (s_update_chai_execution_space) {
         chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
         arrayManager->setExecutionSpace(chai::NONE);
      }
#endif // !defined(CHAI_DISABLE_RM)
   }

   std::string RAJAPlugin::getCurrentLoopFileName() {
      return s_current_loop_file_name;
   }

   int RAJAPlugin::getCurrentLoopLineNumber() {
      return s_current_loop_line_number;
   }

   void RAJAPlugin::addActivePointer(const chai::PointerRecord* record) {
      s_active_pointers_in_loop.emplace_back(record);
   }

   void RAJAPlugin::removeActivePointer(const chai::PointerRecord* record) {
      for (size_t i = 0; i < s_active_pointers_in_loop.size(); ++i) {
         if (s_active_pointers_in_loop[i] == record) {
            s_active_pointers_in_loop[i] = nullptr;
         }
      }
   }

   void RAJAPlugin::setSynchronization(bool synchronizeBefore,
                                       bool synchronizeAfter) {
      s_synchronize_before = synchronizeBefore;
      s_synchronize_after = synchronizeAfter;
   }
} // namespace care

