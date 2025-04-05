//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/DebugPlugin.h"
#include "chai/ExecutionSpaces.hpp"
#include "care/CHAICallback.h"
#include "care/PluginData.h"

// Std library headers
#if defined(CARE_DEBUG) && !defined(_WIN32)
#include <execinfo.h>
#endif // defined(CARE_DEBUG) && !defined(_WIN32)

#include <unordered_set>

namespace care{
   void DebugPlugin::registerPlugin() {
      static RAJA::util::PluginRegistry::add<care::DebugPlugin> L ("Debug plugin", "CARE plugin for debugging");
   }

   void DebugPlugin::preLaunch(const RAJA::util::PluginContext& /* p */) {
#if !defined(CHAI_DISABLE_RM)
      // Prepare to record CHAI data
      if (CHAICallback::isActive()) {
         PluginData::clearActivePointers();
      }
#endif // !defined(CHAI_DISABLE_RM)
   }


   void DebugPlugin::postLaunch(const RAJA::util::PluginContext& p) {
#if !defined(CHAI_DISABLE_RM)
      chai::ExecutionSpace space;

      switch (p.platform) {
         case RAJA::Platform::host:
            space = chai::CPU; break;
#if defined(CHAI_ENABLE_CUDA)
         case RAJA::Platform::cuda:
            space = chai::GPU; break;
#endif
#if defined(CHAI_ENABLE_HIP)
         case RAJA::Platform::hip:
            space = chai::GPU; break;
#endif
         default:
            space = chai::NONE;
      }

      if (CHAICallback::isActive()) {			
         writeLoopData(space, PluginData::getFileName(), PluginData::getLineNumber());
      }

      if (PluginData::isParallelContext()) {
         for (auto const & it : PluginData::get_post_parallel_forall_actions()) {
             it.second(space, PluginData::getFileName(), PluginData::getLineNumber());
          }
          PluginData::clear_post_parallel_forall_actions();
      }
#endif // !defined(CHAI_DISABLE_RM)
   }

   void DebugPlugin::writeLoopData(chai::ExecutionSpace space, const char * fileName, int lineNumber) {
      if (CHAICallback::loggingIsEnabled()) {
         const int s_log_data = CHAICallback::getLogData();

         if (s_log_data == (int) space ||
             s_log_data == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
            // Get the output file
            FILE* callback_output_file = CHAICallback::getLogFile();

            // Write the loop header
            int numArrays = 0;
            std::unordered_set<const chai::PointerRecord*> usedRecords;

            for (const chai::PointerRecord* record : PluginData::getActivePointers()) {
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

            for (const chai::PointerRecord* record : PluginData::getActivePointers()) {
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
}

