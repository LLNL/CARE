//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/CHAICallback.h"
#ifndef CARE_DISABLE_RAJAPLUGIN
#include "care/RAJAPlugin.h"
#endif

// Other library headers
#include "chai/ArrayManager.hpp"
#include "chai/PointerRecord.hpp"

// Std library headers
#include <sstream>
#include <vector>

namespace care {
   bool CHAICallback::s_active = false;
   bool CHAICallback::s_logging_enabled = false;
   bool CHAICallback::s_diff_friendly = false;
   FILE* CHAICallback::s_log_file = stdout;

   int CHAICallback::s_log_allocations = 0;
   int CHAICallback::s_log_deallocations = 0;
   int CHAICallback::s_log_moves = 0;
   int CHAICallback::s_log_captures = 0;
   int CHAICallback::s_log_abandoned = 0;
   int CHAICallback::s_log_leaks = 0;
   int CHAICallback::s_log_data = 0;

   CHAICallback::NameMap& CHAICallback::getNameMap() {
      // Using this approach rather than a static member to avoid a
      // global initialization order bug.
      static NameMap s_names;
      return s_names;
   }

   CHAICallback::PrintCallbackMap& CHAICallback::getPrintCallbackMap() {
      // Using this approach rather than a static member to avoid a
      // global initialization order bug.
      static PrintCallbackMap s_print_callbacks;
      return s_print_callbacks;
   }

   bool CHAICallback::isActive() {
      return s_active;
   }

   void CHAICallback::enableLogging() {
      s_logging_enabled = true;
   }

   void CHAICallback::disableLogging() {
      s_logging_enabled = false;
   }

   bool CHAICallback::loggingIsEnabled() {
      return s_logging_enabled;
   }

   bool CHAICallback::isDiffFriendly() {
      return s_diff_friendly;
   }

   void CHAICallback::setDiffFriendly(bool diffFriendly) {
      s_diff_friendly = diffFriendly;
   }

   FILE* CHAICallback::getLogFile() {
      return s_log_file;
   }

   void CHAICallback::setLogFile(const char* fileName) {
      if (s_log_file != stdout) {
         fclose(s_log_file);
      }

      if (fileName == nullptr) {
         s_log_file = stdout;
      }
      else {
         // Create/truncate the output file
         s_log_file = fopen(fileName, "w");
      }
   }

   int CHAICallback::getLogAllocations() {
      return s_log_allocations;
   }

   void CHAICallback::setLogAllocations(int logSetting) {
      if (validLogSetting(logSetting)) {
         s_log_allocations = logSetting;

         if (logSetting > 0) {
            s_active = true;
         }
      }
      else {
         s_log_allocations = 0;
      }
   }

   int CHAICallback::getLogDeallocations() {
      return s_log_deallocations;
   }

   void CHAICallback::setLogDeallocations(int logSetting) {
      if (validLogSetting(logSetting)) {
         s_log_deallocations = logSetting;

         if (logSetting > 0) {
            s_active = true;
         }
      }
      else {
         s_log_deallocations = 0;
      }
   }

   int CHAICallback::getLogMoves() {
      return s_log_moves;
   }

   void CHAICallback::setLogMoves(int logSetting) {
      if (validLogSetting(logSetting)) {
         s_log_moves = logSetting;

         if (logSetting > 0) {
            s_active = true;
         }
      }
      else {
         s_log_moves = 0;
      }
   }

   int CHAICallback::getLogCaptures() {
      return s_log_captures;
   }

   void CHAICallback::setLogCaptures(int logSetting) {
      if (validLogSetting(logSetting)) {
         s_log_captures = logSetting;

         if (logSetting > 0) {
            s_active = true;
         }
      }
      else {
         s_log_captures = 0;
      }
   }

   int CHAICallback::getLogAbandoned() {
      return s_log_abandoned;
   }

   void CHAICallback::setLogAbandoned(int logSetting) {
      if (validLogSetting(logSetting)) {
         s_log_abandoned = logSetting;

         if (logSetting > 0) {
            s_active = true;
         }
      }
      else {
         s_log_abandoned = 0;
      }
   }

   int CHAICallback::getLogLeaks() {
      return s_log_leaks;
   }

   void CHAICallback::setLogLeaks(int logSetting) {
      if (validLogSetting(logSetting)) {
         s_log_leaks = logSetting;

         if (logSetting > 0) {
            s_active = true;
         }
      }
      else {
         s_log_leaks = 0;
      }
   }

   int CHAICallback::getLogData() {
      return s_log_data;
   }

   void CHAICallback::setLogData(int logSetting) {
      if (validLogSetting(logSetting)) {
#if defined(CARE_DEBUG)
         s_log_data = logSetting;

         if (logSetting > 0) {
            s_active = true;
         }
#else
         if (logSetting > 0) {
            fprintf(s_log_file,
                    "[CARE] [CHAI] Logging CHAI data is not supported in this build!\n");
         }
#endif
      }
      else {
         s_log_data = 0;
      }
   }

   const char* CHAICallback::getName(const chai::PointerRecord* record) {
      const char* name = nullptr;

      const NameMap& s_names = getNameMap();
      auto it = s_names.find(record);

      if (it != s_names.end()) {
         name = it->second.c_str();
      }

      return name;
   }

   void CHAICallback::setName(const chai::PointerRecord* record,
                              std::string name) {
      // TODO: Check if it already exists in the map
      NameMap& s_names = getNameMap();
      s_names.emplace(record, name);
   }

   void CHAICallback::setPrintCallback(const chai::PointerRecord* record,
                                       std::function<void(std::ostream&, const void*, size_t)> callback) {
      // TODO: Check if it already exists in the map
      getPrintCallbackMap().emplace(record, callback);
   }

   CHAICallback::CHAICallback(const chai::PointerRecord* record)
      : m_record(record)
   {
   }

   void CHAICallback::operator()(chai::PointerRecord const * record, chai::Action action, chai::ExecutionSpace space) {
      if (s_active) {
         NameMap& s_names = getNameMap();

         if (s_logging_enabled) {
            size_t size = record->m_size;

            std::string name = "UNKNOWN";

            auto it = s_names.find(m_record);

            if (it != s_names.end()) {
               name = it->second;
            }

            switch(action) {
               case chai::ACTION_ALLOC:
                  if (s_log_allocations == (int) space ||
                      s_log_allocations == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
                     if (s_diff_friendly) {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s: Allocated %lu bytes in space %i\n",
                                name.c_str(), size, (int) space);
                     }
                     else {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s (%p): Allocated %lu bytes in space %i (%p)\n",
                                name.c_str(), m_record, size, (int) space,
                                m_record->m_pointers[space]);
                     }
                  }

                  break;
               case chai::ACTION_FREE:
                  if (s_log_deallocations == (int) space ||
                      s_log_deallocations == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
                     if (s_diff_friendly) {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s: Deallocated %lu bytes in space %i\n",
                                name.c_str(), size, (int) space);
                     }
                     else {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s (%p): Deallocated %lu bytes in space %i (%p)\n",
                                name.c_str(), m_record, size, (int) space,
                                m_record->m_pointers[space]);
                     }
                  }

#ifndef CARE_DISABLE_RAJAPLUGIN
                  if (s_log_data > 0) {
                     RAJAPlugin::removeActivePointer(m_record);
                  }
#endif

                  s_names.erase(m_record);
                  getPrintCallbackMap().erase(m_record);

                  break;
               case chai::ACTION_MOVE:
                  if (s_log_moves == (int) space ||
                      s_log_moves == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
#ifndef CARE_DISABLE_RAJAPLUGIN
                     if (s_diff_friendly) {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s: Moved %lu bytes to space %i at %s:%i\n",
                                name.c_str(), size, (int) space,
                                RAJAPlugin::getCurrentLoopFileName().c_str(),
                                RAJAPlugin::getCurrentLoopLineNumber());
                     }
                     else {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s (%p): Moved %lu bytes to space %i (%p) at %s:%i\n",
                                name.c_str(), m_record, size, (int) space,
                                m_record->m_pointers[space],
                                RAJAPlugin::getCurrentLoopFileName().c_str(),
                                RAJAPlugin::getCurrentLoopLineNumber());
                     }
#endif
                  }

                  break;
               case chai::ACTION_CAPTURED:
#ifndef CARE_DISABLE_RAJAPLUGIN
                  if (s_log_captures == (int) space ||
                      s_log_captures == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
                     if (s_diff_friendly) {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s: Captured %lu bytes to space %i at %s:%i\n",
                                name.c_str(), size, (int) space,
                                RAJAPlugin::getCurrentLoopFileName().c_str(),
                                RAJAPlugin::getCurrentLoopLineNumber());
                     }
                     else {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s (%p): Captured %lu bytes to space %i (%p) at %s:%i\n",
                                name.c_str(), m_record, size, (int) space,
                                m_record->m_pointers[space],
                                RAJAPlugin::getCurrentLoopFileName().c_str(),
                                RAJAPlugin::getCurrentLoopLineNumber());
                     }
                  }
                  if (s_log_data > 0) {
                     RAJAPlugin::addActivePointer(m_record);
                  }
#endif
                  break;
               case chai::ACTION_FOUND_ABANDONED:
                  if (s_log_abandoned == (int) space ||
                      s_log_abandoned == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
                     if (s_diff_friendly) {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s: Found %lu bytes abandoned in space %i\n",
                                name.c_str(), size, (int) space);
                     }
                     else {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s (%p): Found %lu bytes abandoned in space %i (%p)\n",
                                name.c_str(), m_record, size, (int) space,
                                m_record->m_pointers[space]);
                     }
                  }

                  break;
               case chai::ACTION_LEAKED:
                  if (s_log_leaks == (int) space ||
                      s_log_leaks == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
                     if (s_diff_friendly) {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s: Leaked %lu bytes in space %i\n",
                                name.c_str(), size, (int) space);
                     }
                     else {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s (%p): Leaked %lu bytes in space %i (%p)\n",
                                name.c_str(), m_record, size, (int) space,
                                m_record->m_pointers[space]);

                     }
                  }

                  break;
            }

            fflush(s_log_file);
         }
         else {
            // Just update or remove the stored data without logging anything
            switch(action) {
               case chai::ACTION_FREE:
#ifndef CARE_DISABLE_RAJAPLUGIN
                  if (s_log_data > 0) {
                     RAJAPlugin::removeActivePointer(m_record);
                  }
#endif
                  s_names.erase(m_record);
                  getPrintCallbackMap().erase(m_record);

                  break;
               case chai::ACTION_CAPTURED:
#ifndef CARE_DISABLE_RAJAPLUGIN
                  if (s_log_data > 0) {
                     RAJAPlugin::addActivePointer(m_record);
                  }
#endif
                  break;
               default:
                  break;
            }
         }
      }
   }

   void CHAICallback::writeArray(const chai::PointerRecord* record,
                                 chai::ExecutionSpace space) {
      if (s_logging_enabled) {
         if (s_log_data == (int) space ||
             s_log_data == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
            // Get the current data
            const size_t size = record->m_size;
            chai::ExecutionSpace lastSpace = record->m_last_space ;
            void* currentData = const_cast<void*>(record->m_pointers[lastSpace]);

            // Get the array manager
            chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();

            // Allocate space for the current data on the host
            umpire::Allocator allocator = arrayManager->getAllocator(chai::CPU);
            void* currentDataCopy = allocator.allocate(size);

            // Copy data to the host
            arrayManager->copy(currentDataCopy, currentData, size);

            // TODO: Investigate deepCopy. Bring to the host and then compare with
            // currentDataCopy, and we might have Ben's stale data check back.

            // This callback converts to the correct type and number of elements
            const PrintCallbackMap& s_callback_map = getPrintCallbackMap();
            auto it = s_callback_map.find(record);

            if (it != s_callback_map.end()) {
               std::stringstream ss;

               // This callback converts to the correct type and number of elements
               it->second(ss, currentDataCopy, size);
               ss << std::endl;

               // Write to the output file
               fprintf(s_log_file, "%s", ss.str().c_str());
               fflush(s_log_file);
            }

            // Free the copy of the data
            allocator.deallocate(currentDataCopy);
         }
      }
   }

   bool CHAICallback::validLogSetting(int logSetting) {
      if (logSetting >= chai::ExecutionSpace::NONE &&
          logSetting <= chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
         return true;
      }
      else {
         fprintf(s_log_file,
                 "[CARE] [CHAI] Log setting (%d) must be in the inclusive range (%d, %d)!\n",
                 logSetting,
                 (int) chai::ExecutionSpace::NONE,
                 (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES);

         return false;
      }
   }
} // namespace care

