//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE headers
#include "care/CHAICallback.h"
#ifndef CARE_DISABLE_RAJAPLUGIN
#include "care/PluginData.h"
#endif

// Other library headers
#include "chai/ArrayManager.hpp"
#include "chai/PointerRecord.hpp"

// Std library headers
#include <cstddef>
#include <sstream>
#include <typeinfo>
#include <vector>

namespace care {
   namespace detail {
      template <class T>
      class DefaultPrintCallback {
         public:
            std::string operator()(const void* value) {
               const T* typedValue = static_cast<const T*>(value);
               return std::to_string(*typedValue);
            }
      };
   } // namespace detail

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

   CHAICallback::TypeMap& CHAICallback::getTypeMap() {
      // Using this approach rather than a static member to avoid a
      // global initialization order bug.
      static TypeMap s_types;
      return s_types;
   }

   CHAICallback::PrintCallbackMap& CHAICallback::getPrintCallbackMap() {
      // Using this approach rather than a static member to avoid a
      // global initialization order bug.

      // Add in default print callbacks
      // If this map is updated, be sure to update the TypeSizeMap below
      static PrintCallbackMap s_print_callbacks = {
         {typeid(short), detail::DefaultPrintCallback<short>()},
         {typeid(unsigned short), detail::DefaultPrintCallback<unsigned short>()},
         {typeid(int), detail::DefaultPrintCallback<int>()},
         {typeid(unsigned int), detail::DefaultPrintCallback<unsigned int>()},
         {typeid(long), detail::DefaultPrintCallback<long>()},
         {typeid(unsigned long), detail::DefaultPrintCallback<unsigned long>()},
         {typeid(long long), detail::DefaultPrintCallback<long long>()},
         {typeid(unsigned long long), detail::DefaultPrintCallback<unsigned long long>()},
         {typeid(float), detail::DefaultPrintCallback<float>()},
         {typeid(double), detail::DefaultPrintCallback<double>()},
         {typeid(long double), detail::DefaultPrintCallback<long double>()},
         {typeid(bool), detail::DefaultPrintCallback<bool>()}
      };

      // Return the callback map
      return s_print_callbacks;
   }

   CHAICallback::TypeSizeMap& CHAICallback::getTypeSizeMap() {
      // Using this approach rather than a static member to avoid a
      // global initialization order bug.

      // Add in default type sizes
      // If this map is updated, be sure to update the PrintCallbackMap above
      static TypeSizeMap s_type_sizes = {
         {typeid(short), sizeof(short)},
         {typeid(unsigned short), sizeof(unsigned short)},
         {typeid(int), sizeof(int)},
         {typeid(unsigned int), sizeof(unsigned int)},
         {typeid(long), sizeof(long)},
         {typeid(unsigned long), sizeof(unsigned long)},
         {typeid(long long), sizeof(long long)},
         {typeid(unsigned long long), sizeof(unsigned long long)},
         {typeid(float), sizeof(float)},
         {typeid(double), sizeof(double)},
         {typeid(long double), sizeof(long double)},
         {typeid(bool), sizeof(bool)}
      };

      // Return the callback map
      return s_type_sizes;
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

   void CHAICallback::deregisterRecord(const chai::PointerRecord* record) {
#ifndef CARE_DISABLE_RAJAPLUGIN
      if (s_log_data > 0) {
         PluginData::removeActivePointer(record);
      }
#endif
      NameMap& s_names = getNameMap();
      TypeMap& s_types = getTypeMap();

      s_names.erase(record);
      s_types.erase(record);
   }

   bool CHAICallback::hasTypeIndex(const chai::PointerRecord* record) {
      const TypeMap& s_types = getTypeMap();
      auto it = s_types.find(record);
      return it != s_types.end();
   }

   std::type_index CHAICallback::getTypeIndex(const chai::PointerRecord* record) {
      const TypeMap& s_types = getTypeMap();
      auto it = s_types.find(record);
      return it->second;
   }

   void CHAICallback::setTypeIndex(const chai::PointerRecord* record,
                                   std::type_index typeIndex) {
      // TODO: Check if it already exists in the map
      TypeMap& s_types = getTypeMap();
      s_types.emplace(record, typeIndex);
   }

   bool CHAICallback::hasPrintCallback(std::type_index typeIndex) {
      const PrintCallbackMap& s_print_callbacks = getPrintCallbackMap();
      auto it = s_print_callbacks.find(typeIndex);
      return it != s_print_callbacks.end();
   }

   CHAICallback::PrintCallback CHAICallback::getPrintCallback(std::type_index typeIndex) {
      const PrintCallbackMap& s_print_callbacks = getPrintCallbackMap();
      auto it = s_print_callbacks.find(typeIndex);
      return it->second;
   }

   void CHAICallback::setPrintCallback(std::type_index typeIndex,
                                       PrintCallback callback) {
      // TODO: Check if it already exists in the map
      PrintCallbackMap& s_print_callbacks = getPrintCallbackMap();
      s_print_callbacks.emplace(typeIndex, callback);
   }

   bool CHAICallback::hasTypeSize(std::type_index typeIndex) {
      const TypeSizeMap& s_type_sizes = getTypeSizeMap();
      auto it = s_type_sizes.find(typeIndex);
      return it != s_type_sizes.end();
   }

   size_t CHAICallback::getTypeSize(std::type_index typeIndex) {
      const TypeSizeMap& s_type_sizes = getTypeSizeMap();
      auto it = s_type_sizes.find(typeIndex);
      return it->second;
   }

   void CHAICallback::setTypeSize(std::type_index typeIndex,
                                  size_t size) {
      // TODO: Check if it already exists in the map
      TypeSizeMap& s_type_sizes = getTypeSizeMap();
      s_type_sizes.emplace(typeIndex, size);
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

                  deregisterRecord(m_record);

                  break;
               case chai::ACTION_MOVE:
                  if (s_log_moves == (int) space ||
                      s_log_moves == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
#ifndef CARE_DISABLE_RAJAPLUGIN
                     if (s_diff_friendly) {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s: Moved %lu bytes to space %i at %s:%i\n",
                                name.c_str(), size, (int) space,
                                PluginData::getFileName(),
                                PluginData::getLineNumber());
                     }
                     else {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s (%p): Moved %lu bytes to space %i (%p) at %s:%i\n",
                                name.c_str(), m_record, size, (int) space,
                                m_record->m_pointers[space],
                                PluginData::getFileName(),
                                PluginData::getLineNumber());
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
                                PluginData::getFileName(),
                                PluginData::getLineNumber());
                     }
                     else {
                        fprintf(s_log_file,
                                "[CARE] [CHAI] %s (%p): Captured %lu bytes to space %i (%p) at %s:%i\n",
                                name.c_str(), m_record, size, (int) space,
                                m_record->m_pointers[space],
                                PluginData::getFileName(),
                                PluginData::getLineNumber());
                     }
                  }
                  if (s_log_data > 0) {
                     PluginData::addActivePointer(m_record);
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
                  deregisterRecord(m_record);

                  break;
               case chai::ACTION_CAPTURED:
#ifndef CARE_DISABLE_RAJAPLUGIN
                  if (s_log_data > 0) {
                     PluginData::addActivePointer(m_record);
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
      // Check if logging is enabled and requested
      if (s_logging_enabled) {
         if (s_log_data == (int) space ||
             s_log_data == (int) chai::ExecutionSpace::NUM_EXECUTION_SPACES) {
            std::stringstream ss;

            // Print out the name, if it has one
            const char* name = getName(record);

            if (name) {
               ss << "[CARE] Name: " << std::string(name) << std::endl;
            }
            else {
               ss << "[CARE] Name: UNKNOWN" << std::endl;
            }

            // We can always print out the size, even if we don't know the type
            const size_t size = record->m_size;

            // Check if we know the type
            if (hasTypeIndex(record)) {
               std::type_index typeIndex = getTypeIndex(record);

               // Check if we know the type size
               if (hasTypeSize(typeIndex)) {
                  size_t typeSize = getTypeSize(typeIndex);

                  // Check if we know how to print the type
                  if (hasPrintCallback(typeIndex)) {
                     PrintCallback printCallback = getPrintCallback(typeIndex);

                     // Check if the callback actually exists
                     if (printCallback) {
                        // Get the current data
                        chai::ExecutionSpace lastSpace = record->m_last_space;
                        void* currentData = const_cast<void*>(record->m_pointers[lastSpace]);

                        // Get the array manager
                        chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();

                        // Allocate space for the current data on the host
                        umpire::Allocator allocator = arrayManager->getAllocator(chai::CPU);
                        void* currentDataCopy = allocator.allocate(size);

                        // Copy data to the host
                        arrayManager->copy(currentDataCopy, currentData, size);

                        // TODO: Investigate deepCopy. Bring to the host and then compare
                        // with currentDataCopy, and we might have Ben's stale data check
                        // back.

                        // Calculate the number of typed elements
                        const int count = size / typeSize;

                        // Write out count
                        ss << "[CARE] Size: " << count << std::endl;

                        // Cast to a bytes array
                        const char* bytesArray = static_cast<const char*>(currentDataCopy);

                        // Used for alignment
                        const int maxDigits = std::to_string(count).size();

                        // Write out elements
                        for (size_t i = 0; i < size; i += typeSize) {
                           const int digits = std::to_string(i).size();
                           const int numSpaces = maxDigits - digits;

                           // Write out index
                           ss << "[CARE] Index: ";

                           // Take care of alignment
                           for (int j = 0; j < numSpaces; ++j) {
                              ss << " ";
                           }

                           ss << i;

                           // Write out white space between index and value
                           ss << "    ";

                           // Write out value
                           ss << "Value: " << printCallback(&bytesArray[i]) << std::endl;
                        }

                        ss << std::endl;

                        // Free the copy of the data
                        allocator.deallocate(currentDataCopy);
                     }
                     else {
                        // Write out size
                        ss << "[CARE] Size: " << size << " bytes" << std::endl;

                        // TODO: Decide if we should write out the bytes
                        ss << "[CARE] Print callback invalid for type " << typeIndex.name() << "! Not printing out contents." << std::endl;
                     }
                  }
                  else {
                     // Write out size
                     ss << "[CARE] Size: " << size << " bytes" << std::endl;

                     // TODO: Decide if we should write out the bytes
                     ss << "[CARE] No print callback for type " << typeIndex.name() << "! Not printing out contents." << std::endl;
                  }
               }
               else {
                  // Write out size
                  ss << "[CARE] Size: " << size << " bytes" << std::endl;

                  // TODO: Decide if we should write out the bytes
                  ss << "[CARE] Unknown type size! Not printing out contents." << std::endl;
               }
            }
            else {
               // Write out size
               ss << "[CARE] Size: " << size << " bytes" << std::endl;

               // TODO: Decide if we should write out the bytes
               ss << "[CARE] Unknown type! Not printing out contents." << std::endl;
            }

            // Write to the output file
            fprintf(s_log_file, "%s", ss.str().c_str());
            fflush(s_log_file);
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

