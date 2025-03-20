//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_CHAI_CALLBACK_H_
#define _CARE_CHAI_CALLBACK_H_

// CARE config header
#include "care/config.h"

// CHAI headers
// TODO: Forward declarations would be sufficient if the enums were typed
#include "chai/ExecutionSpaces.hpp"
#include "chai/Types.hpp"

// Std library headers
#include <functional>
#include <string>
#include <typeindex>
#include <unordered_map>

// Forward declarations
namespace chai {
   struct PointerRecord;
}

namespace care {
   class CHAICallback {
      public:
         using PrintCallback = std::function<std::string(const void*)>;
         using NameMap = std::unordered_map<const chai::PointerRecord*, std::string>;
         using TypeMap = std::unordered_map<const chai::PointerRecord*, std::type_index>;
         using PrintCallbackMap = std::unordered_map<std::type_index, PrintCallback>;
         using TypeSizeMap = std::unordered_map<std::type_index, size_t>;

         ///
         /// Whether or not this infrastructure is turned on, meaning that CHAI
         /// ManagedArrays will have names and callbacks associated with them
         /// among other things. This does result in significant overhead, so
         /// it is only enabled if the log setting for any event/data is greater
         /// than 0.
         ///
         /// @note Logging can be temporarily enabled/disabled by calling
         ///       enableLogging() or disableLogging(). However, the infrastructure
         ///       will remain on (if it did not, callbacks and names would not get
         ///       registered and therefore would not be available when you actually
         ///       wanted to log data).
         ///
         /// @return true if at any point logging is requested for CHAI events
         ///         and/or data. Otherwise returns false.
         ///
         CARE_DLL_API static bool isActive();

         ///
         /// Temporarily enables logging of CHAI events and data.
         ///
         CARE_DLL_API static void enableLogging();

         ///
         /// Temporarily disables logging of CHAI events and data.
         ///
         CARE_DLL_API static void disableLogging();

         ///
         /// Checks if logging of CHAI events and data is enabled.
         ///
         /// @return true if logging is enabled, false otherwise
         ///
         CARE_DLL_API static bool loggingIsEnabled();

         ///
         /// Checks if logging of CHAI events and data is diff-friendly.
         ///
         /// @return true if logging is diff-friendly, false otherwise
         ///
         CARE_DLL_API static bool isDiffFriendly();

         ///
         /// Sets whether or not logging of CHAI events and data
         ///        should be diff-friendly.
         ///
         /// @param[in] diffFriendly Whether or not logging should be diff-friendly
         ///
         CARE_DLL_API static void setDiffFriendly(bool diffFriendly);

         ///
         /// Returns the log file being used.
         ///
         /// @return The log file being used
         ///
         CARE_DLL_API static FILE* getLogFile();

         ///
         /// Sets the log file to use.
         ///
         /// @param[in] fileName The log file to used
         ///
         CARE_DLL_API static void setLogFile(const char* fileName);

         ///
         /// Gets the log setting for CHAI memory allocations.
         ///
         /// @return The log setting for CHAI memory allocations
         ///
         CARE_DLL_API static int getLogAllocations();

         ///
         /// Sets the log setting for CHAI memory allocations.
         ///
         /// @param[in] logSetting The log setting for CHAI memory allocations
         ///
         CARE_DLL_API static void setLogAllocations(int logSetting);

         ///
         /// Gets the log setting for CHAI memory deallocations.
         ///
         /// @return The log setting for CHAI memory deallocations
         ///
         CARE_DLL_API static int getLogDeallocations();

         ///
         /// Sets the log setting for CHAI memory deallocations.
         ///
         /// @param[in] logSetting The log setting for CHAI memory deallocations
         ///
         CARE_DLL_API static void setLogDeallocations(int logSetting);

         ///
         /// Gets the log setting for CHAI memory motion.
         ///
         /// @return The log setting for CHAI memory motion
         ///
         CARE_DLL_API static int getLogMoves();

         ///
         /// Sets the log setting for CHAI memory motion.
         ///
         /// @param[in] logSetting The log setting for CHAI memory motion
         ///
         CARE_DLL_API static void setLogMoves(int logSetting);

         ///
         /// Gets the log setting for CHAI captures.
         ///
         /// @return The log setting for CHAI captures
         ///
         CARE_DLL_API static int getLogCaptures();

         ///
         /// Sets the log setting for CHAI captures.
         ///
         /// @param[in] logSetting The log setting for CHAI captures
         ///
         CARE_DLL_API static void setLogCaptures(int logSetting);

         ///
         /// Gets the log setting for CHAI abandoned pointer records.
         ///
         /// @return The log setting for CHAI abandoned pointer records
         ///
         CARE_DLL_API static int getLogAbandoned();

         ///
         /// Sets the log setting for CHAI abandoned pointer records.
         ///
         /// @param[in] logSetting The log setting for CHAI abandoned pointer records
         ///
         CARE_DLL_API static void setLogAbandoned(int logSetting);

         ///
         /// Gets the log setting for CHAI memory leaks.
         ///
         /// @return The log setting for CHAI memory leaks
         ///
         CARE_DLL_API static int getLogLeaks();

         ///
         /// Sets the log setting for CHAI memory leaks.
         ///
         /// @param[in] logSetting The log setting for CHAI memory leaks
         ///
         CARE_DLL_API static void setLogLeaks(int logSetting);

         ///
         /// Gets the log setting for CHAI arrays.
         ///
         /// @return The log setting for CHAI arrays
         ///
         CARE_DLL_API static int getLogData();

         ///
         /// Sets the log setting for CHAI arrays.
         ///
         /// @param[in] logSetting The log setting for CHAI arrays
         ///
         /// @note Only works if built with -DCARE_DEBUG=ON
         ///
         CARE_DLL_API static void setLogData(int logSetting);

         ///
         /// Gets the name associated with the given pointer record
         ///
         /// @param[in] record The record used to look up the name
         ///
         /// @return The name associated with the given pointer record if there is one,
         ///         nullptr otherwise.
         ///
         CARE_DLL_API static const char* getName(const chai::PointerRecord* record);

         ///
         /// Sets the name associated with the given pointer record
         ///
         /// @param[in] record The record to associate the name with
         /// @param[in] name The name of the record
         ///
         CARE_DLL_API static void setName(const chai::PointerRecord* record,
                                          std::string name);

         ///
         /// Deregister the given pointer record
         ///
         /// @param[in] record The record to deregister
         ///
         CARE_DLL_API static void deregisterRecord(const chai::PointerRecord* record);

         ///
         /// Returns true or false depending on whether there is a type index
         /// associated with the given pointer record.
         ///
         /// @param[in] record The record used to look up whether there is an
         ///                   associated type index
         ///
         /// @return True if there is a type index associated with the given
         ///         pointer record, false otherwise.
         ///
         CARE_DLL_API static bool hasTypeIndex(const chai::PointerRecord* record);

         ///
         /// Gets the type index associated with the given pointer record
         ///
         /// @param[in] record The record used to look up the type index
         ///
         /// @return The type index associated with the given pointer record
         ///         if there is one.
         ///
         CARE_DLL_API static std::type_index getTypeIndex(const chai::PointerRecord* record);

         ///
         /// Sets the type index associated with the given pointer record
         ///
         /// @param[in] record The record to associate the type index with
         /// @param[in] typeIndex The type index of the record
         ///
         CARE_DLL_API static void setTypeIndex(const chai::PointerRecord* record,
                                               std::type_index typeIndex);

         ///
         /// Returns true or false depending on whether there is a print callback
         /// associated with the given type index.
         ///
         /// @param[in] typeIndex The type index used to look up whether there is an
         ///                      associated print callback
         ///
         /// @return True if there is a print callback associated with the given
         ///         type index, false otherwise.
         ///
         CARE_DLL_API static bool hasPrintCallback(std::type_index typeIndex);

         ///
         /// Gets the print callback associated with the given type index.
         ///
         /// @param[in] typeIndex The type index used to look up the print callback
         ///
         /// @return The print callback associated with the given type index
         ///         if there is one.
         ///
         CARE_DLL_API static PrintCallback getPrintCallback(std::type_index typeIndex);

         ///
         /// Sets the print callback associated with the given type index.
         ///
         /// @param[in] typeIndex The type index to associate the callback with
         /// @param[in] name The callback to tie the type index to
         ///
         CARE_DLL_API static void setPrintCallback(std::type_index typeIndex,
                                                   PrintCallback callback);

         ///
         /// Returns true or false depending on whether there is a size
         /// associated with the given type index.
         ///
         /// @param[in] typeIndex The type index used to look up whether
         ///                      there is an associated size
         ///
         /// @return True if there is a size associated with the given
         ///         type index, false otherwise.
         ///
         CARE_DLL_API static bool hasTypeSize(std::type_index typeIndex);

         ///
         /// Gets the size associated with the given type index.
         ///
         /// @param[in] typeIndex The type index used to look up the size
         ///
         /// @return The size associated with the given type index
         ///         if there is one.
         ///
         CARE_DLL_API static size_t getTypeSize(std::type_index typeIndex);

         ///
         /// Sets the size associated with the given pointer record
         ///
         /// @param[in] typeIndex The type to associate the size with
         /// @param[in] size The size of the given type
         ///
         CARE_DLL_API static void setTypeSize(std::type_index typeIndex,
                                              size_t size);

         ///
         /// Writes out the data in the given execution space
         ///
         /// @param[in] record The record associated with the data to print
         /// @param[in] space The execution space in which to print out data
         ///
         CARE_DLL_API static void writeArray(const chai::PointerRecord* record,
                                             chai::ExecutionSpace space);

         ///
         /// Constructor
         ///
         /// @param[in] record The record associated with this callback
         ///
         CARE_DLL_API CHAICallback(const chai::PointerRecord* record);

         ///
         /// The callback registered with CHAI
         ///
         /// @param[in] record The pointer record
         /// @param[in] action The CHAI event that was triggered
         /// @param[in] space The execution space that the CHAI event was triggered in
         ///
         CARE_DLL_API void operator()(const chai::PointerRecord* record,
                                      chai::Action action,
                                      chai::ExecutionSpace space);

      private:
         ///
         /// Gets the map of pointer records to names
         ///
         /// @return The map of pointer records to names
         ///
         static NameMap& getNameMap();

         ///
         /// Gets the map of pointer records to type indices
         ///
         /// @return The map of pointer records to type indices
         ///
         static TypeMap& getTypeMap();

         ///
         /// Gets the map of types to print callbacks
         ///
         /// @return The map of types to print callbacks
         ///
         static PrintCallbackMap& getPrintCallbackMap();

         ///
         /// Gets the map of types to type sizes
         ///
         /// @return The map of types to type sizes
         ///
         static TypeSizeMap& getTypeSizeMap();

         ///
         /// Checks if the log setting is valid.
         ///
         /// @param[in] logSetting The log setting to check
         ///
         /// @return true if the log setting is valid, false otherwise
         ///
         static bool validLogSetting(int logSetting);

         ///
         /// Whether or not this infrastructure is active
         ///
         static bool s_active;

         ///
         /// Whether or not logging is currently enabled
         ///
         static bool s_logging_enabled;

         ///
         /// Whether or not logging is diff-friendly (i.e. no pointer addresses)
         ///
         static bool s_diff_friendly;

         ///
         /// The log file for CHAI events and/or data
         ///
         static FILE* s_log_file;

         ///
         /// The log setting for CHAI memory allocations
         ///
         static int s_log_allocations;

         ///
         /// The log setting for CHAI memory deallocations
         ///
         static int s_log_deallocations;

         ///
         /// The log setting for CHAI memory motion
         ///
         static int s_log_moves;

         ///
         /// The log setting for CHAI captures
         ///
         static int s_log_captures;

         ///
         /// The log setting for CHAI abandoned pointer records
         ///
         static int s_log_abandoned;

         ///
         /// The log setting for CHAI memory leaks
         ///
         static int s_log_leaks;

         ///
         /// The log setting for CHAI arrays
         ///
         static int s_log_data;

         ///
         /// The pointer record associated with this callback
         ///
         const chai::PointerRecord* m_record = nullptr;
   }; // class CHAICallback
} // namespace care

#endif // !defined(_CARE_CHAI_CALLBACK_H_)

