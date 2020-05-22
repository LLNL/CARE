//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_RAJA_PLUGIN_H_
#define _CARE_RAJA_PLUGIN_H_

// CARE config header
#include "care/config.h"

// Other library headers
#include "chai/ExecutionSpaces.hpp"

// Std library headers
#include <string>
#include <vector>

// Forward declarations
namespace chai {
   struct PointerRecord;
}

namespace care {
   class RAJAPlugin {
      public:
         CARE_DLL_API static void pre_forall_hook(chai::ExecutionSpace space,
                                                  const char * fileName,
                                                  int lineNumber);

         CARE_DLL_API static void post_forall_hook(chai::ExecutionSpace space,
                                                   const char * fileName,
                                                   int lineNumber);

         CARE_DLL_API static std::string getCurrentLoopFileName();

         CARE_DLL_API static int getCurrentLoopLineNumber();

         CARE_DLL_API static void addActivePointer(const chai::PointerRecord* record);

         CARE_DLL_API static void removeActivePointer(const chai::PointerRecord* record);

         CARE_DLL_API static void setSynchronization(bool synchronizeBefore,
                                                     bool synchronizeAfter);

      private:
         static void writeLoopData(chai::ExecutionSpace space,
                                   const char * fileName,
                                   int lineNumber);

         static bool s_update_chai_execution_space;
         static bool s_debug_chai_data;
         static bool s_profile_host_loops;
         static bool s_synchronize_before;
         static bool s_synchronize_after;

         static uint32_t s_colors[7];
         static int s_num_colors;
         static unsigned int s_current_color;

         static std::string s_current_loop_file_name;
         static int s_current_loop_line_number;

         static std::vector<const chai::PointerRecord*> s_active_pointers_in_loop;
   }; // class RAJAPlugin
} // namespace care

#endif // !defined(_CARE_RAJA_PLUGIN_H_)

