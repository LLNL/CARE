//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_PluginData_H_
#define _CARE_PluginData_H_

#include "chai/PointerRecord.hpp"
#include "care/config.h"
#include <vector>
#include <functional>
#include <unordered_map>

namespace chai {
   struct PointerRecord;
}

namespace care{
   using ActionMap = std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>>;
   
   //class for shared plugin functions and variables
   class CARE_DLL_API PluginData	{
      public:
         PluginData() = default;

         static void setFileName(const char * name);

         static void setLineNumber(int num);	

         static const char * getFileName();
         
         static int getLineNumber();   

         static void setParallelContext(bool isParallel);

         static bool isParallelContext();

         static bool post_parallel_forall_action_registered(void * key); 

         static ActionMap get_post_parallel_forall_actions();

         static void register_post_parallel_forall_action(void * key, std::function<void(chai::ExecutionSpace, const char *, int)> action);

         static void clear_post_parallel_forall_actions();

         static std::vector<const chai::PointerRecord*> getActivePointers();

         static void addActivePointer(const chai::PointerRecord* record);

         static void removeActivePointer(const chai::PointerRecord* record);

         static void clearActivePointers();

      private:
         static const char * s_file_name;

         static int s_line_number;

         static bool s_parallel_context;

         static ActionMap s_post_parallel_forall_actions;

         static std::vector<const chai::PointerRecord*> s_active_pointers_in_loop;
   };
}

#endif
