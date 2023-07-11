#ifndef _CARE_PluginData_H_
#define _CARE_PluginData_H_

#include "chai/ExecutionSpaces.hpp"
#include "care/config.h"
#include "care/CHAICallback.h"
#include <vector>

namespace chai {
   struct PointerRecord;
}

namespace care{

	class PluginData	{
		public:
			PluginData() = default;

         static void setFileName(const char * name);

         static void setLineNumber(int num);	

         static const char * getFileName();
         
         static int getLineNumber();   

         static void setParallelContext(bool isParallel);

         static bool isParallelContext();

         static bool post_parallel_forall_action_registered(void * key); 

         static void register_post_parallel_forall_action(void * key, std::function<void(chai::ExecutionSpace, const char *, int)> action);

         static std::vector<const chai::PointerRecord*> getActivePointers();

         static void addActivePointer(const chai::PointerRecord* record);

         static void removeActivePointer(const chai::PointerRecord* record);

         static void clearActivePointers();

         static int s_threadID;


		private:
         static const char * s_file_name;

         static int s_line_number;

         static bool s_parallel_context;

         static std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>> s_post_parallel_forall_actions;

         static std::vector<const chai::PointerRecord*> s_active_pointers_in_loop;
	};
}

#endif
