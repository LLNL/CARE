#ifndef _CARE_PluginData_H_
#define _CARE_PluginData_H_

#include "RAJA/util/PluginStrategy.hpp"
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
			PluginData();

         friend class DebugPlugin;

         friend class ProfilePlugin;

         static void setFileName(const char * name);

         static void setLineNumber(int num);		

         static const char * getCurrentLoopFileName();
         
         static int getCurrentLoopLineNumber();

         static void setParallelContext(bool isParallel);

         static bool isParallelContext();

         static void register_post_parallel_forall_action(void * key, std::function<void(chai::ExecutionSpace, const char *, int)> action);

         static bool post_parallel_forall_action_registered(void * key);
         

         static void addActivePointer(const chai::PointerRecord* record);

         static void removeActivePointer(const chai::PointerRecord* record);

         static void setSynchronization(bool synchronizeBefore, bool synchronizeAfter);

         static int s_threadID;

		private:
         static const char * fileName;

         static int lineNumber;

			static std::string s_current_loop_file_name;

         static int s_current_loop_line_number;

			static std::vector<const chai::PointerRecord*> s_active_pointers_in_loop;

         static bool s_synchronize_before;

         static unsigned int s_current_color;

         static uint32_t s_colors[7];

			static int s_num_colors;
         
			static bool s_profile_host_loops;
			
         static bool s_parallel_context;

         static std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>> s_post_parallel_forall_actions; 
         
         static bool s_synchronize_after;
	};
}

#endif
