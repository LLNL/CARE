#ifndef _CARE_ProfilePlugin_H_
#define _CARE_ProfilePlugin_H_

#include "RAJA/util/PluginStrategy.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "care/config.h"
#include "care/CHAICallback.h"

namespace care{

	class ProfilePlugin : public RAJA::util::PluginStrategy
	{
		public:
			ProfilePlugin();

         static void setFileName(const char * name);

         static void setLineNumber(int num);
						
			void preLaunch(const RAJA::util::PluginContext& p) override;

			void postLaunch(const RAJA::util::PluginContext& p) override;

         static const char * getCurrentLoopFileName();
         
         static int getCurrentLoopLineNumber();

         static void setParallelContext(bool isParallel);

         static bool isParallelContext();

         static void register_post_parallel_forall_action(void * key, std::function<void(chai::ExecutionSpace, const char *, int)> action);

         static bool post_parallel_forall_action_registered(void * key);
         
		private:
         static const char * fileName;

         static int lineNumber;

			static unsigned int s_current_color;

         static uint32_t s_colors[7];

			static int s_num_colors;
         
			static bool s_profile_host_loops;
			
         static bool s_parallel_context;

         static std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>> s_post_parallel_forall_actions;         
	};
}

#endif
