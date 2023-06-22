#ifndef _CARE_DebugPlugin_H_
#define _CARE_DebugPlugin_H_

#include "RAJA/util/PluginStrategy.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "care/config.h"
#include "care/CHAICallback.h"
#include "care/RAJAPlugin.h"

namespace care{

	class DebugPlugin : public RAJA::util::PluginStrategy
	{
		public:
			DebugPlugin();
						
			void preLaunch(const RAJA::util::PluginContext& p) override;

			void postLaunch(const RAJA::util::PluginContext& p) override;

			void writeLoopData(chai::ExecutionSpace space, const char * fileName, int lineNumber);

		private:

			std::string s_current_loop_file_name;

         int s_current_loop_line_number;

			std::vector<const chai::PointerRecord*> s_active_pointers_in_loop;
	};
}

#endif
