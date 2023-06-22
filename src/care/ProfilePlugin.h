#ifndef _CARE_ProfilePlugin_H_
#define _CARE_ProfilePlugin_H_

#include "RAJA/util/PluginStrategy.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "care/config.h"
#include "care/CHAICallback.h"
#include "care/RAJAPlugin.h"

using namespace care;

namespace care{

class RAJAPlugin;

}

namespace chai{

	class ProfilePlugin : public RAJA::util::PluginStrategy
	{
		public:
			ProfilePlugin();

			char * fileName;

			int lineNumber;
						
			void preLaunch(const RAJA::util::PluginContext& p) override;

			void postLaunch(const RAJA::util::PluginContext& p) override;

		private:

			bool s_profile_host_loops;

			unsigned int s_current_color;

			int s_num_colors;

			uint32_t s_colors[7];

	};
}

#endif
