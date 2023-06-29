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
						
			void preLaunch(const RAJA::util::PluginContext& p) override;

			void postLaunch(const RAJA::util::PluginContext& p) override;        
	};
}

#endif
