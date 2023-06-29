#ifndef _CARE_DebugPlugin_H_
#define _CARE_DebugPlugin_H_

#include "RAJA/util/PluginStrategy.hpp"
#include "chai/ExecutionSpaces.hpp"
#include "care/config.h"
#include "care/CHAICallback.h"
#include <vector>

namespace chai {
   struct PointerRecord;
}

namespace care{

	class DebugPlugin : public RAJA::util::PluginStrategy
	{
		public:
			DebugPlugin();
						
			void preLaunch(const RAJA::util::PluginContext& p) override;

			void postLaunch(const RAJA::util::PluginContext& p) override;

			static void writeLoopData(chai::ExecutionSpace space, const char * fileName, int lineNumber);
	};
}

#endif
