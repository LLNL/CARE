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

         static void setFileName(const char * name);

         static void setLineNumber(int num);
						
			void preLaunch(const RAJA::util::PluginContext& p) override;

			void postLaunch(const RAJA::util::PluginContext& p) override;

			static void writeLoopData(chai::ExecutionSpace space, const char * fileName, int lineNumber);

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
         
         static bool s_synchronize_after;
	};
}

#endif
