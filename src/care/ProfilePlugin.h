#ifndef _CARE_ProfilePlugin_H_
#define _CARE_ProfilePlugin_H_

#include "RAJA/util/PluginStrategy.hpp"

namespace care{

	class ProfilePlugin : public RAJA::util::PluginStrategy
	{
		public:
			ProfilePlugin();
						
			void preLaunch(const RAJA::util::PluginContext& p) override;

			void postLaunch(const RAJA::util::PluginContext& p) override;

      private:
         static unsigned int s_current_color;

         static uint32_t s_colors[7];

			static int s_num_colors;
         
			static bool s_profile_host_loops;
			
         
                  
	};
}

#endif
