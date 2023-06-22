#include "care/ProfilePlugin.h"

#include "chai/ArrayManager.hpp"

#include "chai/ExecutionSpaces.hpp"

#include "care/CHAICallback.h"

#include "care/RAJAPlugin.h"

/* CUDA profiling macros */
#if defined(__CUDACC__) && CARE_HAVE_NVTOOLSEXT
#include "nvToolsExt.h"
#endif

// Std library headers
#if defined(CARE_DEBUG) && !defined(_WIN32)
#include <execinfo.h>
#endif // defined(CARE_DEBUG) && !defined(_WIN32)

#include <unordered_set>

using namespace care;

namespace chai{

	bool s_profile_host_loops = true;
	static uint32_t s_colors[7];
   static int s_num_colors;
   static unsigned int s_current_color;

ProfilePlugin::ProfilePlugin() {}

   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @brief Set up to be done before executing a RAJA loop.
   ///
   /// @arg[in] space The execution space
   /// @arg[in] name The name of the loop
   ///
   /////////////////////////////////////////////////////////////////////////////////

void 
ProfilePlugin::preLaunch(const RAJA::util::PluginContext& p) {
#if CARE_HAVE_NVTOOLSEXT
#if defined(CARE_GPUCC)
      // Profile the host loops
      if (s_profile_host_loops) {
         if (p.platform == RAJA::Platform::host) {
            std::string name = fileName + std::to_string(lineNumber);

            int color_id = s_current_color + 1;
            color_id = color_id % s_num_colors;

            // TODO: Add error checking
            nvtxEventAttributes_t eventAttrib = { 0 };
            eventAttrib.version = NVTX_VERSION;
            eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
            eventAttrib.colorType = NVTX_COLOR_ARGB;
            eventAttrib.color = s_colors[color_id];
            eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
            eventAttrib.message.ascii = name.c_str();
            nvtxRangePushEx(&eventAttrib);
         }
      }
#endif // CARE_HAVE_NVTOOLSEXT
#endif // defined(CARE_GPUCC)
   }

void 
ProfilePlugin::postLaunch(const RAJA::util::PluginContext& p) {
#if defined(CARE_GPUCC)
      if (s_profile_host_loops) {
         if (p.platform == RAJA::Platform::host) {
            // TODO: Add error checking
            nvtxRangePop();
         }
      }
#endif

}
}

static RAJA::util::PluginRegistry::add<chai::ProfilePlugin> L ("Care profile plugin", "test");


