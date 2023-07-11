#include "care/ProfilePlugin.h"
#include "care/PluginData.h"

/* CUDA profiling macros */
#if defined(__CUDACC__) && CARE_HAVE_NVTOOLSEXT
#include "nvToolsExt.h"
#endif

namespace care{
   unsigned int ProfilePlugin::s_current_color = 0;
   uint32_t ProfilePlugin::s_colors[7] = { 0x0000ff00,   //green
                                           0x000000ff,   //blue
                                           0x00ffff00,   //yellow
                                           0x00ff00ff,   //pink
                                           0x0000ffff,   //cyan
                                           0x00ff0000,   //red
                                           0x00ffffff }; //white
   int ProfilePlugin::s_num_colors = sizeof(s_colors) / sizeof(uint32_t);
   bool ProfilePlugin::s_profile_host_loops = true; 
   
   void ProfilePlugin::registerPlugin() {
#if defined(__HIPCC_)
      printf("[CARE] Warning: Profiling is only supported in CUDA builds.\n");
#endif
      static RAJA::util::PluginRegistry::add<care::ProfilePlugin> L ("Profile plugin", "CARE plugin for profiling");
   }

   void ProfilePlugin::preLaunch(const RAJA::util::PluginContext& p) {
#if defined(__CUDACC__) && CARE_HAVE_NVTOOLSEXT
      // Profile the host loops
      if (s_profile_host_loops) {
         if (p.platform == RAJA::Platform::host) {
            std::string name = PluginData::getFileName() + std::to_string(PluginData::getLineNumber());
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
#endif // defined(__CUDACC__) && CARE_HAVE_NVTOOLSEXT
   }


   void ProfilePlugin::postLaunch(const RAJA::util::PluginContext& p) {
#if defined(__CUDACC__) && CARE_HAVE_NVTOOLSEXT
      if (s_profile_host_loops) {
         if (p.platform == RAJA::Platform::host) {
            // TODO: Add error checking
            nvtxRangePop();
         }
      }
#endif

   }
}
