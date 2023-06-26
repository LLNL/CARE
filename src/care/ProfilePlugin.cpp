#include "care/ProfilePlugin.h"

#include "chai/ArrayManager.hpp"

#include "chai/ExecutionSpaces.hpp"

#include "care/CHAICallback.h"

/* CUDA profiling macros */
#if defined(__CUDACC__) && CARE_HAVE_NVTOOLSEXT
#include "nvToolsExt.h"
#endif

namespace care{
   const char * ProfilePlugin::fileName = "N/A";
   int ProfilePlugin::lineNumber = -1;
   unsigned int ProfilePlugin::s_current_color = 0;
   uint32_t ProfilePlugin::s_colors[7] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
   int ProfilePlugin::s_num_colors = sizeof(s_colors) / sizeof(uint32_t);
   bool ProfilePlugin::s_profile_host_loops = true;
   bool ProfilePlugin::s_parallel_context = false;
   std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>> ProfilePlugin::s_post_parallel_forall_actions 
      = std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>>{};      

   ProfilePlugin::ProfilePlugin() {}
   
   void ProfilePlugin::setFileName(const char * name) {ProfilePlugin::fileName = name;}

   
   void ProfilePlugin::setLineNumber(int num) {ProfilePlugin::lineNumber = num;}


   void ProfilePlugin::preLaunch(const RAJA::util::PluginContext& p) {
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


   void ProfilePlugin::postLaunch(const RAJA::util::PluginContext& p) {
#if defined(CARE_GPUCC)
      if (s_profile_host_loops) {
         if (p.platform == RAJA::Platform::host) {
            // TODO: Add error checking
            nvtxRangePop();
         }
      }
#endif

   }

   const char * ProfilePlugin::getCurrentLoopFileName() {
      return fileName;
   }

   int ProfilePlugin::getCurrentLoopLineNumber() {
      return lineNumber;
   }

   void ProfilePlugin::setParallelContext(bool isParallel) {
      s_parallel_context = isParallel;
   }

   bool ProfilePlugin::isParallelContext(){
      return s_parallel_context;
   }

   void ProfilePlugin::register_post_parallel_forall_action(void * key, std::function<void(chai::ExecutionSpace, const char *, int)> action) { 
      s_post_parallel_forall_actions[key] = action;
   }
   bool ProfilePlugin::post_parallel_forall_action_registered(void * key) {
      bool registered = s_post_parallel_forall_actions.count(key) > 0;
      return registered;
   }
}


static RAJA::util::PluginRegistry::add<care::ProfilePlugin> L ("Care profile plugin", "test");


