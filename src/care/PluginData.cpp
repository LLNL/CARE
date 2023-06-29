#include "chai/ExecutionSpaces.hpp"
#include "care/config.h"
#include "care/CHAICallback.h"
#include <vector>
#include "PluginData.h"

namespace care{
   const char * PluginData::fileName = "N/A";
   int PluginData::lineNumber = -1;
   unsigned int PluginData::s_current_color = 0;
   uint32_t PluginData::s_colors[7] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
   int PluginData::s_num_colors = sizeof(s_colors) / sizeof(uint32_t);
   bool PluginData::s_profile_host_loops = true;
   bool PluginData::s_parallel_context = false;
   std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>> PluginData::s_post_parallel_forall_actions 
      = std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>>{};      
   std::string PluginData::s_current_loop_file_name = "N/A";
   int PluginData::s_current_loop_line_number = -1;
   std::vector<const chai::PointerRecord*> PluginData::s_active_pointers_in_loop = std::vector<const chai::PointerRecord*>{};
   bool PluginData::s_synchronize_before = false; 
   bool PluginData::s_synchronize_after = false;
   int PluginData::s_threadID = -1;

   PluginData::PluginData() {}
   
   void PluginData::setFileName(const char * name) {PluginData::fileName = name;}

   
   void PluginData::setLineNumber(int num) {PluginData::lineNumber = num;}

   const char * PluginData::getCurrentLoopFileName() {
      return fileName;
   }

   int PluginData::getCurrentLoopLineNumber() {
      return lineNumber;
   }

   void PluginData::setParallelContext(bool isParallel) {
      s_parallel_context = isParallel;
   }

   bool PluginData::isParallelContext(){
      return s_parallel_context;
   }

   void PluginData::register_post_parallel_forall_action(void * key, std::function<void(chai::ExecutionSpace, const char *, int)> action) { 
      s_post_parallel_forall_actions[key] = action;
   }
   bool PluginData::post_parallel_forall_action_registered(void * key) {
      bool registered = s_post_parallel_forall_actions.count(key) > 0;
      return registered;
   }

   void PluginData::setSynchronization(bool synchronizeBefore,
                                       bool synchronizeAfter) {
      s_synchronize_before = synchronizeBefore;
      s_synchronize_after = synchronizeAfter;
   }   

   void PluginData::addActivePointer(const chai::PointerRecord* record) {
      s_active_pointers_in_loop.emplace_back(record);
   }

   void PluginData::removeActivePointer(const chai::PointerRecord* record) {
      for (size_t i = 0; i < s_active_pointers_in_loop.size(); ++i) {
         if (s_active_pointers_in_loop[i] == record) {
            s_active_pointers_in_loop[i] = nullptr;
         }
      }
   }
}
