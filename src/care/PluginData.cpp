#include "PluginData.h"

namespace care{
   const char * PluginData::s_file_name = "N/A";
   int PluginData::s_line_number = -1;
   bool PluginData::s_parallel_context = false;
   std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>> PluginData::s_post_parallel_forall_actions 
      = std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>>{};     
   std::vector<const chai::PointerRecord*> PluginData::s_active_pointers_in_loop = std::vector<const chai::PointerRecord*>{};
   int PluginData::s_threadID = -1;

   void PluginData::setFileName(const char * name) {PluginData::s_file_name = name;}
   
   void PluginData::setLineNumber(int num) {PluginData::s_line_number = num;}

   const char * PluginData::getFileName() {return s_file_name;}

   int PluginData::getLineNumber() {return s_line_number;}

   void PluginData::setParallelContext(bool isParallel) {
      s_parallel_context = isParallel;
   }

   bool PluginData::isParallelContext(){
      return s_parallel_context;
   }

   std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>> PluginData::get_post_parallel_forall_actions() {
      return s_post_parallel_forall_actions;
   }

   void PluginData::register_post_parallel_forall_action(void * key, std::function<void(chai::ExecutionSpace, const char *, int)> action) { 
      s_post_parallel_forall_actions[key] = action;
   }

   bool PluginData::post_parallel_forall_action_registered(void * key) {
      bool registered = s_post_parallel_forall_actions.count(key) > 0;
      return registered;
   }

   void PluginData::clear_post_parallel_forall_actions(){s_post_parallel_forall_actions.clear();}

   std::vector<const chai::PointerRecord*> PluginData::getActivePointers() {return s_active_pointers_in_loop;}

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

   void PluginData::clearActivePointers() {s_active_pointers_in_loop.clear();}

}
