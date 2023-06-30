#include "PluginData.h"

namespace care{
   const char * PluginData::fileName = "N/A";
   int PluginData::lineNumber = -1;
   bool PluginData::s_parallel_context = false;
   std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>> PluginData::s_post_parallel_forall_actions 
      = std::unordered_map<void *, std::function<void(chai::ExecutionSpace, const char *, int)>>{};     
   std::vector<const chai::PointerRecord*> PluginData::s_active_pointers_in_loop = std::vector<const chai::PointerRecord*>{};
   bool PluginData::s_synchronize_before = false; 
   bool PluginData::s_synchronize_after = false;
   int PluginData::s_threadID = -1;

      
   void PluginData::setFileName(const char * name) {PluginData::fileName = name;}
   
   void PluginData::setLineNumber(int num) {PluginData::lineNumber = num;}

   const char * PluginData::getFileName() {return fileName;}

   int PluginData::getLineNumber() {return lineNumber;}

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
