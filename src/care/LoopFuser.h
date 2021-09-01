//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////
#ifndef _CARE_LOOP_FUSER_H_
#define _CARE_LOOP_FUSER_H_

#define CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT 256

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/host_ptr.h"
#include "care/scan.h"

#include <cfloat>

// Priority phase value for the default loop fuser
constexpr double CARE_DEFAULT_PHASE = -FLT_MAX/2.0;

#if CARE_ENABLE_LOOP_FUSER

#include "umpire/Allocator.hpp"
#include "umpire/TypedAllocator.hpp"

// Std library headers
#include <cstdint>
#include <iostream>
#include <vector>

#if defined CARE_GPUCC && defined GPU_ACTIVE
#define FUSIBLE_DEVICE CARE_DEVICE
#else
#define FUSIBLE_DEVICE CARE_HOST
#endif

namespace care {
   ///////////////////////////////////////////////////////////////////////////
   /// @author Ben Liu, Peter Robinson, Alan Dayton
   /// @brief Checks whether an array of type T is sorted and optionally unique.
   /// @param[in] array           - The array to check
   /// @param[in] len             - The number of elements contained in the sorter
   /// @param[in] name            - The name of the calling function
   /// @param[in] argname         - The name of the sorter in the calling function
   /// @param[in] allowDuplicates - Whether or not to allow duplicates
   /// @param[in] warnOnFailure   - Whether to print a warning if array not sorted
   /// @return true if sorted, false otherwise
   ///////////////////////////////////////////////////////////////////////////
   template <typename T>
   inline bool CheckSorted(const T* array, const int len,
                           const char* name, const char* argname,
                           const bool allowDuplicates = false,
                           const bool warnOnFailure = true) {
      if (len > 0) {
         int last = 0;
         bool failed = false;

         if (allowDuplicates) {
            for (int k = 1 ; k < len ; ++k) {
               failed = array[k] < array[last];

               if (failed) {
                  break;
               }
               else {
                  last = k;
               }
            }
         }
         else {
            for (int k = 1 ; k < len ; ++k) {
               failed = array[k] <= array[last];

               if (failed) {
                  break;
               }
               else {
                  last = k;
               }
            }
         }

         if (failed) {
            if (warnOnFailure) {
               std::cout << name << " " << argname << " not in ascending order at index " << last + 1 << std::endl;
            }
            return false;
         }
      }

      return true;
   }

   /************************************************************************
    * Function  : binarySearch
    * Author(s) : Brad Wallin, Peter Robinson
    * Purpose   : Every good code has to have one.  Searches a sorted array,
    *             or a sorted subarray, for a particular value.  This used to
    *             be in NodesGlobalToLocal.  The algorithm was taken from
    *             Numerical Recipes in C, Second Edition.
    *
    *             Important Note: mapSize is the length of the region you
    *             are searching.  For example, if you have an array that has
    *             100 entries in it, and you want to search from index 5 to
    *             40, then you would set start=5, and mapSize=(40-5)=35.
    *             In other words, mapSize is NOT the original length of the
    *             array and it is also NOT the ending index for your search.
    *
    *             If returnUpperBound is set to true, this will return the
    *             index corresponding to the earliest entry that is greater
    *             than num.
    ************************************************************************/

   template <typename T>
   CARE_HOST_DEVICE inline int binarySearch(const T *map, const int start,
                                        const int mapSize, const T num,
                                        bool returnUpperBound = false)
   {
      int klo = start ;
      int khi = start + mapSize;
      int k = ((khi+klo) >> 1) + 1 ;

      if ((map == nullptr) || (mapSize == 0)) {
         return -1 ;
      }

#ifndef CARE_DEVICE_COMPILE
#ifdef CARE_DEBUG
      CheckSorted(&(map[start]), mapSize, "binarySearch", "map") ;
#endif
#endif

      while (khi-klo > 1) {
         k = (khi+klo) >> 1 ;

         if (map[k] == num) {
            if (returnUpperBound) {
               khi = k+1;
               klo = k;
               continue;
            }
            else {
               return k ;
            }
         }
         else if (map[k] > num) {
            khi = k ;
         }
         else {
            klo = k ;
         }
      }

      if (returnUpperBound) {
         k = klo;

         // the lower option bounds num
         if (map[k] > num) {
            return k;
         }

         // the upper option is within the range of the map index set
         if (khi < start + mapSize) {
            // the upper option bounds num
            if (map[khi] > num) {
               return khi;
            }

            // neither the upper or lower option bound num
            return -1;
         }
         else {
            // the lower option does not bound num, and the upper option is out of bounds
            return -1;
         }
      }

      if (map[--k] == num) {
         return k ;
      }
      else {
         return -1 ;
      }
   }
} // namespace care


///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson
/// @brief a collection of FusedActions. Anything that inherits with this and
///        registered with the FusedActionsObserver will be controlled via
///        FUSIBLE_LOOPS_START and FUSIBLE_LOOPS_END macros.
///////////////////////////////////////////////////////////////////////////
class FusedActions {
public:
   CARE_DLL_API static int non_scan_store;
   CARE_DLL_API static bool verbose;
   CARE_DLL_API static bool very_verbose;

   FusedActions() = default;
   ///////////////////////////////////////////////////////////////////////////
   /// @brief starts recording. If recording is stopped, registerAction calls will
   ///        execute the lambda immediately. If recording has started, they will
   ///        be gathered up until flushed either by filling up our buffer or via
   ///        a flush call.
   ///////////////////////////////////////////////////////////////////////////
   virtual void startRecording() {
      m_recording = true; warnIfNotFlushed();
   }

   ///////////////////////////////////////////////////////////////////////////
   /// @brief stops recording. If recording is stopped, registerAction calls will
   ///        execute the lambda immediately. If recording has started, they will
   ///        be gathered up until flushed either by filling up our buffer or via
   ///        a flush call.
   ///////////////////////////////////////////////////////////////////////////
   virtual void stopRecording() { m_recording = false; }

   ///////////////////////////////////////////////////////////////////////////
   /// @brief whether we are currently recording
   ///////////////////////////////////////////////////////////////////////////
   bool isRecording() { return m_recording;}

   ///////////////////////////////////////////////////////////////////////////
   /// @brief number of actions we will fuse
   ///////////////////////////////////////////////////////////////////////////
   int actionCount() { return m_action_count;}

   ///////////////////////////////////////////////////////////////////////////
   /// @brief execute all actions as a fused action
   ///////////////////////////////////////////////////////////////////////////
   virtual void flushActions(bool async, const char * fileName, int lineNumber) = 0;

   ///////////////////////////////////////////////////////////////////////////
   /// @brief execute all actions as a fused action
   ///////////////////////////////////////////////////////////////////////////
   virtual void reset(bool async, const char * fileName, int lineNumber) = 0;

   ///////////////////////////////////////////////////////////////////////////
   /// @brief set preserveOrder mode
   ///////////////////////////////////////////////////////////////////////////
   virtual void preserveOrder(bool preserveOrder) { m_preserve_action_order = preserveOrder;}

   ///////////////////////////////////////////////////////////////////////////
   /// @brief set scan mode
   ///////////////////////////////////////////////////////////////////////////
   virtual void setScan(bool scan) { m_is_scan = scan; }

   ///////////////////////////////////////////////////////////////////////////
   /// @brief set countsToOffsetsScan
   ///////////////////////////////////////////////////////////////////////////
   virtual void setCountsToOffsetsScan(bool scan) { m_is_counts_to_offsets_scan = scan; }


   ///////////////////////////////////////////////////////////////////////////
   /// @brief the destructor
   ///////////////////////////////////////////////////////////////////////////
   virtual ~FusedActions() {};
protected:
   ///
   /// warn if not flushed
   ///
   void warnIfNotFlushed() {
      if (m_action_count > 0) {
         std::cout << "FusedActions not flushed when expected." << std::endl;
      }
   }


   ///
   /// whether to we are actually recording anything registered with us
   ///
   bool m_recording;

   ///
   /// number of actions we have registered to fuse
   ///
   int m_action_count;

   ///
   /// whether to preserve order (TODO - do we need this anymore?)
   ///
   bool m_preserve_action_order;

   ///
   /// whether we are a scan operation
   ///
   bool m_is_scan;

   ///
   /// whether we are a counts to offsets operation
   ///
   bool m_is_counts_to_offsets_scan;
};


using allocator = umpire::TypedAllocator<char>;

///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson
/// @brief The observer of FusesActions. Any FusedActions that is
///        registered with the default FusedActionsObserver (accessed via
///        FusedActionsObserver::getInstance() or FusedActionsObserver::getActiveObserver()
///        will be controlled via FUSIBLE_LOOPS_START and FUSIBLE_LOOPS_END macros.
///////////////////////////////////////////////////////////////////////////
class FusedActionsObserver : public FusedActions {
protected:
   CARE_DLL_API static FusedActionsObserver * activeObserver;
public:
   CARE_DLL_API static FusedActionsObserver * getActiveObserver();
   CARE_DLL_API static std::vector<FusedActionsObserver *> allObservers;
   CARE_DLL_API static void setActiveObserver(FusedActionsObserver * observer);


   FusedActionsObserver(bool registerWithAllObservers = true) : FusedActions(),
                            m_fused_action_order(),
                            m_last_insert_priority(-FLT_MAX),
                            m_to_be_freed(),
                            m_to_be_freed_device(),
                            m_recording(false) 
    {
       if (registerWithAllObservers) {
          allObservers.push_back(this);
       }
    }

   void startRecording() {
      for (auto & priority_action: m_fused_action_order) {
         priority_action.second->startRecording();
      }
      m_recording = true;
   }

   void stopRecording() {
      for ( auto & priority_action: m_fused_action_order) {
         priority_action.second->stopRecording();
      }
      m_recording = false;
   }

   ///////////////////////////////////////////////////////////////////////////
   /// @brief set preserveOrder mode
   ///////////////////////////////////////////////////////////////////////////
   void preserveOrder(bool preserveOrder) {
      for (auto & priority_action: m_fused_action_order) {
         priority_action.second->preserveOrder(preserveOrder);
      }
      m_preserve_action_order = preserveOrder;
   }


   ///////////////////////////////////////////////////////////////////////////
   /// @brief set scan mode
   ///////////////////////////////////////////////////////////////////////////
   virtual void setScan(bool scan) {
      for (auto & priority_action: m_fused_action_order) {
         priority_action.second->setScan(scan);
      }
      m_is_scan = scan;
   }

   ///////////////////////////////////////////////////////////////////////////
   /// @brief set counts_to_offsets_scan mode
   ///////////////////////////////////////////////////////////////////////////
   virtual void setCountsToOffsetsScan(bool scan) {
      for (auto & priority_action: m_fused_action_order) {
         priority_action.second->setCountsToOffsetsScan(scan);
      }
      m_is_counts_to_offsets_scan = scan;
   }


   inline void flushActions(bool async, const char * fileName, int lineNumber) {
      for (auto & priority_action : m_fused_action_order) {
         FusedActions * const & actions = priority_action.second;
         if (actions->actionCount() > 0) {
            if (verbose) {
               printf("flushing actions at priority %g\n", priority_action.first);
            }
            // We allow the Observer to flush actions asynchronously.
            // If async is false, this routine is still synchronous in
            // that the work will be done before we leave it (and everything
            // will be synchronized), but we allow it to be asynchronous internally.
            actions->flushActions(true, fileName, lineNumber);
         }
      }
      if (!async) {
         care::gpuDeviceSynchronize(fileName, lineNumber);
      }
      for (auto & array: m_to_be_freed) {
         array.free();
      }
      m_to_be_freed.clear();
      for (auto & array: m_to_be_freed_device) {
         // Prevent two duplicate frees when wrapping the same host pointer more than once
         if (chai::ArrayManager::getInstance()->getPointerRecord((void *)array.getActivePointer()) != &chai::ArrayManager::s_null_record) {
            array.freeDeviceMemory();
         }
      }
      m_to_be_freed_device.clear();
   }

   template<typename ActionsType>
   inline ActionsType * getFusedActions(double priority) {
      ActionsType * actions = nullptr;
      auto iter = m_fused_action_order.find(priority);
      if (iter == m_fused_action_order.end()) {
#if defined(CARE_GPUCC)
         static allocator a(chai::ArrayManager::getInstance()->getAllocator(chai::PINNED));
#else
         static allocator a(chai::ArrayManager::getInstance()->getAllocator(chai::CPU));
#endif
         
         actions = new ActionsType(a);
         if (m_recording) {
            actions->startRecording();
         } else {
            actions->stopRecording();
         }
         if (m_last_insert_priority > priority) {
            printf("CARE: WARNING fused action encountered out of priority order\n");
         }
         m_fused_action_order[priority] = actions;
      }
      else {
         actions = static_cast<ActionsType *>(iter->second);
      }
      m_last_insert_priority = priority;
      return actions;
   }

   inline void reset_phases() {
      m_last_insert_priority = -FLT_MAX;
   }


   inline int actionCount() {
      int count = 0;
      for (auto priority_action: m_fused_action_order) {
         count += priority_action.second->actionCount();
      }
      return count;
   }

 
   inline void reset(bool /*async*/, const char * /*fileName*/, int /*lineNumber*/) {
      for (auto priority_action: m_fused_action_order) {
         delete priority_action.second;
      }
      m_fused_action_order.clear();
      m_to_be_freed.clear();
      m_to_be_freed_device.clear();
      m_recording = false;
   }


   ///////////////////////////////////////////////////////////////////////////
   /// @author Peter Robinson, Benjamin Liu
   /// @brief registers an array to be released after a flushActions()
   /// @param[in] array : the array to be freed after a flushActions
   /// @param[in] freeDeviceOnly: whether to only free device memory
   ///////////////////////////////////////////////////////////////////////////
   template <typename T>
   inline void registerFree(care::host_device_ptr<T> & array, bool freeDeviceOnly = false) {
      if (!freeDeviceOnly) {
         if (m_recording) {
            m_to_be_freed.push_back(reinterpret_cast<care::host_device_ptr<char> &>(array));
         }
         else {
            array.free();
         }
      }
      else {
         if (m_recording) {
            m_to_be_freed_device.push_back(reinterpret_cast<care::host_device_ptr<char> &>(array));
         }
         else {
            array.freeDeviceMemory();
         }
      }
   }


   ///////////////////////////////////////////////////////////////////////////
   /// @author Peter Robinson
   /// @brief cleans up all FusedActions observed by FusedActionsObserver::allObservers();
   /// @param[in] array : the array to be freed after a flushActions
   ///////////////////////////////////////////////////////////////////////////
   CARE_DLL_API static void cleanupAllFusedActions();

   protected:
      std::map<double, FusedActions *> m_fused_action_order;
      double m_last_insert_priority;
      std::vector<care::host_device_ptr<char> > m_to_be_freed;
      std::vector<care::host_device_ptr<char> > m_to_be_freed_device;
      bool m_recording;

};

   

template <int REGISTER_COUNT>
struct fusible_registers_t {
   static const int CUDA_WORKGROUP_BLOCK_SIZE = 2048/(REGISTER_COUNT/32);
};


using index_type = int;
// This class is meant to orchestrate fusing a bunch of loops together. The initial use case
// is our communication routines. The goal is to do one giant scan at the end over the entire pack
// buffer.
// You register loops' bodies as actions, the start and end of the index set you want for that action,
// and a conditional lambda takes a single argument and returns a boolean if you want the action
// to occur at that index.
// flushActions() will then orchestrate the scan operation to fuse all of your scans into a single one.
template <int REGISTER_COUNT, typename...XARGS>
class LoopFuser : public FusedActions {
   public:
      using fusible_registers = fusible_registers_t<REGISTER_COUNT>;
      using action_xargs = RAJA::xargs<index_type * /*scan_var*/, XARGS...>;
      using conditional_xargs = RAJA::xargs<index_type * /*scan_var*/, index_type const * /*scan_offsets*/,  int /*total length */, XARGS...>;

      // TODO - explore varying policy block execution based off of register binning types
#if defined CARE_GPUCC && defined GPU_ACTIVE
      using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::cuda_work_async<fusible_registers::CUDA_WORKGROUP_BLOCK_SIZE>,
                                 RAJA::unordered_cuda_loop_y_block_iter_x_threadblock_average,
                                 RAJA::constant_stride_array_of_objects >;
      using workgroup_ordered_policy = RAJA::WorkGroupPolicy <
                                 RAJA::cuda_work_async<fusible_registers::CUDA_WORKGROUP_BLOCK_SIZE>,
                                 RAJA::ordered,
                                 RAJA::constant_stride_array_of_objects >;
#else
      using workgroup_policy = RAJA::WorkGroupPolicy <
                                 RAJA::loop_work,
                                 RAJA::ordered,
                                 RAJA::ragged_array_of_objects >;
      using workgroup_ordered_policy = RAJA::WorkGroupPolicy <
                                 RAJA::loop_work,
                                 RAJA::ordered,
                                 RAJA::ragged_array_of_objects >;
#endif
 
      using action_workpool = RAJA::WorkPool< workgroup_policy,
                                     index_type,
                                     action_xargs,
                                     allocator >;

      using action_workgroup = RAJA::WorkGroup< workgroup_policy,
                                       index_type,
                                       action_xargs,
                                       allocator >;

      using action_ordered_workgroup = RAJA::WorkGroup< workgroup_ordered_policy,
                                       index_type,
                                       action_xargs,
                                       allocator >;

      using action_worksite = RAJA::WorkSite< workgroup_policy,
                                     index_type, 
                                     action_xargs,
                                     allocator >;

      using action_ordered_worksite = RAJA::WorkSite< workgroup_ordered_policy,
                                     index_type, 
                                     action_xargs,
                                     allocator >;

      using conditional_workpool = RAJA::WorkPool< workgroup_policy,
                                     index_type,
                                     conditional_xargs,
                                     allocator >;

      using conditional_workgroup = RAJA::WorkGroup< workgroup_policy,
                                       index_type,
                                       conditional_xargs,
                                       allocator >;

      using conditional_worksite = RAJA::WorkSite< workgroup_policy,
                                     index_type, 
                                     conditional_xargs,
                                     allocator >;
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief The default constructor. Intentionally am not keeping this private
      ///        in the event that a user wants to maintain multiple independent
      ///        LoopFuser objects.
      ///////////////////////////////////////////////////////////////////////////
      CARE_DLL_API LoopFuser<REGISTER_COUNT, XARGS...>(allocator);


      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief The destructor.
      ///////////////////////////////////////////////////////////////////////////
      CARE_DLL_API ~LoopFuser<REGISTER_COUNT, XARGS...>();

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief gets a static singleton instance of a LoopFuser.
      /// @return The default instance.
      ///////////////////////////////////////////////////////////////////////////
      CARE_DLL_API static LoopFuser<REGISTER_COUNT, XARGS...> * getInstance();

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief registers a loop lambda with the packer.
      ///////////////////////////////////////////////////////////////////////////
      template <typename LB, typename Conditional>
      void registerAction(const char * fileName, int lineNumber, int start, int end, int & start_pos, Conditional && conditional, 
                          LB && action, int scan_type = 0, int & pos_store = non_scan_store,
                          care::host_device_ptr<int> counts_to_offsets_scanvar = nullptr);
      

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief starts recording. If recording is stopped, registerAction calls will
      ///        execute the lambda immediately. If recording has started, they will
      ///        be gathered up until flushed either by filling up our buffer or via
      ///        a flush call.
      ///////////////////////////////////////////////////////////////////////////
      void startRecording();

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief stops recording. If recording is stopped, registerAction calls will
      ///        execute the lambda immediately. If recording has started, they will
      ///        be gathered up until flushed either by filling up our buffer or via
      ///        a flush call.
      ///////////////////////////////////////////////////////////////////////////
      void stopRecording();

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief execute all recorded actions
      ///////////////////////////////////////////////////////////////////////////
      CARE_DLL_API void flushActions(bool async=false , const char * fileName = "\0", int lineNumber=-1);

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief execute all recorded actions in parallel
      ///////////////////////////////////////////////////////////////////////////
      void flush_parallel_actions(bool async, const char * fileName, int lineNumber);

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief execute all recorded actions in a sequence
      ///////////////////////////////////////////////////////////////////////////
      void flush_order_preserving_actions(bool async, const char * fileName, int lineNumber);

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief execute all recorded scans and actions in parallel
      ///////////////////////////////////////////////////////////////////////////
      void flush_parallel_scans(const char * fileName, int lineNumber);
      
      /// @author Peter Robinson
      /// @brief execute all recorded counts_to_offsets scans and actions in parallel
      ///////////////////////////////////////////////////////////////////////////
      void flush_parallel_counts_to_offsets_scans(bool async, const char * fileName, int lineNumber);

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief allocate buffers for size lambdas.
      /// @param[in] size - number of lambdas to support recording before flushing
      ///////////////////////////////////////////////////////////////////////////
      void reserve(size_t size);

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief allocate buffers for serialized lambda data
      /// @param[in] size - number of bytes to allocate a buffer for.
      ///////////////////////////////////////////////////////////////////////////
      void reserve_lambda_buffer(size_t size);

      int size() { return m_action_count; }

      int reserved() { return m_reserved; }

      void reset(bool async, const char * fileName, int lineNumber);

      int getOffset() {
         if (!m_preserve_action_order) {
            return m_action_count == 0 ? 0 : m_action_offsets[m_action_count-1];
         }
         else {
            return 0;
         }
      }

      index_type * getScanPosStarts() { return m_scan_pos_starts;}
      index_type * getScanPosOutputs() { return m_scan_pos_outputs;}

      void setVerbose(bool v) { verbose = v; }

      void setReverseIndices(bool reverse) { m_reverse_indices = reverse; }

      void waitIfNeeded();

   private:
      ///
      /// warn if not flushed
      ///
      void warnIfNotFlushed();
      ///
      /// allocator for any of our buffers. Needs to be writeable on host and device. 
      ///
      allocator m_allocator;

      ///
      /// the max length of an action's index set
      ///
      int m_max_action_length;

      ///
      /// How many lambdas we are supporting
      ///
      int m_reserved;

      ///
      /// How big of a buffer in pinned memory we have reserved
      ///
      int m_totalsize;

      ///
      /// Host pointer (pinned) for action offsets
      ///
      index_type *m_action_offsets;

      ///
      /// conditional workpool, used for initializing m_scanvar
      ///
      conditional_workpool m_conditionals;
      
      ///
      /// conditional workgroup
      ///
      conditional_workgroup m_cw;

      ///
      /// worksite for conditional workpool
      ///
      conditional_worksite m_cws;

      ///
      /// container of action serialized lambdas
      ///
      action_workpool m_actions;

      ///
      /// action workgroup
      ///
      action_workgroup m_aw;


      ///
      /// worksite for action workpool
      ///
      action_worksite m_aws;
      
      ///
      /// Type of scan (0 = no scan, 1 = regular scan, 2 = counts_to_offsets scan)
      ///
      int m_scan_type;

      ///
      /// The pinned buffer to store scan position outputs
      ///
      index_type * m_scan_pos_outputs;

      ///
      /// The pinned buffer for scan pos starts
      ///
      index_type * m_scan_pos_starts;


      ///
      /// cached scan position output addresses
      ///
      care::host_ptr<int> *m_pos_output_destinations;

      ///
      /// cached output destination so we notice when there's a new one
      ///
      care::host_ptr<int> m_prev_pos_output;

      ///
      /// runtime control on whether to reverse indices during flush_parallel_actions
      ///
      bool m_reverse_indices = false;

      ///
      /// resource whose stream we will use for asynchronous events.
      ///
      using StreamResource = RAJA::resources::get_resource<RAJADeviceExec>::type;
      StreamResource m_async_resource;

      ///
      /// Event that we will use for asynchronous events.
      ///
       using EventType = RAJA::resources::Event;
       EventType m_wait_for_event;

      ///
      /// whether we are waiting for an active event
      ///
      bool m_wait_needed = false;
};

// The FUSIBLE_REGISTERS_* macros define the type signature for the XARGS that LoopFuser will be templated on. They are designed
// to provide function signatures that are unique to the each respective register bin. This ensures that the linker will only consider
// fused loops within a register bin when doing block/grid/register size determinations. 
// Note that for nvcc, the linker appears to consider LoopFuser<32> to be compatible with LoopFuser<64> etc, so differentiating the signature
// by the integer template parameter was not sufficient. Thus we differentiate by number of parameters as well. 
//
#ifdef CARE_ENABLE_FUSER_BIN_32
#define FUSIBLE_REGISTERS_32 LoopFuser<32>::fusible_registers
#endif
#define FUSIBLE_REGISTERS_64 LoopFuser<64>::fusible_registers, LoopFuser<64>::fusible_registers
#define FUSIBLE_REGISTERS_128 LoopFuser<128>::fusible_registers, LoopFuser<128>::fusible_registers, LoopFuser<128>::fusible_registers
#define FUSIBLE_REGISTERS_256 LoopFuser<256>::fusible_registers, LoopFuser<256>::fusible_registers, LoopFuser<256>::fusible_registers, LoopFuser<256>::fusible_registers

#define _FUSIBLE_REGISTERS(REGISTER_COUNT) FUSIBLE_REGISTERS_##REGISTER_COUNT
#define FUSIBLE_REGISTERS(REGISTER_COUNT) _FUSIBLE_REGISTERS(REGISTER_COUNT)

#ifdef CARE_ENABLE_FUSER_BIN_32
extern template class LoopFuser<32 , FUSIBLE_REGISTERS(32)>; 
#endif
extern template class LoopFuser<64 , FUSIBLE_REGISTERS(64)>; 
extern template class LoopFuser<128, FUSIBLE_REGISTERS(128)>; 
extern template class LoopFuser<256, FUSIBLE_REGISTERS(256)>; 


///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson
/// @brief registers a loop lambda with the packer.
/// @param[in] start index of the loop
/// @param[end] end index of the loop
/// @param[body] The loop body lambda.
/// @param[conditional] A conditional lambda that returns whether to execute
///                     this loop (currently mostly ignored, will be used
///                     when scan support is added.
///////////////////////////////////////////////////////////////////////////
template<int REGISTER_COUNT, typename...XARGS>
template <typename LB, typename Conditional>
void LoopFuser<REGISTER_COUNT, XARGS...>::registerAction(const char * fileName, int lineNumber, int start, int end, int &start_pos, Conditional && conditional, LB && action, int scan_type, int &pos_store, care::host_device_ptr<int> counts_to_offsets_scanvar) {
   int length = end - start;
   if (length) {
      /* switch to scan mode if we encounter a scan before we flush */
      if (scan_type == 1) {
         m_is_scan = true;
         m_is_counts_to_offsets_scan = false;
#ifdef CARE_DEBUG
         if (&start_pos != &pos_store) {
            std::cout << "LoopFuser<"<<REGISTER_COUNT<<">::registerAction : pos initializer must be same lvalue as pos destination for scans to be fusible" << std::endl;
         }
#endif
      }
      else if (scan_type == 2) {
         m_is_counts_to_offsets_scan = true;
         if (m_is_scan) {
            std::cout << "LoopFuser<"<<REGISTER_COUNT<<">::registerAction : counts_to_offsets scan is not fusible with normal scans" << std::endl;
         }
         m_is_scan = false;
      }
      if (m_recording) {
         if (verbose) {
            printf("%p: Registering action %i type %i with start %i and end %i\n", this, m_action_count, scan_type, start, end);
         }
         waitIfNeeded();
         m_actions.enqueue(RAJA::RangeSegment(0,length), action);
         m_conditionals.enqueue(RAJA::RangeSegment(0,length), conditional);

         m_action_offsets[m_action_count] = m_action_count == 0 ? length : m_action_offsets[m_action_count-1] + length;
         m_scan_pos_starts[m_action_count] = start_pos;

         if (verbose) {
            printf("Registered action %i with offset %i\n",
                   m_action_count, m_action_offsets[m_action_count]);
         }
         m_max_action_length = std::max(m_max_action_length, end-start);

         // SCAN related variables
         if (m_prev_pos_output == nullptr) {
            // initialize m_prev_pos_output
            m_prev_pos_output = care::host_ptr<int>(&pos_store);
         }
         else {
            // if we encounter a different output, remember it
            if (m_prev_pos_output.cdata() != &pos_store) {
               m_prev_pos_output = care::host_ptr<int>(&pos_store);
            }
            // if we haven't encountered a different output yet, mark this index for continuation
            else if (m_prev_pos_output.cdata() == &pos_store) {
               // mark the start for continuation
               m_scan_pos_starts[m_action_count] = -999;
            }
         }
         m_pos_output_destinations[m_action_count] = &pos_store;
         ++m_action_count;

         if (m_action_count == m_reserved) {
            if (verbose) {
               printf("hit reserved flushActions\n");
            }
            flushActions();
         }
         // if we are approaching the 2^31-1 limit proactively flush
         else if (m_action_offsets[m_action_count-1] > 2000000000) {
            if (verbose) {
               printf("hit m_action_offsets flushActions\n");
            }
            flushActions();
         }
      }
      else {
         if (verbose) {
            printf("calling as packed %s:%i\n", fileName, lineNumber);
         }
         switch(scan_type) {
            case 0:
#if defined CARE_GPUCC && defined GPU_ACTIVE
               care::forall(care::raja_fusible {}, 0, length, action, fileName, lineNumber, XARGS{}...);
#else
               care::forall(care::raja_fusible_seq {}, 0, length, action, XARGS{}...); 
#endif
               break;
            case 1:
               {
                  m_scan_pos_starts[m_action_count] = start_pos;
#if defined GPU_ACTIVE || defined CARE_ALWAYS_USE_RAJA_SCAN
                  if (verbose) {
                     printf("calling GPU_ACTIVE scan with start_pos %i action_count %i\n", start_pos, m_action_count);
                  }
                  auto conditional_wrapper = [=] FUSIBLE_DEVICE(index_type i, int * scanvar, int global_end) -> bool {
                     conditional(i, scanvar, nullptr, global_end, XARGS{}...);
                     return scanvar[i];
                  };
                  // need to store a copy of start_pos, as start_pos and pos_store are aliased by design
                  int start_pos_before_scan = start_pos;
                  SCAN_LOOP(i, 0, length, pos, 0, conditional_wrapper(i,SCANVARNAME(pos).data(),length)) {
                     action(i, SCANVARNAME(pos).data(), XARGS{}... );
                  } SCAN_LOOP_END(length, pos, pos_store)
                  pos_store += start_pos_before_scan;
#else
                  if (verbose) {
                     printf("calling not GPU_ACTIVE scan\n");
                  }
                  auto conditional_wrapper = [=] FUSIBLE_DEVICE(index_type i, int * scanvar, int global_end) -> bool {
                     // pass scanvar twice so that the lambda knows to treat scanvar as an address to a scalar
                     conditional(i, scanvar, scanvar, global_end, XARGS{}...);
                     return *scanvar;
                  };
                  // need to store a copy of start_pos, as start_pos and pos_store are aliased by design
                  int start_pos_before_scan = start_pos;
                  SCAN_LOOP(i, 0, length, pos, 0, conditional_wrapper(i,&SCANVARNAME(pos),length)) {
                     // when !(GPU_ACTIVE || CARE_ALWAYS_USE_RAJA_SCAN), SCANVARNAME evaluates to a scalar, 
                     // but action will look at the ith entry of an array, so we take the address and subtract
                     // off i to land back at the scalar
                     action(i, &SCANVARNAME(pos)-i, XARGS{}... );
                  } SCAN_LOOP_END(length, pos, pos_store)
                  pos_store += start_pos_before_scan;
#endif
               }
               break;
            case 2:
               SCAN_COUNTS_TO_OFFSETS_LOOP(i, 0, length, counts_to_offsets_scanvar) {
                  action(i, nullptr, XARGS{}...);
               } SCAN_COUNTS_TO_OFFSETS_LOOP_END(i, length, counts_to_offsets_scanvar)
               break;
            default:
               printf("care::LoopFuser<%i>::encountered unhandled scan type\n", REGISTER_COUNT);
               break;
         }
      }
   }
}


#if defined(CARE_GPUCC)
#define DEFAULT_ALLOCATOR allocator(chai::ArrayManager::getInstance()->getAllocator(chai::PINNED))
#else
#define DEFAULT_ALLOCATOR allocator(chai::ArrayManager::getInstance()->getAllocator(chai::CPU))
#endif
#if defined(CARE_FUSIBLE_LOOPS_DISABLE)
#define START_RECORDING(FUSER)
#else
#define START_RECORDING(FUSER) FUSER->startRecording()
#endif

#define LOOPFUSER(REGISTER_COUNT) LoopFuser<REGISTER_COUNT, FUSIBLE_REGISTERS(REGISTER_COUNT)>

#if defined(CARE_ENABLE_FUSER_BIN_32)
#define FUSED_ACTION_INSTANCE_32 static_cast<FusedActions *> (LOOPFUSER(32)::getInstance())
#define FUSED_INSTANCE_COMMA ,
#else
#define FUSED_ACTION_INSTANCE_32
#define FUSED_INSTANCE_COMMA
#endif

#if defined(CARE_DEBUG) || defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE



// Start recording
#define FUSIBLE_LOOPS_START { \
   static FusedActionsObserver * __phase_observer = new FusedActionsObserver(); \
   for ( FusedActions *__fuser__ : { \
                                    static_cast<FusedActions *> (LOOPFUSER(256)::getInstance()),\
                                    static_cast<FusedActions *> (LOOPFUSER(128)::getInstance()),\
                                    static_cast<FusedActions *> (LOOPFUSER(64)::getInstance()),\
                                    FUSED_ACTION_INSTANCE_32 FUSED_INSTANCE_COMMA \
                                    static_cast<FusedActions *>(__phase_observer)}) { \
      START_RECORDING(__fuser__); \
      __fuser__->preserveOrder(false); \
      __fuser__->setScan(false); \
   } \
   FusedActionsObserver::setActiveObserver(__phase_observer); \
}


#define FUSIBLE_LOOPS_PRESERVE_ORDER_START { \
   static FusedActionsObserver * __phase_observer = new FusedActionsObserver(); \
   for ( FusedActions *__fuser__ : {\
                                    static_cast<FusedActions *> (LOOPFUSER(256)::getInstance()),\
                                    static_cast<FusedActions *> (LOOPFUSER(128)::getInstance()),\
                                    static_cast<FusedActions *> (LOOPFUSER(64)::getInstance()),\
                                    FUSED_ACTION_INSTANCE_32 FUSED_INSTANCE_COMMA \
                                    static_cast<FusedActions *>(__phase_observer)}) { \
      START_RECORDING(__fuser__); \
      __fuser__->preserveOrder(true); \
      __fuser__->setScan(false); \
   } \
   FusedActionsObserver::setActiveObserver(__phase_observer); \
}

// Execute, then stop recording. Note only need to pass ASYNC to last flush call. Prevents
// extra synchronizes between each loop fuser.
#define _FUSIBLE_LOOPS_STOP(ASYNC) { \
   for ( FusedActions *__fuser__ : {\
                                    static_cast<FusedActions *> (LOOPFUSER(256)::getInstance()),\
                                    static_cast<FusedActions *> (LOOPFUSER(128)::getInstance()),\
                                    static_cast<FusedActions *> (LOOPFUSER(64)::getInstance()),\
                                    FUSED_ACTION_INSTANCE_32 \
                                    }) { \
      __fuser__->flushActions(true, __FILE__, __LINE__); \
      __fuser__->stopRecording(); \
   } \
   FusedActions * __fuser__ = static_cast<FusedActions *>(FusedActionsObserver::getActiveObserver()); \
   __fuser__->flushActions(ASYNC, __FILE__, __LINE__); \
   __fuser__->stopRecording(); \
   FusedActionsObserver::setActiveObserver(nullptr); \
}

// Execute, then stop recording
#define FUSIBLE_LOOPS_STOP _FUSIBLE_LOOPS_STOP(false)

// Execute asynchronously, then stop recording
#define FUSIBLE_LOOPS_STOP_ASYNC _FUSIBLE_LOOPS_STOP(true)


// frees
#define FUSIBLE_FREE(A) FusedActionsObserver::getActiveObserver()->registerFree(A);
#define FUSIBLE_FREE_DEVICE(A) FusedActionsObserver::getActiveObserver()->registerFree(A, true);

#else // defined(CARE_DEBUG) || defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE

// in opt, non cuda builds, never start recording
#define FUSIBLE_LOOPS_START \
{ \
   static FusedActionsObserver * __phase_observer = new FusedActionsObserver(); \
   for ( FusedActions * __fuser__ : {\
                                     static_cast<FusedActions *> (LOOPFUSER(256)::getInstance()),\
                                     static_cast<FusedActions *> (LOOPFUSER(128)::getInstance()),\
                                     static_cast<FusedActions *> (LOOPFUSER(64)::getInstance()),\
                                     FUSED_ACTION_INSTANCE_32 FUSED_INSTANCE_COMMA \
                                     static_cast<FusedActions *>(__phase_observer)}) { \
      __fuser__->stopRecording(); \
      __fuser__->setScan(false); \
      __fuser__->preserveOrder(false); \
      __fuser__->setCountsToOffsetsScan(false); \
   } \
   FusedActionsObserver::setActiveObserver(__phase_observer); \
}


#define FUSIBLE_LOOPS_PRESERVE_ORDER_START
#define FUSIBLE_LOOPS_STOP FusedActionsObserver::setActiveObserver(nullptr);
#define FUSIBLE_LOOPS_STOP_ASYNC FusedActionsObserver::setActiveObserver(nullptr);
#define FUSIBLE_FREE(A) A.free();
#define FUSIBLE_FREE_DEVICE(A) A.freeDeviceMemory();

#endif // defined(CARE_DEBUG) || defined(CARE_GPUCC)

#define FUSIBLE_KERNEL_BOOKKEEPING(FUSER) \
   auto __fusible_offset__ = FUSER->getOffset(); \
   auto __fusible_start_index__ = 0;

// initializes index start, end and offset variables for boilerplate reduction
// Note that __fusible_scan_pos_*_ are raw pointers being captured into a lambda,
// In many situations this can lead to dereferencing of a host pointer, so it is desirable
// to have a clang-query that can check for this. Below is a query that checks for raw
// pointer captures and also ignores these particular variables.
//
//let comment "### Identify device lambda contexts"
//let b1 callExpr(hasArgument(0,hasType(asString("struct care::gpu"))))
//let b2 callExpr(hasArgument(0,hasType(asString("struct care::parallel"))))
//let b3 cxxMemberCallExpr(on(hasType(pointsTo(cxxRecordDecl(hasName("LoopFuser"))))))
//let rajaDeviceContext callExpr(anyOf(b1,b2,b3))
//let inDeviceLambda hasAncestor(lambdaExpr(hasAncestor(rajaDeviceContext)))
//let comment "### Ignore implicit casts (to prevent duplicate matches)"
//let notInImplicitCast unless(hasAncestor(implicitCastExpr()))
//let comment "### Set up detection of reference of raw pointer in device lambda"
//let comment "### ignoringImplicit and no implicitCastExpr ancestor is required to prevent duplicate matches"
//match expr(inDeviceLambda, notInImplicitCast, ignoringImplicit(declRefExpr(to(varDecl(hasType(isAnyPointer()), 
//           unless(matchesName("__fusible_scan_pos.*__")), unless(inDeviceLambda)))))).bind("capture_of_raw_pointer_in_lambda")

#define FUSIBLE_BOOKKEEPING(FUSER,START,END,REGISTER_COUNT) \
   FUSIBLE_KERNEL_BOOKKEEPING(FUSER) ; \
   auto __fusible_action_index__ = FUSER->actionCount(); \
   index_type *__fusible_scan_pos_starts__ = FUSER->getScanPosStarts(); \
   index_type *__fusible_scan_pos_outputs__ = FUSER->getScanPosOutputs(); \
   __fusible_start_index__ = START; \
   auto __fusible_end_index__ = END; \
   auto __fusible_verbose__ = FusedActions::verbose; \
   __fusible_offset__ = __fusible_offset__; \
   __fusible_action_index__ = __fusible_action_index__ ; \
   __fusible_scan_pos_starts__ = __fusible_scan_pos_starts__ ;  \
   __fusible_scan_pos_outputs__ = __fusible_scan_pos_outputs__ ; \
   __fusible_verbose__ = __fusible_verbose__ ;
   

// adjusts the index by adding the loop start index and subtracting off the
// loop fuser offset to bring the loop
// from the fuser global index space back into its own index space.
// NOTE: no adjustment necessary with 2D launch
 #define FUSIBLE_INDEX_ADJUST(INDEX) \
    int __fusible_global_index__ = __fusible_offset__+ INDEX; \
    INDEX += __fusible_start_index__;  \
    __fusible_global_index__ = __fusible_global_index__; // quiet compiler where unused
// NOTE: adjustment necessary with 1D launch
// #define FUSIBLE_INDEX_ADJUST(INDEX) int __fusible_global_index__ = INDEX; INDEX += __fusible_start_index__ - __fusible_offset__ ;

// adjusts the index and then ensures the loop is only executed if the
// resulting index is within the index range of the loop
#define FUSIBLE_LOOP_PREAMBLE(INDEX) \
   FUSIBLE_INDEX_ADJUST(INDEX) ; \
   if (INDEX < __fusible_end_index__)


// adjusts the index and then ensures the loop is only executed if the
// resulting index is within the index range of the loop,
// as well as ensuring we only execute where our scan was true
// also initializes POS to an appropriate value, searching for the pos start
// for this action's group of scans (actions share a scan if their output is the same reference)
// TODO: drop the use of BOOL_EXPR in favor of inspecting the scan var values
#define FUSIBLE_SCAN_LOOP_PREAMBLE(INDEX, BOOL_EXPR, GLOBAL_SCAN_VAR, POS) \
   FUSIBLE_INDEX_ADJUST(INDEX) ;  \
   int __startIndex = __fusible_action_index__; \
   while (__fusible_scan_pos_starts__[__startIndex] == -999) { \
      --__startIndex; \
   } \
   const int __scan_pos_start = __fusible_scan_pos_starts__[__startIndex]; \
   const int __scan_pos_offset = __startIndex == 0 ? 0 : __fusible_scan_pos_outputs__[__startIndex-1]; \
   const int POS = GLOBAL_SCAN_VAR[__fusible_global_index__]  + __scan_pos_start - __scan_pos_offset; \
   if (INDEX < __fusible_end_index__ && (BOOL_EXPR))
   

// first couple of arguments to registerAction are defined in above macros, so
// we have them wrapped up in a macro to enforce name consistency
#define FUSIBLE_REGISTER_ARGS __FILE__, __LINE__, __fusible_start_index__, __fusible_end_index__


// conditional xargs to pass in to lambdas
#define FUSIBLE_CONDITIONAL_XARGS(REGISTER_COUNT) int * __fusible_scan_var__, index_type const * __fusible_scan_offsets__, FUSIBLE_REGISTERS(REGISTER_COUNT)
#define FUSIBLE_ALWAYS_TRUE(INDEX, REGISTER_COUNT) [=] FUSIBLE_DEVICE(index_type INDEX, int * __fusible_scan_var__, index_type const *, int, FUSIBLE_REGISTERS(REGISTER_COUNT)) { FUSIBLE_INDEX_ADJUST(INDEX);  __fusible_scan_var__[__fusible_global_index__] = true; }
// actions xargs to pass in to lambdas
#define FUSIBLE_ACTION_XARGS(REGISTER_COUNT) index_type *, FUSIBLE_REGISTERS(REGISTER_COUNT)

#define FUSIBLE_LOOP_STREAM_R(INDEX, START, END, REGISTER_COUNT) { \
   auto __fuser__ = LOOPFUSER(REGISTER_COUNT)::getInstance(); \
   static int __fusible_scan_pos__ ; \
   __fusible_scan_pos__ = 0; \
   FUSIBLE_BOOKKEEPING(__fuser__,START,END, REGISTER_COUNT); \
   __fuser__->registerAction( FUSIBLE_REGISTER_ARGS, __fusible_scan_pos__, \
                              FUSIBLE_ALWAYS_TRUE(INDEX, REGISTER_COUNT), \
                              [=] FUSIBLE_DEVICE(index_type INDEX, FUSIBLE_ACTION_XARGS(REGISTER_COUNT)) { \
                              FUSIBLE_LOOP_PREAMBLE(INDEX) {

#define FUSIBLE_LOOP_STREAM_R_END \
                              } }); }

#define FUSIBLE_LOOP_STREAM(INDEX, START, END) FUSIBLE_LOOP_STREAM_R(INDEX, START, END, CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT)
#define FUSIBLE_LOOP_STREAM_END FUSIBLE_LOOP_STREAM_R_END


#define FUSIBLE_KERNEL_R(REGISTER_COUNT) { \
   auto __fuser__ = LOOPFUSER(REGISTER_COUNT)::getInstance(); \
   FUSIBLE_KERNEL_BOOKKEEPING(__fuser__) ; \
   static int __fusible_scan_pos__ ; \
   __fusible_scan_pos__ = 0; \
   __fuser__->registerAction(__FILE__, __LINE__, 0, 1, __fusible_scan_pos__, \
                             FUSIBLE_ALWAYS_TRUE(__i__, REGISTER_COUNT), \
                             [=] FUSIBLE_DEVICE(int, FUSIBLE_ACTION_XARGS(REGISTER_COUNT))->int {

#define FUSIBLE_KERNEL FUSIBLE_KERNEL_R(CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT)

#define FUSIBLE_LOOP_PHASE_R(INDEX, START, END, PRIORITY, REGISTER_COUNT) { \
   if (END > START) { \
      LOOPFUSER(REGISTER_COUNT) * __fuser__ = FusedActionsObserver::getActiveObserver()->getFusedActions<LOOPFUSER(REGISTER_COUNT)>(PRIORITY); \
      FUSIBLE_BOOKKEEPING(__fuser__, START, END, REGISTER_COUNT); \
      static int __fusible_scan_pos__; \
      __fusible_scan_pos__ = 0; \
      __fuser__->registerAction( FUSIBLE_REGISTER_ARGS, __fusible_scan_pos__, \
                                 FUSIBLE_ALWAYS_TRUE(INDEX, REGISTER_COUNT), \
                                 [=] FUSIBLE_DEVICE(int INDEX, FUSIBLE_ACTION_XARGS(REGISTER_COUNT)) { \
                                    FUSIBLE_LOOP_PREAMBLE(INDEX) { \


#define FUSIBLE_LOOP_PHASE_R_END \
                                    } \
                                    }); }}

#define FUSIBLE_LOOP_PHASE(INDEX, START, END, PRIORITY) FUSIBLE_LOOP_PHASE_R(INDEX, START, END, PRIORITY, CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT)
#define FUSIBLE_LOOP_PHASE_END FUSIBLE_LOOP_PHASE_R_END


#define FUSIBLE_KERNEL_PHASE_R(PRIORITY, REGISTER_COUNT) { \
   LOOPFUSER(REGISTER_COUNT) * __fuser__ = FusedActionsObserver::getActiveObserver()->getFusedActions<LOOPFUSER(REGISTER_COUNT)>(PRIORITY); \
   static int __fusible_scan_pos__ ; \
   __fusible_scan_pos__ = 0; \
   __fuser__->registerAction(__FILE__, __LINE__, 0, 1, __fusible_scan_pos__, \
                             FUSIBLE_ALWAYS_TRUE(__i__, REGISTER_COUNT), \
                             [=] FUSIBLE_DEVICE(int, FUSIBLE_ACTION_XARGS(REGISTER_COUNT)) {



#define FUSIBLE_KERNEL_R_END return 0;}); }
#define FUSIBLE_KERNEL_PHASE_R_END FUSIBLE_KERNEL_R_END
#define FUSIBLE_KERNEL_END FUSIBLE_KERNEL_R_END

#define FUSIBLE_KERNEL_PHASE(PRIORITY) FUSIBLE_KERNEL_PHASE_R(PRIORITY, CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT)

#define FUSIBLE_PHASE_RESET FusedActionsObserver::getActiveObserver()->reset_phases();

#ifdef CARE_ENABLE_FUSER_BIN_32
#define INSTANCE_32_INCREMENT_SIZE __fusible_action_count__ += LOOPFUSER(32)::getInstance()->size();
#else
#define INSTANCE_32_INCREMENT_SIZE
#endif

#define FUSIBLE_LOOPS_FENCEPOST { \
   int __fusible_action_count__ = LOOPFUSER(256)::getInstance()->size(); \
   __fusible_action_count__ += LOOPFUSER(128)::getInstance()->size(); \
   __fusible_action_count__ += LOOPFUSER(64)::getInstance()->size(); \
   INSTANCE_32_INCREMENT_SIZE \
   if (__fusible_action_count__ > 0) { \
      std::cout << __FILE__ << "FUSIBLE_FENCEPOST reached before FUSIBLE_LOOPS_STOP occurred!" << std::endl; \
   } \
}

// SCANS
#define _FUSIBLE_LOOP_SCAN_R(FUSER, INDEX, START, END, POS, INIT_POS, BOOL_EXPR, REGISTER_COUNT) { \
   auto __fuser__ = FUSER; \
   FUSIBLE_BOOKKEEPING(__fuser__, START, END, REGISTER_COUNT); \
   __fuser__->registerAction( FUSIBLE_REGISTER_ARGS, INIT_POS, \
                              [=] FUSIBLE_DEVICE(int INDEX, int * SCANVAR, index_type const * HACK_FLAG, int GLOBAL_END, \
                                 FUSIBLE_REGISTERS(REGISTER_COUNT)){ \
                                 FUSIBLE_INDEX_ADJUST(INDEX); \
                                 if (HACK_FLAG) { \
                                    *SCANVAR = (int) (__fusible_global_index__ != GLOBAL_END && (BOOL_EXPR)); \
                                 } else { \
                                    SCANVAR[__fusible_global_index__] = (int) (__fusible_global_index__ != GLOBAL_END && (BOOL_EXPR)); \
                                 } \
                              }, \
                              [=] FUSIBLE_DEVICE(int INDEX, int * SCANVAR, FUSIBLE_REGISTERS(REGISTER_COUNT)){ \
                                 FUSIBLE_SCAN_LOOP_PREAMBLE(INDEX, BOOL_EXPR, SCANVAR, POS) {

#define FUSIBLE_LOOP_SCAN_R(INDEX, START, END, POS, INIT_POS, BOOL_EXPR, REGISTER_COUNT) \
   _FUSIBLE_LOOP_SCAN_R(LOOPFUSER(REGISTER_COUNT)::getInstance(), INDEX, START, END, POS, INIT_POS, BOOL_EXPR, REGISTER_COUNT)

#define FUSIBLE_LOOP_SCAN(INDEX, START, END, POS, INIT_POS, BOOL_EXPR) \
   FUSIBLE_LOOP_SCAN_R(INDEX, START, END, POS, INIT_POS, BOOL_EXPR, CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT)

#define FUSIBLE_LOOP_SCAN_R_END(LENGTH, POS, POS_STORE_DESTINATION) } return 0; }, 1, POS_STORE_DESTINATION); }

#define FUSIBLE_LOOP_SCAN_END(LENGTH, POS, POS_STORE_DESTINATION) FUSIBLE_LOOP_SCAN_R_END(LENGTH, POS, POS_STORE_DESTINATION)

#define FUSIBLE_LOOP_SCAN_PHASE_R(INDEX, START, END, POS, INIT_POS, BOOL_EXPR, PRIORITY, REGISTER_COUNT) \
   _FUSIBLE_LOOP_SCAN_R(FusedActionsObserver::getActiveObserver()->getFusedActions<LOOPFUSER(REGISTER_COUNT)>(PRIORITY), \
                        INDEX, START, END, POS, INIT_POS, BOOL_EXPR, REGISTER_COUNT)

#define FUSIBLE_LOOP_SCAN_PHASE(INDEX, START, END, POS, INIT_POS, BOOL_EXPR, PRIORITY) \
   FUSIBLE_LOOP_SCAN_PHASE_R(INDEX, START, END, POS, INIT_POS, BOOL_EXPR, PRIORITY, CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT)

#define FUSIBLE_LOOP_SCAN_PHASE_END(LENGTH, POS, POS_STORE_DESTINATION) FUSIBLE_LOOP_SCAN_END(LENGTH, POS, POS_STORE_DESTINATION)
#define FUSIBLE_LOOP_SCAN_PHASE_R_END(LENGTH, POS, POS_STORE_DESTINATION) FUSIBLE_LOOP_SCAN_R_END(LENGTH, POS, POS_STORE_DESTINATION)


// note - FUSED_SCANVAR will be nullptr if m_call_as_packed is set in registerAction, as there will be no need for an intermediate
// FUSED_SCANVAR, so we won't need to write to it in the action or store into it in the conditional
#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_R(INDEX,START,END,SCANVAR, REGISTER_COUNT)  { \
   auto __fuser__ = LOOPFUSER(REGISTER_COUNT)::getInstance(); \
   FUSIBLE_BOOKKEEPING(__fuser__, START, END, REGISTER_COUNT); \
   static int __fusible_scan_pos__; \
   __fusible_scan_pos__ = 0; \
   __fuser__->registerAction( FUSIBLE_REGISTER_ARGS, __fusible_scan_pos__, \
                              [=] FUSIBLE_DEVICE(int INDEX, int  *FUSED_SCANVAR , index_type const * SCANVAR_OFFSET, int, FUSIBLE_REGISTERS(REGISTER_COUNT)) {  \
                                 if (FUSED_SCANVAR != nullptr) { \
                                    FUSIBLE_INDEX_ADJUST(INDEX) ; \
                                    int __offset = __fusible_action_index__ == 0 ? 0 : SCANVAR_OFFSET[__fusible_action_index__-1]; \
                                    SCANVAR[INDEX] = FUSED_SCANVAR[__fusible_global_index__] - FUSED_SCANVAR[__offset]; \
                                 } \
                              },  \
                              [=] FUSIBLE_DEVICE(int INDEX, int *FUSED_SCANVAR, FUSIBLE_REGISTERS(REGISTER_COUNT)) { \
                                 FUSIBLE_LOOP_PREAMBLE(INDEX) {

#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_R_END(INDEX, LENGTH, SCANVAR)  \
                                 } \
                                 if (FUSED_SCANVAR != nullptr) { \
                                    FUSED_SCANVAR[__fusible_global_index__] = SCANVAR[INDEX]; \
                                 } \
                                 }, \
                              2, __fusible_scan_pos__ , SCANVAR); }

#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN(INDEX,START,END,SCANVAR) \
   FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_R(INDEX,START,END,SCANVAR,CARE_DEFAULT_LOOP_FUSER_REGISTER_COUNT)

#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_END(INDEX, LENGTH, SCANVAR) \
   FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_R_END(INDEX, LENGTH, SCANVAR)

#else /* CARE_ENABLE_LOOP_FUSER */

#define FUSIBLE_LOOP_STREAM_R(INDEX, START, END, REGISTER_COUNT) CARE_STREAM_LOOP(INDEX, START, END)
#define FUSIBLE_LOOP_STREAM(INDEX, START, END) CARE_STREAM_LOOP(INDEX, START, END)

#define FUSIBLE_LOOP_PHASE_R(INDEX, START, END, PRIORITY, REGISTER_COUNT) CARE_STREAM_LOOP(INDEX, START, END)
#define FUSIBLE_LOOP_PHASE(INDEX, START, END, PRIORITY) CARE_STREAM_LOOP(INDEX, START, END)

#define FUSIBLE_LOOP_PHASE_END CARE_STREAM_LOOP_END
#define FUSIBLE_LOOP_PHASE_R_END CARE_STREAM_LOOP_END

#define FUSIBLE_PHASE_RESET

#define FUSIBLE_KERNEL_R(REGISTER_COUNT) CARE_PARALLEL_KERNEL
#define FUSIBLE_KERNEL CARE_PARALLEL_KERNEL

#define FUSIBLE_KERNEL_PHASE_R(REGISTER_COUNT) CARE_PARALLEL_KERNEL
#define FUSIBLE_KERNEL_PHASE CARE_PARALLEL_KERNEL

#define FUSIBLE_LOOP_STREAM_R_END  CARE_STREAM_LOOP_END
#define FUSIBLE_LOOP_STREAM_END  CARE_STREAM_LOOP_END

#define FUSIBLE_KERNEL_PHASE_R_END CARE_PARALLEL_KERNEL_END
#define FUSIBLE_KERNEL_R_END CARE_PARALLEL_KERNEL_END
#define FUSIBLE_KERNEL_END CARE_PARALLEL_KERNEL_END

#define FUSIBLE_LOOPS_FENCEPOST
#define FUSIBLE_LOOPS_START
#define FUSIBLE_LOOPS_PRESERVE_ORDER_START
#define FUSIBLE_LOOPS_STOP
#define FUSIBLE_LOOPS_STOP_ASYNC

#define FUSIBLE_LOOP_SCAN_R(INDEX, START, END, POS, INIT_POS, BOOL_EXPR, REGISTER_COUNT) SCAN_LOOP(INDEX, START, END, POS, INIT_POS, BOOL_EXPR)
#define FUSIBLE_LOOP_SCAN(INDEX, START, END, POS, INIT_POS, BOOL_EXPR) SCAN_LOOP(INDEX, START, END, POS, INIT_POS, BOOL_EXPR)

#define FUSIBLE_LOOP_SCAN_R_END(LENGTH, POS, POS_STORE_DESTINATION) SCAN_LOOP_END(LENGTH, POS, POS_STORE_DESTINATION)
#define FUSIBLE_LOOP_SCAN_END(LENGTH, POS, POS_STORE_DESTINATION) SCAN_LOOP_END(LENGTH, POS, POS_STORE_DESTINATION)

#define FUSIBLE_LOOP_SCAN_PHASE_R(INDEX, START, END, POS, INIT_POS, BOOL_EXPR, PHASE, REGISTER_COUNT) \
    SCAN_LOOP(INDEX, START, END, POS, INIT_POS, BOOL_EXPR)
#define FUSIBLE_LOOP_SCAN_PHASE(INDEX, START, END, POS, INIT_POS, BOOL_EXPR, PHASE) SCAN_LOOP(INDEX, START, END, POS, INIT_POS, BOOL_EXPR)

#define FUSIBLE_LOOP_SCAN_PHASE_R_END(LENGTH, POS, POS_STORE_DESTINATION) SCAN_LOOP_END(LENGTH, POS, POS_STORE_DESTINATION)
#define FUSIBLE_LOOP_SCAN_PHASE_END(LENGTH, POS, POS_STORE_DESTINATION) SCAN_LOOP_END(LENGTH, POS, POS_STORE_DESTINATION)

#define FUSIBLE_FREE(A) A.free()
#define FUSIBLE_FREE_DEVICE(A) A.freeDeviceMemory()

#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_R(INDX,START,END,SCANVAR, REGISTER_COUNT) SCAN_COUNTS_TO_OFFSETS_LOOP(INDX, START, END, SCANVAR)
#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN(INDX,START,END,SCANVAR) SCAN_COUNTS_TO_OFFSETS_LOOP(INDX, START, END, SCANVAR)

#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_END(INDEX, LENGTH, SCANVAR) SCAN_COUNTS_TO_OFFSETS_LOOP_END(INDEX, LENGTH, SCANVAR)

#endif /* CARE_ENABLE_LOOP_FUSER */


#endif // !defined(_CARE_LOOP_FUSER_H_)
