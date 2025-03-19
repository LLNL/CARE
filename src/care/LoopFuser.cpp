//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// Loop Fuser uses the CUDA/HIP default stream and wants to enqueue events in the default stream
#define CAMP_USE_PLATFORM_DEFAULT_STREAM 1

#include "umpire/Allocator.hpp"
#include "umpire/TypedAllocator.hpp"

// CARE config header
#include "care/config.h"

#if CARE_ENABLE_LOOP_FUSER

// Other CARE headers
#include "care/DefaultMacros.h"
#include "care/LoopFuser.h"
#include "care/scan.h"
#include "care/Setup.h"

CARE_DLL_API int FusedActions::non_scan_store = 0;
CARE_DLL_API bool FusedActions::verbose = false;
CARE_DLL_API bool FusedActions::very_verbose = false;
// set flush length to 8M by default (default value is defined in CMakeLists.txt)
CARE_DLL_API int FusedActions::flush_length = CARE_LOOP_FUSER_FLUSH_LENGTH; 
CARE_DLL_API bool FusedActions::flush_now = false;

CARE_DLL_API std::vector<FusedActionsObserver *> FusedActionsObserver::allObservers{};

static FusedActionsObserver * defaultObserver = new FusedActionsObserver();

CARE_DLL_API FusedActionsObserver * FusedActionsObserver::activeObserver = nullptr;

CARE_DLL_API FusedActionsObserver * FusedActionsObserver::getActiveObserver() {
   if (activeObserver == nullptr) {
      activeObserver = defaultObserver;
   }
   return activeObserver;
}

CARE_DLL_API  void FusedActionsObserver::setActiveObserver(FusedActionsObserver * observer) {
   activeObserver = observer;
}

CARE_DLL_API void FusedActionsObserver::cleanupAllFusedActions() {
   for (FusedActionsObserver * observer : FusedActionsObserver::allObservers) {
      observer->reset(false, __FILE__, __LINE__);
      delete observer;
   }
   allObservers.clear();
}

template<int REGISTER_COUNT, typename...XARGS>
CARE_DLL_API LoopFuser<REGISTER_COUNT,XARGS...>::LoopFuser(allocator a) : FusedActions(),
   m_allocator(a),
   m_max_action_length(0),
   m_reserved(0),
   m_totalsize(0),
   m_action_offsets(nullptr),
   m_conditionals(a),
   m_cw(m_conditionals.instantiate()),
   m_cws(m_cw.run(nullptr, nullptr, -1, XARGS{}...)),
   m_actions(a),
   m_aw(m_actions.instantiate()),
   m_aws(m_aw.run(nullptr, XARGS{}...)),
   m_scan_type(0),
   m_scan_pos_outputs(nullptr),
   m_scan_pos_starts(nullptr),
   m_reverse_indices(false),
   m_async_resource(RAJA::resources::get_default_resource<RAJADeviceExec>()),
   m_wait_for_event(),
   m_wait_needed(false) {

      // Supports fusing up to 10k loops of average lambda size of 256 bytes
      // will flush if we exceed the 10k count or if the lambda size requirements
      // are exceeded.
      // Note that while Workpools can grow their internal allocations dynamically, in many
      // cases when CHAI is involved this enforces a more strict requirement on data persistency
      // while loops are being fused by the application, and causes later time chai data movement
      // checks that may be undesired. The hope here is that we've allocated enough to prevent need
      // for growth. 
      // TODO: pipe in num_loops as a variable to constructor / allow macro to define the reserved
      // size. 
      reserve(10*1024);
}

template<int REGISTER_COUNT, typename...XARGS>
CARE_DLL_API LoopFuser<REGISTER_COUNT,XARGS...> * LoopFuser<REGISTER_COUNT,XARGS...>::getInstance() {
   static LoopFuser<REGISTER_COUNT,XARGS...> * instance = nullptr;
   if (instance == nullptr) {
      instance = defaultObserver->getFusedActions<LoopFuser<REGISTER_COUNT, XARGS...>>(CARE_DEFAULT_PHASE);
   }
   return instance;
}

template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::startRecording(bool warn) {
   if (FusedActions::flush_length > 1) {
      m_recording = true;
      if (warn) {
         warnIfNotFlushed();
      }
   }
}

template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::stopRecording() { m_recording = false;  }

template<int REGISTER_COUNT, typename...XARGS>
CARE_DLL_API LoopFuser<REGISTER_COUNT,XARGS...>::~LoopFuser() {
   warnIfNotFlushed();
   if (m_reserved > 0) {
      m_allocator.deallocate((char *)m_action_offsets, m_totalsize);
   }

   if (m_pos_output_destinations) {
      free(m_pos_output_destinations);
   }
}

template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::reserve(size_t size) {
   static char * pinned_buf;
   m_totalsize = size*(sizeof(int)*3);
   pinned_buf = (char *)m_allocator.allocate(m_totalsize);
   m_pos_output_destinations = (care::host_ptr<int>*)malloc(size * sizeof(care::host_ptr<int>));

   m_action_offsets   = (int *) pinned_buf;
   m_scan_pos_outputs = (int *) (pinned_buf  + sizeof(int)*size);
   m_scan_pos_starts  = (int *) (pinned_buf  + 2*sizeof(int)*size);
   m_reserved = size;
   int const bytes_per_lambda = 256;
   m_conditionals.reserve(size,bytes_per_lambda*size);
   m_actions.reserve(size,bytes_per_lambda*size);
}

/* resets lambda_size and m_action_count to 0, keeping our buffers
 * the same */
template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::reset(bool async, const char * fileName, int lineNumber) {
   m_action_count = 0;
   m_max_action_length = 0;
   m_prev_pos_output = nullptr;
   m_is_scan = false;
   m_is_counts_to_offsets_scan = false;
   // need to do a synchronize data so the previous fusion data doesn't accidentally
   // get reused for the next one. (Yes, this was a very fun race condition to find).
   if (!async) {
      care::gpuDeviceSynchronize(fileName, lineNumber);
      // clear out the workgroups and worksites now that their work is done
      m_aw.clear();
      m_cw.clear();
      m_aws.clear();
      m_cws.clear();
   }
   else {
      m_wait_for_event = RAJA::resources::EventProxy<StreamResource>(m_async_resource);
      m_wait_needed = true;
   }
   m_conditionals.reserve(m_reserved, 256*m_reserved);
   m_actions.reserve(m_reserved, 256*m_reserved);
}

/*ensures any previous asynchronous launch from this fuser is done before proceeding. */
template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::waitIfNeeded() {
   if (m_wait_needed) {
      // ensure asynchronous launch from previous flush is done
      m_async_resource.wait_for(&m_wait_for_event);
      // clear out our worksites now that their work is done
      m_aw.clear();
      m_cw.clear();
      m_aws.clear();
      m_cws.clear();
      m_wait_needed = false;
   }
}

template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::warnIfNotFlushed() {
   if (m_action_count > 0) {
      std::cout << (void *)this<<" LoopFuser<"<< REGISTER_COUNT <<"> not flushed when expected." << std::endl;
   }
}

template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::flush_parallel_actions(bool async, const char * fileName, int lineNumber) {
   // Do the thing
   if (verbose) {
      printf("in flush_parallel_actions at %s:%i with %zu, %i\n", fileName, lineNumber, m_actions.num_loops(), m_max_action_length);
   }
   m_aw = m_actions.instantiate();
   m_aws = m_aw.run(nullptr, XARGS{}...);
   // this resets m_conditionals, which we will never need to run
   m_conditionals.clear();
   if (verbose) {
      printf("done with flush_parallel_actions at %s:%i with %zu, %i, async %i\n", fileName, lineNumber, m_actions.num_loops(), m_max_action_length, (int) async);
   }
   reset(async, fileName, lineNumber);
}


template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::flush_order_preserving_actions(bool async, const char * fileName, int  lineNumber) {
   // Do the thing
   m_aw = m_actions.instantiate();
   m_aws = m_aw.run(nullptr, XARGS{}...);
   // this resets m_conditionals, don't run them.
   m_conditionals.clear();
   reset(async, fileName, lineNumber);
}


template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::flush_parallel_scans(const char * fileName, int lineNumber) {
   if (verbose) {
      printf("in flush_parallel_scans at %s:%i with %i,%i\n", fileName, lineNumber, m_action_count, m_max_action_length);
   }
   const int * offsets = (const int *)m_action_offsets;
   int * scan_pos_outputs = m_scan_pos_outputs;

   int end = m_action_offsets[m_action_count-1];
   int action_count = m_action_count;

   // handle the last index by enqueuing a specialized lambda to batch with the rest.
   m_conditionals.enqueue(RAJA::RangeSegment(0,1), [=]FUSIBLE_DEVICE(int , int * SCANVAR, int const*, int, XARGS...) {
      SCANVAR[end] = false;
   });

   // the xarg input to the conditional is the bulk scan var the conditional needs to initialize 
   care::host_device_ptr<int> scan_var(end+1, "scan_var");
   // this will fill scan_var up from the fused conditionals 
   m_cw = m_conditionals.instantiate();
   m_cws = m_cw.run(scan_var.data(chai::GPU,true), nullptr, end+1, XARGS{}...);

   if (very_verbose) {
      CARE_SEQUENTIAL_LOOP(i, 0, end+1) {
         if (scan_var[i] == 1) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } CARE_SEQUENTIAL_LOOP_END
      printf("SCAN\n");
   }
   int scanvar_offset = 0;
   care::exclusive_scan(RAJAExec{}, scan_var, nullptr, end+1, scanvar_offset, true);

   if (very_verbose) {
      CARE_SEQUENTIAL_LOOP(i, 1, end+1) {
         if (scan_var[i-1] != scan_var[i]) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } CARE_SEQUENTIAL_LOOP_END
   }
   if (verbose) {
      CARE_SEQUENTIAL_LOOP(i, 0, m_action_count) {
         printf("offsets[%i] = %i\n", i, offsets[i]);
      } CARE_SEQUENTIAL_LOOP_END
   }
   
   // grab the outputs for the individual scans
   CARE_STREAM_LOOP(i,0,m_action_count) {
      scan_pos_outputs[i] = scan_var[offsets[i]];
   } CARE_STREAM_LOOP_END

   if (verbose) {
      care::gpuDeviceSynchronize(fileName, lineNumber);
      CARE_SEQUENTIAL_LOOP(i,0,m_action_count) {
         printf("scan_pos_outputs[%i] = %i\n", i, scan_pos_outputs[i]);
      } CARE_SEQUENTIAL_LOOP_END
   }
   
   // execute the loop body
   m_aw = m_actions.instantiate();
   m_aws = m_aw.run(scan_var.data(chai::GPU,true), XARGS{}...);

   // need to do a synchronize data so pinned memory reads are valid
   care::gpuDeviceSynchronize(fileName,lineNumber);

   /* need to write the scan positions to the output destinations */
   /* each destination is computed */
   CARE_SEQUENTIAL_LOOP(actionIndex, 0, action_count) {
      int scan_pos_offset = actionIndex == 0 ? 0 : scan_pos_outputs[actionIndex-1];
      int pos = scan_pos_outputs[actionIndex];
      pos -= scan_pos_offset;
      *(m_pos_output_destinations[actionIndex].data()) += pos;
      if (very_verbose) {
         printf("actionIndex %i: scan_pos_offset %i scan_pos_output %i pos %i store %i \n",
                 actionIndex, scan_pos_offset, scan_pos_outputs[actionIndex], pos, *(m_pos_output_destinations[actionIndex].data()));
      }
   } CARE_SEQUENTIAL_LOOP_END
   scan_var.free();

   if (verbose) {
      printf("done with flush_parallel_scans at %s:%i with %i,%i\n", fileName, lineNumber, m_action_count, m_max_action_length);
   }
   // async is true because we just synchronized
   reset(true, fileName, lineNumber);
}


template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::flush_parallel_counts_to_offsets_scans(bool async, const char * fileName, int lineNumber) {
   if (verbose) {
     printf("in flush_counts_to_offsets_parallel_scans at %s:%i with %i,%i\n", fileName,lineNumber, m_action_count, m_max_action_length);
   }
   const int * offsets = (const int *)m_action_offsets;

   int end = m_action_offsets[m_action_count-1];

   care::host_device_ptr<int> scan_var(end, "scan_var");
   
   m_aw = m_actions.instantiate();
   m_aws = m_aw.run(scan_var.data(chai::GPU, true), XARGS{}...);
   
   if (very_verbose) {
      CARE_SEQUENTIAL_LOOP(i, 0, end) {
         if (scan_var[i] == 1) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } CARE_SEQUENTIAL_LOOP_END
      printf("SCAN TO OFFSETS\n");
   }
   care::exclusive_scan(RAJAExec{}, scan_var, nullptr, end, 0, true);
   if (very_verbose) {
      CARE_SEQUENTIAL_LOOP(i, 1, end) {
         if (scan_var[i-1] != scan_var[i]) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } CARE_SEQUENTIAL_LOOP_END
   }

   m_cw = m_conditionals.instantiate();
   m_cws = m_cw.run(scan_var.data(chai::GPU,true), offsets, end, XARGS{}...);


   scan_var.free();
   if (verbose) {
     printf("done with flush_counts_to_offsets_parallel_scans at %s:%i with %i,%i\n", fileName,lineNumber, m_action_count, m_max_action_length);
   }
   reset(async, fileName, lineNumber);
}

template<int REGISTER_COUNT, typename...XARGS>
void LoopFuser<REGISTER_COUNT,XARGS...>::flushActions(bool async, const char * fileName, int lineNumber) {
   if (verbose) {
      printf("Loop fuser flushActions\n");
   }
   if (m_action_count > 0) {
      if (m_is_scan) {
         if (verbose) {
            printf("loop fuser flush parallel scans\n");
         }
         flush_parallel_scans(fileName, lineNumber);
      }
      else if (m_is_counts_to_offsets_scan) {
         if (verbose) {
            printf("loop fuser flush counts to offsets scans\n");
         }
         flush_parallel_counts_to_offsets_scans(async, fileName, lineNumber);
      }
      else {
         if (m_preserve_action_order) {
            if (verbose) {
               printf("loop fuser flush order preserving actions\n");
            }
            flush_order_preserving_actions(async, fileName, lineNumber);
         }
         else {
            if (verbose) {
               printf("loop fuser flush parallel actions\n");
            }
            flush_parallel_actions(async, fileName, lineNumber);
         }
      }
   }
}

#ifdef CARE_ENABLE_FUSER_BIN_32
template class LoopFuser<32, FUSIBLE_REGISTERS(32)>; 
#endif
template class LoopFuser<64, FUSIBLE_REGISTERS(64)>; 
template class LoopFuser<128, FUSIBLE_REGISTERS(128)>; 
template class LoopFuser<256, FUSIBLE_REGISTERS(256)>; 

#endif
