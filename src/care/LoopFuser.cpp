//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#define GPU_ACTIVE

// CARE config header
#include "care/config.h"

#if CARE_ENABLE_LOOP_FUSER

// Other CARE headers
#include "care/DefaultMacros.h"
#include "care/LoopFuser.h"
#include "care/scan.h"
#include "care/Setup.h"

CARE_DLL_API int LoopFuser::non_scan_store = 0;
CARE_DLL_API bool LoopFuser::verbose = false;
CARE_DLL_API bool LoopFuser::very_verbose = false;
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

CARE_DLL_API LoopFuser::LoopFuser(allocator a) : FusedActions(),
   m_allocator(a),
   m_max_action_length(0),
   m_reserved(0),
   m_action_offsets(nullptr),
   m_conditionals(a),
   m_actions(a), 
   m_scan_type(0),
   m_scan_pos_outputs(nullptr),
   m_scan_pos_starts(nullptr),
   m_reverse_indices(false) {

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

CARE_DLL_API LoopFuser * LoopFuser::getInstance() {
   static LoopFuser * instance = nullptr;
   if (instance == nullptr) {
      instance = defaultObserver->getFusedActions<LoopFuser>(CARE_DEFAULT_PHASE);
   }
   return instance;
}

void LoopFuser::startRecording() {
   m_recording = true;  warnIfNotFlushed();
}

void LoopFuser::stopRecording() { m_recording = false;  }

CARE_DLL_API LoopFuser::~LoopFuser() {
   warnIfNotFlushed();
   if (m_reserved > 0) {
      m_allocator.free(m_action_offsets);
   }

   if (m_pos_output_destinations) {
      free(m_pos_output_destinations);
   }
}

void LoopFuser::reserve(size_t size) {
   static char * pinned_buf;
   size_t totalsize = size*(sizeof(int)*3);
   pinned_buf = (char *)m_allocator.allocate(totalsize);
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
void LoopFuser::reset(bool async, const char * fileName, int lineNumber) {
   m_action_count = 0;
   m_max_action_length = 0;
   m_prev_pos_output = nullptr;
   m_is_scan = false;
   m_is_counts_to_offsets_scan = false;
   // need to do a synchronize data so the previous fusion data doesn't accidentally
   // get reused for the next one. (Yes, this was a very fun race condition to find).
   if (!async) {
      care::gpuDeviceSynchronize(fileName, lineNumber);
   }
   m_conditionals.reserve(m_reserved, 256*m_reserved);
   m_actions.reserve(m_reserved, 256*m_reserved);
}

void LoopFuser::warnIfNotFlushed() {
   if (m_action_count > 0) {
      std::cout << (void *)this<<" LoopFuser not flushed when expected." << std::endl;
   }
}

void LoopFuser::flush_parallel_actions(bool async, const char * fileName, int lineNumber) {
   // Do the thing
   if (verbose) {
      printf("in flush_parallel_actions at %s:%i with %zu, %i\n", fileName, lineNumber, m_actions.num_loops(), m_max_action_length);
   }
   action_workgroup aw = m_actions.instantiate();
   action_worksite aws = aw.run(nullptr, fusible_registers{});
   // this resets m_conditionals, which we will never need to run
   m_conditionals.clear();
   if (verbose) {
      printf("done with flush_parallel_actions at %s:%i with %zu, %i, async %i\n", fileName, lineNumber, m_actions.num_loops(), m_max_action_length, (int) async);
   }
   reset(async, fileName, lineNumber);
}

void LoopFuser::flush_order_preserving_actions(bool async, const char * fileName, int  lineNumber) {
   // Do the thing
   action_workgroup aw = m_actions.instantiate();
   action_worksite aws = aw.run(nullptr, fusible_registers{});
   // this resets m_conditionals, don't run them.
   m_conditionals.clear();
   reset(async, fileName, lineNumber);
}


void LoopFuser::flush_parallel_scans(const char * fileName, int lineNumber) {
   if (verbose) {
      printf("in flush_parallel_scans at %s:%i with %i,%i\n", fileName, lineNumber, m_action_count, m_max_action_length);
   }
   const int * offsets = (const int *)m_action_offsets;
   int * scan_pos_outputs = m_scan_pos_outputs;

   int end = m_action_offsets[m_action_count-1];
   int action_count = m_action_count;

   // handle the last index by enqueuing a specialized lambda to batch with the rest.
   m_conditionals.enqueue(RAJA::RangeSegment(0,1), [=]FUSIBLE_DEVICE(int , int * SCANVAR, int const*, int, fusible_registers) {
      SCANVAR[end] = false;
   });

   // the xarg input to the conditional is the bulk scan var the conditional needs to initialize 
   care::host_device_ptr<int> scan_var(end+1, "scan_var");
   // this will fill scan_var up from the fused conditionals 
   conditional_workgroup cw = m_conditionals.instantiate();
   conditional_worksite cws = cw.run(scan_var.data(chai::GPU,true), nullptr, end+1, fusible_registers{});

   if (very_verbose) {
      CARE_SEQUENTIAL_LOOP(i, 0, end+1) {
         if (scan_var[i] == 1) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } CARE_SEQUENTIAL_LOOP_END
      printf("SCAN\n");
   }
   int scanvar_offset = 0;
   exclusive_scan<int, RAJAExec>(scan_var, nullptr, end+1, RAJA::operators::plus<int>{}, scanvar_offset, true);

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
   action_workgroup aw = m_actions.instantiate();
   action_worksite aws = aw.run(scan_var.data(chai::GPU,true), fusible_registers{});

   // need to do a synchronize data so pinned memory reads are valid
   care::gpuDeviceSynchronize(fileName,lineNumber);

   /* need to write the scan positions to the output destinations */
   /* each destination is computed */
   CARE_SEQUENTIAL_LOOP(actionIndex, 0, action_count) {
      int scan_pos_offset = actionIndex == 0 ? 0 : scan_pos_outputs[actionIndex-1];
      int pos = scan_pos_outputs[actionIndex];
      pos -= scan_pos_offset;
      *(m_pos_output_destinations[actionIndex].data()) += pos;
   } CARE_SEQUENTIAL_LOOP_END
   scan_var.free();

   if (verbose) {
      printf("done with flush_parallel_scans at %s:%i with %i,%i\n", fileName, lineNumber, m_action_count, m_max_action_length);
   }
   // async is true because we just synchronized
   reset(true, fileName, lineNumber);
}


void LoopFuser::flush_parallel_counts_to_offsets_scans(bool async, const char * fileName, int lineNumber) {
   if (verbose) {
     printf("in flush_counts_to_offsets_parallel_scans at %s:%i with %i,%i\n", fileName,lineNumber, m_action_count, m_max_action_length);
   }
   const int * offsets = (const int *)m_action_offsets;

   int end = m_action_offsets[m_action_count-1];

   care::host_device_ptr<int> scan_var(end, "scan_var");
   
   action_workgroup aw = m_actions.instantiate();
   action_worksite aws = aw.run(scan_var.data(chai::GPU, true), fusible_registers{});
   
   if (very_verbose) {
      CARE_SEQUENTIAL_LOOP(i, 0, end) {
         if (scan_var[i] == 1) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } CARE_SEQUENTIAL_LOOP_END
      printf("SCAN TO OFFSETS\n");
   }
   exclusive_scan<int, RAJAExec>(scan_var, nullptr, end, RAJA::operators::plus<int>{}, 0, true);
   if (very_verbose) {
      CARE_SEQUENTIAL_LOOP(i, 1, end) {
         if (scan_var[i-1] != scan_var[i]) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } CARE_SEQUENTIAL_LOOP_END
   }

   conditional_workgroup cw = m_conditionals.instantiate();
   conditional_worksite cws = cw.run(scan_var.data(chai::GPU,true), offsets, end, fusible_registers{});


   scan_var.free();
   if (verbose) {
     printf("done with flush_counts_to_offsets_parallel_scans at %s:%i with %i,%i\n", fileName,lineNumber, m_action_count, m_max_action_length);
   }
   reset(async, fileName, lineNumber);
}

void LoopFuser::flushActions(bool async, const char * fileName, int lineNumber) {
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
   for (auto arr : m_to_be_freed) {
      arr.free();
   }
   m_to_be_freed.clear();
}

#endif
