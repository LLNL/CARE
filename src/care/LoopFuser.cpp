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
#if defined(CARE_GPUCC)
static FusedActionsObserver * defaultObserver = new FusedActionsObserver(allocator(chai::ArrayManager::getInstance()->getAllocator(chai::PINNED))); 
#else
static FusedActionsObserver * defaultObserver = new FusedActionsObserver(allocator(chai::ArrayManager::getInstance()->getAllocator(chai::CPU))); 
#endif

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

CARE_DLL_API LoopFuser::LoopFuser(allocator a) : FusedActions(),
   m_allocator(a),
   m_delay_pack(false),
   m_call_as_packed(true),
   m_max_action_length(0),
   m_reserved(0),
   m_action_offsets(nullptr),
   m_scan_var(nullptr),
   //m_action_starts(nullptr),
   //m_action_ends(nullptr),
   m_conditionals(a),
   m_actions(a), 
   //m_lambda_size(0),
   //m_lambda_data(nullptr),
   m_scan_type(0),
   m_scan_pos_outputs(nullptr),
   m_scan_pos_starts(nullptr),
   m_verbose(false),
   m_reverse_indices(false) {

      // Supports fusing up to 10k loops of average lambda size of 256 bytes
      // will flush if we exceed the 10k count or if the lambda size requirements
      // are exceeded.
      reserve(10*1024);
      //reserve_lambda_buffer(256*10*1024);
}

CARE_DLL_API LoopFuser * LoopFuser::getInstance() {
   static LoopFuser * instance = nullptr;
   if (instance == nullptr) {
      instance = defaultObserver->getFusedActions<LoopFuser>(CARE_DEFAULT_PHASE);
   }
   return instance;
}

CARE_DLL_API LoopFuser::~LoopFuser() {
   warnIfNotFlushed();
   if (m_reserved > 0) {
#if defined(CARE_GPUCC)
      care::gpuFree(m_action_offsets);
#else
      free(m_action_offsets);
#endif
   }
/*
   if (m_lambda_reserved > 0) {
#if defined(CARE_GPUCC)
      care::gpuFree(m_lambda_data);
#else
      free(m_lambda_data);
#endif
   }
*/

   if (m_pos_output_destinations) {
      free(m_pos_output_destinations);
   }
}

void LoopFuser::reserve(size_t size) {
   static char * pinned_buf;
   size_t totalsize = size*(sizeof(int)*3)+sizeof(int *);
   pinned_buf = (char *)m_allocator.allocate(totalsize);
   m_pos_output_destinations = (care::host_ptr<int>*)malloc(size * sizeof(care::host_ptr<int>));

   m_action_offsets   = (int *) pinned_buf;
   /*
   m_action_starts    = (int *) (pinned_buf  +   sizeof(int)*size);
   m_action_ends      = (int *) (pinned_buf  + 2*sizeof(int)*size);
   */
   m_scan_pos_outputs = (int *) (pinned_buf  + sizeof(int)*size);
   m_scan_pos_starts  = (int *) (pinned_buf  + 2*sizeof(int)*size);
   m_scan_var =         (int **)(pinned_buf  + 3*sizeof(int)*size);
   /*
   m_conditionals     = (SerializableDeviceLambda<bool> *)(pinned_buf  + 5*sizeof(int)*size);
   m_actions          = (SerializableDeviceLambda<int> *)(pinned_buf  + 5*sizeof(int)*size + sizeof(SerializableDeviceLambda<bool>)*size);
   */
   m_reserved = size;
}
/*
void LoopFuser::reserve_lambda_buffer(size_t size) {
   // the buffer we will slice out of for packing the lambdas
   m_lambda_reserved = size;
   char * tmp;
#if defined(CARE_GPUCC)
   care::gpuHostAlloc((void**)&tmp, size, gpuHostAllocDefault);
   if (m_lambda_data) {
      care::gpuMemcpy(tmp, m_lambda_data, size, gpuMemcpyHostToHost);
      care::gpuFreeHost((void *)m_lambda_data);
   }
#else
   tmp = (char *) malloc(size);
   if (m_lambda_data) {
      memcpy(tmp, m_lambda_data, size);
      free(m_lambda_data);
   }
#endif
   m_lambda_data = tmp;
}
*/

/* resets lambda_size and m_action_count to 0, keeping our buffers
 * the same */
void LoopFuser::reset(bool async) {
//   m_lambda_size = 0;
   m_action_count = 0;
   m_max_action_length = 0;
   m_prev_pos_output = nullptr;
   m_is_scan = false;
   m_is_counts_to_offsets_scan = false;
   // need to do a synchronize data so the previous fusion data doesn't accidentally
   // get reused for the next one. (Yes, this was a very fun race condition to find).
   if (!async) {
      care::syncIfNeeded();
   }
}

void LoopFuser::warnIfNotFlushed() {
   if (m_action_count > 0) {
      std::cout << (void *)this<<" LoopFuser not flushed when expected." << std::endl;
   }
}

void LoopFuser::flush_parallel_actions(bool async) {
   // Do the thing
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("in flush_parallel_actions with %i,%i\n", m_actions.num_loops(), m_max_action_length);
   }
#endif
   action_workgroup aw = m_actions.instantiate();
   action_worksite aws = aw.run(nullptr);
/*
   int * offsets = m_action_offsets;

   int end = m_action_offsets[m_action_count-1];
   int action_count = m_action_count;
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("launching with end %i and action_count %i\n", end, action_count);
   }
#endif
   bool reverse_indices = m_reverse_indices;
   CARE_STREAM_LOOP(i, 0, end) {
      int index = i;
      if (reverse_indices) {
         // do indices in reverse order to discover any order dependencies between loops
         // and possibly therefore debug GPU race conditions in debug CPU builds of the code.
         index = end-1-i;
      }
      int actionIndex = care::binarySearch<int>(offsets, 0, action_count, index, true);
#ifdef FUSER_VERBOSE
      if (m_verbose) {
         printf("launching action %i with index %i\n", actionIndex, index);
      }
#endif
      actions[actionIndex](index, true, actionIndex, -1, -1);
   } CARE_STREAM_LOOP_END
*/
   if (!async) {
#ifdef FUSER_VERBOSE
      if (m_verbose) {
         printf("syncing \n");
      }
#endif
      care::syncIfNeeded();
   }
}

void LoopFuser::flush_order_preserving_actions(bool // async
                                               ) {
   // Do the thing
   action_workgroup aw = m_actions.instantiate();
   action_ordered_workgroup & aow = reinterpret_cast<action_ordered_workgroup&>(aw);
   action_ordered_worksite aws = aow.run(nullptr);

/*   int action_count = m_action_count;

#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("in flush_order_preserving with %i,%i\n", action_count, m_max_action_length);
   }
#endif

#ifdef FUSER_VERBOSE
   bool verbose = m_verbose;
#endif
   CARE_STREAM_LOOP(i, 0, m_max_action_length) {
      for (int actionIndex = 0; actionIndex < action_count; ++actionIndex) {
#ifdef FUSER_VERBOSE
         if (i == 0 && verbose) {
            printf("calling action %i [%i:%i] at index %i\n", actionIndex, -1, -1, i);
         }
#endif
         actions[actionIndex](i, true, actionIndex, -1, -1);
      }
   } CARE_STREAM_LOOP_END
   */
}


void LoopFuser::flush_parallel_scans() {
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("in flush_parallel_scans with %i,%i\n", m_action_count, m_max_action_length);
   }
#endif
   const int * offsets = (const int *)m_action_offsets;
   int * scan_pos_outputs = m_scan_pos_outputs;
//   int * scan_pos_starts = m_scan_pos_starts;

   int end = m_action_offsets[m_action_count-1];
   int action_count = m_action_count;
   // store the address of the gpu data in the pinned memory where kernels expect to find it
//   *m_scan_var = scan_var.data(chai::GPU,true);

   // handle the last index by enqueuing a specialized lambda to batch with the rest.
   m_conditionals.enqueue(RAJA::RangeSegment(0,1), [=]FUSIBLE_DEVICE(int , int * SCANVAR, int const*, int) {
      SCANVAR[end] = false;
   });

   // the xarg input to the conditional is the bulk scan var the conditional needs to initialize 
   care::host_device_ptr<int> scan_var(end+1, "scan_var");
   // this will fill scan_var up from the fused conditionals 
   conditional_workgroup cw = m_conditionals.instantiate();
   conditional_worksite cws = cw.run(scan_var.data(chai::GPU,true), nullptr, end+1);

   /*
#ifdef FUSER_VERBOSE
   bool verbose = m_verbose;
   if (verbose) {
      CARE_STREAM_LOOP(actionIndex, 0, action_count) {
         printf("offsets[%i] = %i\n", actionIndex, offsets[actionIndex]);
      } CARE_STREAM_LOOP_END
   }
#endif
   bool reverse_indices = m_reverse_indices;
   CARE_STREAM_LOOP(i, 0, end+1) {
      int index = i;
      if (reverse_indices) {
         // do indices in reverse order to discover any order dependencies between loops
         // and possibly therefore debug GPU race conditions in debug CPU builds of the code.
         index = end-i;
      }
      if (index == end) {
         scan_var[index] = 0;
      }
      else {
         int actionIndex = care::binarySearch<int>(offsets, 0, action_count, index, true);
#ifdef FUSER_VERY_VERBOSE
         if (verbose) {
            printf("launching conditional %i with index %i \n", actionIndex, index);
         }
#endif
         scan_var[index] = (index != end) && conditionals[actionIndex](index, true, actionIndex, -1, -1);
#ifdef FUSER_VERBOSE
         if (verbose && scan_var[index] == 1) {
            printf("conditional %i with index %i returned true \n", actionIndex, index);
         }
#endif
      }
   } CARE_STREAM_LOOP_END
   */
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      CARE_STREAM_LOOP(i, 0, end+1) {
         if (scan_var[i] == 1) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } CARE_STREAM_LOOP_END
      printf("SCAN\n");
   }
#endif
   int scanvar_offset = 0;
   exclusive_scan<int, RAJAExec>(scan_var, nullptr, end+1, RAJA::operators::plus<int>{}, scanvar_offset, true);

#ifdef FUSER_VERBOSE
   if (m_verbose) {
      CARE_STREAM_LOOP(i, 1, end+1) {
         if (scan_var[i-1] != scan_var[i]) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } CARE_STREAM_LOOP_END
   }
#endif
   // grab the outputs for the individual scans
   CARE_STREAM_LOOP(i, 0, m_action_count) {
#ifdef FUSER_VERBOSE
      if (verbose) {
         printf("offsets[%i] = %i\n", i, offsets[i]);
      }
#endif
      scan_pos_outputs[i] = scan_var[offsets[i]];
#ifdef FUSER_VERBOSE
      if (verbose) {
         printf("scan_pos_outputs[%i] = %i\n", i, scan_pos_outputs[i]);
      }
#endif
   } CARE_STREAM_LOOP_END
   
   action_workgroup aw = m_actions.instantiate();
   action_worksite aws = aw.run(scan_var.data(chai::GPU,true));

   // execute the loop body
   /*CARE_STREAM_LOOP(i, 0, end) {
      int index = i;
      if (reverse_indices) {
         // do indices in reverse order to discover any order dependencies between loops
         // and possibly therefore debug GPU race conditions in debug CPU builds of the code.
         index = end-1-i;
      }

      int actionIndex = care::binarySearch<int>(offsets, 0, action_count, index, true);

      // find the start Index of the current group of scans
      int startIndex = actionIndex;
      while (scan_pos_starts[startIndex] == -999) {
         --startIndex;
      }
      int scan_pos_start = scan_pos_starts[startIndex];

      int scan_pos_offset = startIndex == 0 ? 0 : scan_pos_outputs[startIndex-1];
      int pos = scan_var[index];
      pos += scan_pos_start - scan_pos_offset;
#ifdef FUSER_VERBOSE
      if (verbose) {
         printf("scan_var[%i] = %i, offset %i start %i startIndex %i\n", index, scan_var[index], scan_pos_offset, scan_pos_start, startIndex);
         printf("launching scan action %i with index %i and pos %i \n", actionIndex, index, pos);
      }
#endif
      actions[actionIndex](index, true, actionIndex, pos, -1);
   } CARE_STREAM_LOOP_END
   */
   // need to do a synchronize data so pinned memory reads are valid
   care::syncIfNeeded();

   /* need to write the scan positions to the output destinations */
   /* each destination is computed */
   CARE_SEQUENTIAL_LOOP(actionIndex, 0, action_count) {
      int scan_pos_offset = actionIndex == 0 ? 0 : scan_pos_outputs[actionIndex-1];
      int pos = scan_pos_outputs[actionIndex];
      pos -= scan_pos_offset;
      *(m_pos_output_destinations[actionIndex].data()) += pos;
   } CARE_SEQUENTIAL_LOOP_END
   scan_var.free();
}
void LoopFuser::flush_parallel_counts_to_offsets_scans(bool async) {
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("in flush_counts_to_offsets_parallel_scans with %i,%i\n", m_action_count, m_max_action_length);
   }
#endif
  const int * offsets = (const int *)m_action_offsets;

  int end = m_action_offsets[m_action_count-1];
  // int action_count = m_action_count;

   care::host_device_ptr<int> scan_var(end, "scan_var");
   *m_scan_var = scan_var.data(chai::GPU, true);
   
   action_workgroup aw = m_actions.instantiate();
   action_worksite aws = aw.run(scan_var.data(chai::GPU, true));
   /*
#ifdef FUSER_VERBOSE
   bool verbose = m_verbose;
   if (verbose) {
      CARE_STREAM_LOOP(actionIndex, 0, action_count) {
         printf("offsets[%i] = %i\n", actionIndex, offsets[actionIndex]);
      } CARE_STREAM_LOOP_END
   }
#endif
   bool reverse_indices = m_reverse_indices;
   CARE_STREAM_LOOP(i, 0, end) {
      int index = i;
      if (reverse_indices) {
         // do indices in reverse order to discover any order dependencies between loops
         // and possibly therefore debug GPU race conditions in debug CPU builds of the code.
         index = end-1-i;
      }
      
      int actionIndex = care::binarySearch<int>(offsets, 0, action_count, index, true);
#ifdef FUSER_VERY_VERBOSE
      if (verbose) {
         printf("launching action %i with index %i \n", actionIndex, index);
      }
#endif
      scan_var[index] = actions[actionIndex](index, true, actionIndex, -1, -1);
#ifdef FUSER_VERY_VERBOSE
      if (verbose ) {
         printf("%i with index %i returned %i \n", actionIndex, index, scan_var[index]);
      }
#endif
      
   } CARE_STREAM_LOOP_END
*/
   exclusive_scan<int, RAJAExec>(scan_var, nullptr, end, RAJA::operators::plus<int>{}, 0, true);

   conditional_workgroup cw = m_conditionals.instantiate();
   conditional_worksite cws = cw.run(scan_var, offsets, end);
/*
   CARE_STREAM_LOOP(i, 0, end) {
      int index = i;
      if (reverse_indices) {
         // do indices in reverse order to discover any order dependencies between loops
         // and possibly therefore debug GPU race conditions in debug CPU builds of the code.
         index = end-1-i;
      }
      
      int actionIndex = care::binarySearch<int>(offsets, 0, action_count, index, true);
#ifdef FUSER_VERY_VERBOSE
      if (verbose) {
         printf("setting scan var using conditional %i with index %i offset %i\n", actionIndex, index, offsets[actionIndex]);
      }
#endif
      int offset = actionIndex == 0 ? 0 : offsets[actionIndex-1];
#ifdef FUSER_VERY_VERBOSE
      if (verbose) {
         printf("new offset %i\n", offset);
         printf("LOOPFUSER::scan_var[%i] = %i\n", i, scan_var[index]-scan_var[offset]);
      }
#endif
      conditionals[actionIndex](index,true,scan_var[index]-scan_var[offset],-1,-1);
   } CARE_STREAM_LOOP_END
   */

   if (!async) {
      // need to do a synchronize data so subsequent writes to this fuser's buffers do not overlap with zero copy reads.
      // If async is on, programmer takes responsibility for ensuring this does not happen.
      care::syncIfNeeded();
   }

   scan_var.free();
}

void LoopFuser::flushActions(bool async) {
#ifdef FUSER_VERBOSE
   printf("Loop fuser flushActions\n");
#endif
   if (m_action_count > 0) {
      if (m_is_scan) {
#ifdef FUSER_VERBOSE
         printf("loop fuser flush parallel scans\n");
#endif
         flush_parallel_scans();
      }
      else if (m_is_counts_to_offsets_scan) {
#ifdef FUSER_VERBOSE
         printf("loop fuser flush counts to offsets scans\n");
#endif
         flush_parallel_counts_to_offsets_scans(async);
      }
      else {
         if (m_preserve_action_order) {
#ifdef FUSER_VERBOSE
            printf("loop fuser flush order preserving actions\n");
#endif
            flush_order_preserving_actions(async);
         }
         else {
#ifdef FUSER_VERBOSE
            printf("loop fuser flush parallel actions\n");
#endif
            flush_parallel_actions(async);
         }
      }
   }
   for (auto arr : m_to_be_freed) {
      arr.free();
   }
   m_to_be_freed.clear();
   reset(async);
}

#endif
