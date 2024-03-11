//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#define GPU_ACTIVE

// CARE config header
#include "care/config.h"

#if CARE_HAVE_LOOP_FUSER

// Other CARE headers
#include "care/care.h"
#include "care/util.h"
#include "care/LoopFuser.h"

CARE_DLL_API int LoopFuser::non_scan_store = 0;

LoopFuser::LoopFuser() :
   m_delay_pack(false),
   m_call_as_packed(true),
   m_preserve_action_order(false),
   m_max_action_length(0),
   m_action_count(0),
   m_reserved(0),
   m_action_offsets(nullptr),
   m_action_starts(nullptr),
   m_action_ends(nullptr),
   m_conditionals(nullptr),
   m_actions(nullptr),
   m_lambda_reserved(0),
   m_lambda_size(0),
   m_lambda_data(nullptr),
   m_is_scan(false),
   m_is_counts_to_offsets_scan(false),
   m_scan_pos_outputs(nullptr),
   m_scan_pos_starts(nullptr),
   m_verbose(false),
   m_reverse_indices(false) {
}

LoopFuser * LoopFuser::getInstance() {
   static LoopFuser * instance = nullptr;
   if (instance == nullptr) {
      instance = new LoopFuser();

      // Supports fusing up to 1M loops of average lambda size of 256 bytes
      // will flush if we exceed the 1M count or if the lambda size requirements
      // are exceeded.
      instance->reserve(1024*1024);
      instance->reserve_lambda_buffer(256*1024*1024);
   }
   return instance;
}

LoopFuser::~LoopFuser() {
   warnIfNotFlushed();
   if (m_reserved > 0) {
#ifdef __CUDACC__
      cudaFree(m_action_offsets);
#else
      free(m_action_offsets);
#endif
   }

   if (m_lambda_reserved > 0) {
#ifdef __CUDACC__
      cudaFree(m_lambda_data);
#else
      free(m_lambda_data);
#endif
   }

   if (m_pos_output_destinations) {
      free(m_pos_output_destinations);
   }
}

void LoopFuser::reserve(size_t size) {
   static char * pinned_buf;
   size_t totalsize = size*(sizeof(int)*5+sizeof(SerializableDeviceLambda<int>) + sizeof(SerializableDeviceLambda<bool>));
#ifdef __CUDACC__
   cudaHostAlloc((void **)&pinned_buf, totalsize, cudaHostAllocDefault);
#else
   pinned_buf = (char*) malloc(totalsize);
#endif
   m_pos_output_destinations = (care::host_ptr<int>*)malloc(size * sizeof(care::host_ptr<int>));

   m_action_offsets   = (int *) pinned_buf;
   m_action_starts    = (int *)(pinned_buf  +   sizeof(int)*size);
   m_action_ends      = (int *)(pinned_buf  + 2*sizeof(int)*size);
   m_scan_pos_outputs = (int *)(pinned_buf  + 3*sizeof(int)*size);
   m_scan_pos_starts  = (int *)(pinned_buf  + 4*sizeof(int)*size);
   m_conditionals     = (SerializableDeviceLambda<bool> *)(pinned_buf  + 5*sizeof(int)*size);
   m_actions          = (SerializableDeviceLambda<int> *)(pinned_buf  + 5*sizeof(int)*size + sizeof(SerializableDeviceLambda<bool>)*size);
   m_reserved = size;
}

void LoopFuser::reserve_lambda_buffer(size_t size) {
   /* the buffer we will slice out of for packing the lambdas */
   m_lambda_reserved = size;
   char * tmp;
#ifdef __CUDACC__
   cudaHostAlloc((void**)&tmp, size, cudaHostAllocDefault);
   if (m_lambda_data) {
      cudaMemcpy(tmp, m_lambda_data, size, cudaMemcpyHostToHost);
      cudaFreeHost((void *)m_lambda_data);
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

/* resets lambda_size and m_action_count to 0, keeping our buffers
 * the same */
void LoopFuser::reset() {
   m_lambda_size = 0;
   m_action_count = 0;
   m_max_action_length = 0;
   m_prev_pos_output = nullptr;
   m_is_scan = false;
   m_is_counts_to_offsets_scan = false;
   // need to do a synchronize data so the previous fusion data doesn't accidentally
   // get reused for the next one. (Yes, this was a very fun race condition to find).
   care::syncIfNeeded();
}

void LoopFuser::warnIfNotFlushed() {
   if (m_action_count > 0) {
      std::cout << "LoopFuser not flushed when expected." << std::endl;
   }
}

void LoopFuser::flush_parallel_actions() {
   // Do the thing
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("in flush_parallel_actions with %i,%i\n", m_action_count, m_max_action_length);
   }
#endif
   SerializableDeviceLambda<int> *actions = m_actions;
   int * offsets = m_action_offsets;

   int end = m_action_offsets[m_action_count-1];
   int action_count = m_action_count;
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("launching with end %i and action_count %i\n", end, action_count);
   }
#endif
   bool reverse_indices = m_reverse_indices;
   LOOP_STREAM(i, 0, end) {
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
   } LOOP_STREAM_END
   care::syncIfNeeded();
}

void LoopFuser::flush_order_preserving_actions() {
   // Do the thing
   SerializableDeviceLambda<int> *actions = m_actions;

   int action_count = m_action_count;

#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("in flush_order_preserving with %i,%i\n", action_count, m_max_action_length);
   }
#endif

#ifdef FUSER_VERBOSE
   bool verbose = m_verbose;
#endif
   LOOP_STREAM(i, 0, m_max_action_length) {
      for (int actionIndex = 0; actionIndex < action_count; ++actionIndex) {
#ifdef FUSER_VERBOSE
         if (i == 0 && verbose) {
            printf("calling action %i [%i:%i] at index %i\n", actionIndex, -1, -1, i);
         }
#endif
         actions[actionIndex](i, true, actionIndex, -1, -1);
      }
   } LOOP_STREAM_END
}

void LoopFuser::flush_parallel_scans() {
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("in flush_parallel_scans with %i,%i\n", m_action_count, m_max_action_length);
   }
#endif
   SerializableDeviceLambda<int> *actions = m_actions;
   SerializableDeviceLambda<bool> *conditionals = m_conditionals;
   const int * offsets = (const int *)m_action_offsets;
   int * scan_pos_outputs = m_scan_pos_outputs;
   int * scan_pos_starts = m_scan_pos_starts;

   int end = m_action_offsets[m_action_count-1];
   int action_count = m_action_count;

   care::host_device_ptr<int> scan_var(end+1, "scan_var");
#ifdef FUSER_VERBOSE
   bool verbose = m_verbose;
   if (verbose) {
      LOOP_STREAM(actionIndex, 0, action_count) {
         printf("offsets[%i] = %i\n", actionIndex, offsets[actionIndex]);
      } LOOP_STREAM_END
   }
#endif
   bool reverse_indices = m_reverse_indices;
   LOOP_STREAM(i, 0, end+1) {
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
   } LOOP_STREAM_END
   int scanvar_offset = 0;
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      LOOP_STREAM(i, 0, end+1) {
         if (scan_var[i] == 1) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } LOOP_STREAM_END
      printf("SCAN\n");
   }
#endif
   exclusive_scan<int, RAJAExec>(scan_var, nullptr, end+1, RAJA::operators::plus<int>{}, scanvar_offset, true);

#ifdef FUSER_VERBOSE
   if (m_verbose) {
      LOOP_STREAM(i, 1, end+1) {
         if (scan_var[i-1] != scan_var[i]) {
            printf("scan_var[%i] = %i\n", i, scan_var[i]);
         }
      } LOOP_STREAM_END
   }
#endif
   // grab the outputs for the individual scans
   LOOP_STREAM(i, 0, m_action_count) {
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
   } LOOP_STREAM_END

   // execute the loop body
   LOOP_STREAM(i, 0, end) {
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
   } LOOP_STREAM_END
   // need to do a synchronize data so pinned memory reads are valid
   care::syncIfNeeded();

   /* need to write the scan positions to the output destinations */
   /* each destination is computed */
   LOOP_SEQUENTIAL(actionIndex, 0, action_count) {
      int scan_pos_offset = actionIndex == 0 ? 0 : scan_pos_outputs[actionIndex-1];
      int pos = scan_pos_outputs[actionIndex];
      pos -= scan_pos_offset;
      *(m_pos_output_destinations[actionIndex]) += pos;
   } LOOP_SEQUENTIAL_END
   scan_var.free();
}
void LoopFuser::flush_parallel_counts_to_offsets_scans() {
#ifdef FUSER_VERBOSE
   if (m_verbose) {
      printf("in flush_counts_to_offsets_parallel_scans with %i,%i\n", m_action_count, m_max_action_length);
   }
#endif
   SerializableDeviceLambda<int> *actions = m_actions;
   SerializableDeviceLambda<bool> *conditionals = m_conditionals;
   const int * offsets = (const int *)m_action_offsets;

   int end = m_action_offsets[m_action_count-1];
   int action_count = m_action_count;

   care::host_device_ptr<int> scan_var(end, "scan_var");
#ifdef FUSER_VERBOSE
   bool verbose = m_verbose;
   if (verbose) {
      LOOP_STREAM(actionIndex, 0, action_count) {
         printf("offsets[%i] = %i\n", actionIndex, offsets[actionIndex]);
      } LOOP_STREAM_END
   }
#endif
   bool reverse_indices = m_reverse_indices;
   LOOP_STREAM(i, 0, end) {
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
      
   } LOOP_STREAM_END

   exclusive_scan<int, RAJAExec>(scan_var, nullptr, end, RAJA::operators::plus<int>{}, 0, true);

   LOOP_STREAM(i, 0, end) {
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
   } LOOP_STREAM_END

   // need to do a synchronize data so pinned memory reads are valid
   care::syncIfNeeded();

   scan_var.free();
}

void LoopFuser::flush() {
   if (m_action_count > 0) {
      if (m_is_scan) {
         flush_parallel_scans();
      }
      else if (m_is_counts_to_offsets_scan) {
         flush_parallel_counts_to_offsets_scans();
      }
      else {
         if (m_preserve_action_order) {
            flush_order_preserving_actions();
         }
         else {
            flush_parallel_actions();
         }
      }
   }
   for (auto arr : m_to_be_freed) {
      arr.free();
   }
   m_to_be_freed.clear();
   reset();
}

#endif
