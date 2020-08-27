//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_LOOP_FUSER_H_
#define _CARE_LOOP_FUSER_H_

// CARE config header
#include "care/config.h"

#if CARE_HAVE_LOOP_FUSER

// Other CARE headers
#include "care/care.h"
#include "care/util.h"

// Other library headers
#ifdef CARE_GPUCC
#include "basil/internal/batch_utils.hpp"
#endif

// Std library headers
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
   /// @return true if sorted, false otherwise
   ///////////////////////////////////////////////////////////////////////////
   template <typename T>
   inline bool CheckSorted(const T* array, const int len,
                           const char* name, const char* argname,
                           const bool allowDuplicates = false) {
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
            std::cout << name << " " << argname << " not in ascending order at index " << last + 1 << std::endl;
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

// templated launcher that provides a device method that knows how to deserialize the lambda LB and call it
template <typename ReturnType, typename LB>
FUSIBLE_DEVICE ReturnType launcher(char * lambda_buf, int i, bool is_fused, int action_index, int start, int end);

template <typename ReturnType, typename LB>
FUSIBLE_DEVICE ReturnType launcher(char * lambda_buf, int i, bool is_fused, int action_index, int start, int end) {
   using lambda_type = typename std::decay<LB>::type;
   LB lambda = *reinterpret_cast<lambda_type *> (lambda_buf);
   return lambda(i, is_fused, action_index, start, end);
}

#ifdef CARE_GPUCC
// cuda global function that writes the device wrapper function pointer
// for the template type to the pointer provided.
template<typename ReturnType, typename LB>
__global__ void write_launcher_ptr(basil::detail::device_wrapper_ptr* out)
{
   using lambda_type = typename std::decay<LB>::type;
   auto p = &launcher<ReturnType, lambda_type>;
   *out = (basil::detail::device_wrapper_ptr) p;
}

#endif

// Function that gets and caches the device wrapper function pointer
// for the templated type. A pointer to a device function can only be taken
// in device code, so this launches a kernel to get the pointer. It then holds
// onto the pointer so a kernel doesn't have to be launched to get the
// function pointer the next time.
template<typename ReturnType, typename kernel_type>
inline void * get_launcher_wrapper_ptr(bool get_if_null)
{
#if defined CARE_GPUCC && defined GPU_ACTIVE
   static_assert(alignof(kernel_type) <= sizeof(basil::detail::device_wrapper_ptr),
                 "kernel_type has excessive alignment requirements");
   static basil::detail::device_wrapper_ptr ptr = nullptr;
   if (ptr == nullptr && get_if_null) {
      basil::detail::device_wrapper_ptr* pinned_buf = basil::detail::get_pinned_device_wrapper_ptr_buf();
      gpuStream_t stream = 0;
      void* func = (void*)&write_launcher_ptr<ReturnType, kernel_type>;
      void* args[] = { &pinned_buf };
      care::gpuLaunchKernel(func, 1, 1, args, 0, stream);
      care::gpuStreamSynchronize(stream);
      ptr = *pinned_buf;
   }

   return (void *)ptr;
#else

   using lambda_type = typename std::decay<kernel_type>::type;
   static void * ptr = nullptr;
   if (ptr == nullptr && get_if_null) {
      ptr =(void *)&launcher<ReturnType, lambda_type> ;
   }
   return ptr;
#endif
}

///////////////////////////////////////////////////////////////////////////
/// This class provides a wrapper to a lambda that allows containerization
/// of a group of lambdas that have the same // return type (templated as
/// ReturnType) Current limitation is that the lambda we are serializing must
/// match the call signature of the operator() of this class. In practice
/// this is not a big deal, as we are generally dealing with RAJA idioms,
/// which take the index of loop as a single argument.
///
/// This inherits from care::CARECopyable so that the host_device_ptr<char>
/// containing the serialized lambda gets deep copied along for the ride.
///////////////////////////////////////////////////////////////////////////
template <typename ReturnType>
class SerializableDeviceLambda {
   public:
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Default constructor
      /// @return a SerializableDeviceLambda instance
      ///////////////////////////////////////////////////////////////////////////
      SerializableDeviceLambda() = default;

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Default constructor
      /// @param[in] lambda: The lambda to serialize.
      /// @param[in] buf: The buffer to serialize the lambda into.
      /// @return a SerializableDeviceLambda instance, which will be serialized
      ///         into buf
      ///////////////////////////////////////////////////////////////////////////
      template <typename LB>
      SerializableDeviceLambda(LB && lambda,  char * buf) {
         using lambda_type = typename std::decay<LB>::type;
         m_launcher = (ReturnType (*)(char *, int, bool, int, int, int))get_launcher_wrapper_ptr<ReturnType, lambda_type>(true);
         //size_t size = sizeof(LB);
         m_lambda = buf;
         /* we make a copy of the lambda to trigger chai copy constructors that are required by captured variables in the lambda*/
         void * ptr = (void *) m_lambda;
         chai::ArrayManager::getInstance()->setExecutionSpace(chai::GPU);
         /* use placement new to get good performance on the serialization */
         new (ptr) lambda_type(lambda);
         chai::ArrayManager::getInstance()->setExecutionSpace(chai::NONE);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief constructor from nullptr_t to support the DeviceCopyable interface
      ///////////////////////////////////////////////////////////////////////////
      SerializableDeviceLambda(std::nullptr_t) :  m_lambda{nullptr}, m_launcher{} {}

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief shallowCopy to support the DeviceCopyable interface
      ///////////////////////////////////////////////////////////////////////////
      void shallowCopy(const SerializableDeviceLambda<ReturnType> & other) {
         m_lambda = other.m_lambda;
         m_launcher = other.m_launcher;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief copy constructor
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE
      SerializableDeviceLambda<ReturnType>(const SerializableDeviceLambda<ReturnType> &other)  :
      m_lambda(other.m_lambda), m_launcher(other.m_launcher) {
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief assignment operator
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE
      SerializableDeviceLambda<ReturnType> & operator=(const SerializableDeviceLambda & other) {
         m_lambda = other.m_lambda;
         m_launcher = other.m_launcher;
         return *this;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief operator() wraps the lambda call
      /// @param[in] i - The loop index the lambda will be called at - should be between (start-end]
      /// @param[in] isFused - Whether this lambda is being called in a fused context
      /// @param[in] actionIndex - Which action this lambda is
      /// @param[in] start - The start index this loop is called with
      /// @param[in] end - The end index this loop is called with
      ///////////////////////////////////////////////////////////////////////////
      FUSIBLE_DEVICE ReturnType operator()(int i, bool isFused, int actionIndex, int start, int end) {
         return m_launcher(m_lambda, i, isFused, actionIndex, start, end);
      }

   protected:
      ///
      /// Our lambda buffer
      ///
      char * m_lambda;
      ///
      /// Our launcher method
      ///
      ReturnType (*m_launcher)(char *, int, bool, int, int, int);
};


// This class is meant to orchestrate fusing a bunch of loops together. The initial use case
// is our communication routines. The goal is to do one giant scan at the end over the entire pack
// buffer.
// You register loops' bodies as actions, the start and end of the index set you want for that action,
// and a conditional lambda takes a single argument and returns a boolean if you want the action
// to occur at that index.
// flush() will then orchestrate the scan operation to fuse all of your scans into a single one.
class LoopFuser {
   public:
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief The default constructor. Intentionally am not keeping this private
      ///        in the event that a user wants to maintain multiple independent
      ///        LoopFuser objects.
      ///////////////////////////////////////////////////////////////////////////
      LoopFuser();


      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief The destructor.
      ///////////////////////////////////////////////////////////////////////////
      ~LoopFuser();

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief gets a static singleton instance of a LoopFuser.
      /// @return The singleton instance.
      ///////////////////////////////////////////////////////////////////////////
      CARE_DLL_API static LoopFuser * getInstance();

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief registers a loop lambda with the packer.
      ///////////////////////////////////////////////////////////////////////////
      CARE_DLL_API static int non_scan_store;
      template <typename LB, typename Conditional>
      void registerAction(int start, int end, int & start_pos, Conditional && conditional, LB && action, int scan_type = 0, int & pos_store = non_scan_store, care::host_device_ptr<int> counts_to_offsets_scanvar = nullptr);
      
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief registers an array to be released after a flush()
      ///////////////////////////////////////////////////////////////////////////
      template <typename T>
      void registerFree(care::host_device_ptr<T> & array);

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief starts recording. If recording is stopped, registerAction calls will
      ///        execute the lambda immediately. If recording has started, they will
      ///        be gathered up until flushed either by filling up our buffer or via
      ///        a flush call.
      ///////////////////////////////////////////////////////////////////////////
      void start() {
#ifndef FUSIBLE_LOOPS_DISABLE
         m_delay_pack = true; m_call_as_packed = false; warnIfNotFlushed();
#endif
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief stops recording. If recording is stopped, registerAction calls will
      ///        execute the lambda immediately. If recording has started, they will
      ///        be gathered up until flushed either by filling up our buffer or via
      ///        a flush call.
      ///////////////////////////////////////////////////////////////////////////
      void stop() { m_delay_pack = false; m_call_as_packed = true; }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief execute all recorded actions
      ///////////////////////////////////////////////////////////////////////////
      CARE_DLL_API void flush( );

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief execute all recorded actions in parallel
      ///////////////////////////////////////////////////////////////////////////
      void flush_parallel_actions();

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief execute all recorded actions in a sequence
      ///////////////////////////////////////////////////////////////////////////
      void flush_order_preserving_actions();

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief execute all recorded scans and actions in parallel
      ///////////////////////////////////////////////////////////////////////////
      void flush_parallel_scans();
      
      /// @author Peter Robinson
      /// @brief execute all recorded counts_to_offsets scans and actions in parallel
      ///////////////////////////////////////////////////////////////////////////
      void flush_parallel_counts_to_offsets_scans();

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

      void reset();

      void preserveOrder(bool preserve) { m_preserve_action_order = preserve; }

      void setScan(bool scan) { m_is_scan = scan; }

      void setCountsToOffsetsScan(bool counts_to_offsets_scan) { m_is_counts_to_offsets_scan = counts_to_offsets_scan; }

      int getOffset() {
         if (!m_preserve_action_order) {
            return m_action_count == 0 ? 0 : m_action_offsets[m_action_count-1];
         }
         else {
            return 0;
         }
      }

      void setVerbose(bool verbose) { m_verbose = verbose; }

      void setReverseIndices(bool reverse) { m_reverse_indices = reverse; }


   private:
      ///
      /// warn if not flushed
      ///
      void warnIfNotFlushed();

      ///
      /// whether to delay execution until a flush is called.
      ///
      bool m_delay_pack;
      ///
      /// whether to execute a lambda when a register is called.
      ///
      bool m_call_as_packed;
      ///
      /// whether to preserve execution order when flushing
      ///
      bool m_preserve_action_order;
      ///
      /// the max length of an action's index set
      ///
      int m_max_action_length;
      ///
      /// how many actions (loops) have been registered with m_delay_pack true
      ///
      int m_action_count;

      ///
      /// How many lambdas we are supporting
      ///
      int m_reserved;

      ///
      /// Host pointer (pinned) for action offsets
      ///
      int *m_action_offsets;

      ///
      /// Host pointer (pinned) for action starts
      ///
      int *m_action_starts;

      ///
      /// Host pointer (pinned) for action ends
      ///
      int *m_action_ends;

      ///
      /// container of condional serialized lambdas
      ///
      SerializableDeviceLambda<bool> * m_conditionals;

      ///
      /// container of action serialized lambdas
      ///
      SerializableDeviceLambda<int> * m_actions;

      ///
      /// The amount of memory reserved for lambda serialization
      ///
      size_t m_lambda_reserved;

      ///
      /// The amount of memory used for lambda serialization
      ///
      size_t m_lambda_size;

      ///
      /// The buffer used for lambda serialization
      ///
      char * m_lambda_data;

      ///
      /// Whether or not to flush as a scan
      ///
      bool m_is_scan;
      
      ///
      /// Whether or not to flush as a counts_to_offsets scan
      ///
      bool m_is_counts_to_offsets_scan;

      ///
      /// Type of scan (0 = no scan, 1 = regular scan, 2 = counts_to_offsets scan)
      ///
      int m_scan_type;

      ///
      /// The pinned buffer to store scan position outputs
      ///
      int * m_scan_pos_outputs;

      ///
      /// The pinned buffer for scan pos starts
      ///
      int *m_scan_pos_starts;


      ///
      /// cached scan position output addresses
      ///
      care::host_ptr<int> *m_pos_output_destinations;

      ///
      /// cached output destination so we notice when there's a new one
      ///
      care::host_ptr<int> m_prev_pos_output;

      ///
      /// if compiled with FUSER_VERBOSE and/or FUSER_VERY_VERBOSE, still
      /// do a runtime check for verbosity
      ///
      bool m_verbose;

      ///
      /// runtime control on whether to reverse indices during flush_parallel_actions
      ///
      bool m_reverse_indices = false;

      ///
      /// collection of arrays to be freed after a flush
      ///
      std::vector<care::host_device_ptr<char>> m_to_be_freed;
};


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
template <typename LB, typename Conditional>
void LoopFuser::registerAction(int start, int end, int &start_pos, Conditional && conditional, LB && action, int scan_type, int &pos_store, care::host_device_ptr<int> counts_to_offsets_scanvar) {
   if (end > start) {
      /* switch to scan mode if we encounter a scan before we flush */
      if (scan_type == 1) {
         m_is_scan = true;
         m_is_counts_to_offsets_scan = false;
#ifdef CARE_DEBUG
         if (&start_pos != &pos_store) {
            std::cout << "LoopFuser::registerAction : pos initializer must be same lvalue as pos destination for scans to be fusible" << std::endl;
         }
#endif
      }
      else if (scan_type == 2) {
         m_is_counts_to_offsets_scan = true;
         if (m_is_scan) {
            std::cout << "LoopFuser::registerAction : counts_to_offsets scan is not fusible with normal scans" << std::endl;
         }
         m_is_scan = false;
      }
      if (m_delay_pack) {
#ifdef FUSER_VERBOSE
         if (m_verbose) {
            printf("Registering action %i with start %i and end %i\n", m_action_count, start, end);
         }
#endif
         /* lambdas need to be written to an alligned memory address , or you get runtime errors (that are surprisingly helpful)*/
         if (m_action_count == m_reserved) {
            flush();
         }
#if defined CARE_GPUCC && defined GPU_ACTIVE
         size_t lambda_size = basil::detail::aligned_sizeof<LB, sizeof(basil::detail::device_wrapper_ptr)>::value;
         size_t conditional_size = basil::detail::aligned_sizeof<Conditional, sizeof(basil::detail::device_wrapper_ptr)>::value;
#else
         size_t lambda_size = sizeof(LB);
         size_t conditional_size = sizeof(Conditional);
#endif
         if (m_lambda_reserved <= lambda_size + conditional_size + m_lambda_size) {
            flush();
         }

         m_actions[m_action_count] = SerializableDeviceLambda<int> { action, &m_lambda_data[m_lambda_size]};
         m_lambda_size += lambda_size;
         m_conditionals[m_action_count] = SerializableDeviceLambda<bool> { conditional, &m_lambda_data[m_lambda_size]};
         m_lambda_size += conditional_size;
         m_action_offsets[m_action_count] = m_action_count == 0 ? end-start : m_action_offsets[m_action_count-1] + end -start;
         m_action_starts[m_action_count] = start;
         m_action_ends[m_action_count] = end;
         m_scan_pos_starts[m_action_count] = start_pos;

#ifdef FUSER_VERBOSE
         if (m_verbose) {
            printf("Registered action %i with start %i and end %i and offset %i\n",
                   m_action_count, m_action_starts[m_action_count], m_action_ends[m_action_count],
                   m_action_offsets[m_action_count]);
         }
#endif
         m_max_action_length = std::max(m_max_action_length, end-start);

         // SCAN related variables
         if (m_prev_pos_output == nullptr) {
            // initialize m_prev_pos_output
            m_prev_pos_output = &pos_store;
         }
         else {
            // if we encounter a different output, remember it
            if (m_prev_pos_output != &pos_store) {
               m_prev_pos_output = &pos_store;
            }
            // if we haven't enountered a different output yet, mark this index for continuation
            else if (m_prev_pos_output == &pos_store) {
               // mark the start for continuation
               m_scan_pos_starts[m_action_count] = -999;
            }
         }
         m_pos_output_destinations[m_action_count] = &pos_store;
         ++m_action_count;
      }
      if (m_call_as_packed) {
         switch(scan_type) {
            case 0:
#if defined CARE_GPUCC && defined GPU_ACTIVE
               care::forall(care::raja_fusible {}, 0, end-start, false, -1, action);
#else
               care::forall(care::raja_fusible_seq {}, 0, end-start, false, -1, action);
#endif
               break;
            case 1:
               SCAN_LOOP(i, start, end, pos, start_pos, conditional(i, false, 0, 0, 0)) {
                  action(i, false, 0, pos, -1);
               } SCAN_LOOP_END(end-start, pos, pos_store)
               break;
            case 2:
               SCAN_COUNTS_TO_OFFSETS_LOOP(i, start,end,counts_to_offsets_scanvar) {
                  action(i, false, 0, -1, -1);
               } SCAN_COUNTS_TO_OFFSETS_LOOP_END(i, end-start,counts_to_offsets_scanvar)
               break;
            default:
               printf("care::LoopFuser::encountered unhandled scan type\n");
               break;
         }
      }
   }
}


///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson
/// @brief registers an array to be released after a flush()
/// @param[in] array : the array to be freed after a flush
///////////////////////////////////////////////////////////////////////////
template <typename T>
void LoopFuser::registerFree(care::host_device_ptr<T> & array) {
   m_to_be_freed.push_back(reinterpret_cast<care::host_device_ptr<char> &>(array));
}

#if defined(CARE_DEBUG) || defined(CARE_GPUCC)

// Start recording
#define FUSIBLE_LOOPS_START { \
   auto __fuser__ = LoopFuser::getInstance(); \
   __fuser__->start(); \
   __fuser__->preserveOrder(false); \
   __fuser__->setScan(false); \
}

#define FUSIBLE_LOOPS_PRESERVE_ORDER_START { \
   LoopFuser::getInstance()->start(); \
   LoopFuser::getInstance()->preserveOrder(true); \
}

// Execute, then stop recording
#define FUSIBLE_LOOPS_STOP { \
   LoopFuser::getInstance()->stop(); \
   LoopFuser::getInstance()->flush(); \
}

// frees
#define FUSIBLE_FREE(A) LoopFuser::getInstance()->registerFree(A);

#else // defined(CARE_DEBUG) || defined(CARE_GPUCC)

// in opt, non cuda builds, never start recording
#define FUSIBLE_LOOPS_START { \
   auto __fuser__ = LoopFuser::getInstance(); \
   __fuser__->setScan(false); \
   __fuser__->setCountsToOffsetsScan(false); \
}

#define FUSIBLE_LOOPS_PRESERVE_ORDER_START
#define FUSIBLE_LOOPS_STOP
#define FUSIBLE_FREE(A) A.free();

#endif // defined(CARE_DEBUG) || defined(CARE_GPUCC)

// Loop definitions
#ifdef FUSIBLE_KERNEL_DEBUGGING

#define FUSIBLE_LOOP_STREAM(INDEX, START, END) { \
   auto __fusible_offset__ = LoopFuser::getInstance()->getOffset(); \
   int __fusible_scan_pos__ = 0; \
   auto __fusible_start_index__ = START; \
   auto __fusible_end_index__ = END; \
   LoopFuser::getInstance()->registerAction( \
      __fusible_start_index__, __fusible_end_index__, __fusible_scan_pos__, \
      [=] FUSIBLE_DEVICE(int, bool, int, int, int)->bool { return true; }, \
      [=] FUSIBLE_DEVICE(int INDEX, bool __is_fused__, int __action_index__, int __fuse_start__, int __fuse_end__) ->int{ \
      INDEX += __fusible_start_index__ -  __fusible_offset__ ; \
      if (INDEX < __fusible_end_index__) { \

#define FUSIBLE_KERNEL { \
   int __fusible_scan_pos__ = 0; \
   LoopFuser::getInstance()->registerAction( \
      0, 1, __fusible_scan_pos__, \
      [=] FUSIBLE_DEVICE(int, bool, int, int, int)->bool { return true; }, \
      [=] FUSIBLE_DEVICE(int, bool __is_fused__, int __action_index__, int __fuse_start__, int __fuse_end__) -> int{

#else // FUSIBLE_KERNEL_DEBUGGING

#define FUSIBLE_LOOP_STREAM(INDEX, START, END) { \
   auto __fuser__ = LoopFuser::getInstance(); \
   auto __fusible_offset__ = __fuser__->getOffset(); \
   int __fusible_scan_pos__ = 0; \
   auto __fusible_start_index__ = START; \
   auto __fusible_end_index__ = END; \
   __fuser__->registerAction( \
      __fusible_start_index__, __fusible_end_index__, __fusible_scan_pos__, \
      [=] FUSIBLE_DEVICE(int, bool, int, int, int)->bool { return true; }, \
      [=] FUSIBLE_DEVICE(int INDEX, bool, int, int, int) -> int{ \
      INDEX += __fusible_start_index__ -  __fusible_offset__ ; \
      if (INDEX  < __fusible_end_index__) { \

#define FUSIBLE_LOOP_STREAM_END \
   } return 0;}); }

#define FUSIBLE_KERNEL { \
   int __fusible_scan_pos__ = 0; \
   LoopFuser::getInstance()->registerAction( \
      0, 1, __fusible_scan_pos__, \
      [=] FUSIBLE_DEVICE(int, bool, int, int, int)->bool { return true; }, \
      [=] FUSIBLE_DEVICE(int, bool, int, int, int)->int {
#endif

#define FUSIBLE_KERNEL_END return 0;}); }

#define FUSIBLE_LOOPS_FENCEPOST { \
   int __fusible_action_count__ = LoopFuser::getInstance()->size(); \
   if (__fusible_action_count__ > 0) { \
      std::cout << __FILE__ << "FUSIBLE_FENCEPOST reached before FUSIBLE_LOOPS_STOP occurred!" << std::endl; \
   } \
}

// SCANS
#define FUSIBLE_LOOP_SCAN(INDEX, START, END, POS, INIT_POS, BOOL_EXPR) { \
   auto __fusible_offset__ = LoopFuser::getInstance()->getOffset(); \
   auto __fusible_start_index__ = START; \
   auto __fusible_end_index__ = END; \
   LoopFuser::getInstance()->registerAction( \
      __fusible_start_index__, __fusible_end_index__, INIT_POS, \
      [=] FUSIBLE_DEVICE(int INDEX, bool, int, int, int)->bool { \
         INDEX += __fusible_start_index__ -  __fusible_offset__ ; \
         return BOOL_EXPR; \
      }, \
      [=] FUSIBLE_DEVICE(int INDEX, bool /*__is_fused__*/, int /*__action_index__*/, int POS, int)->int { \
         INDEX += __fusible_start_index__ -  __fusible_offset__ ; \
         if (INDEX < __fusible_end_index__ && BOOL_EXPR) { \

#define FUSIBLE_LOOP_SCAN_END(LENGTH, POS, POS_STORE_DESTINATION) } return 0; }, 1, POS_STORE_DESTINATION); }

#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN(INDEX,START,END,SCANVAR)  { \
   auto __fusible_offset__ = LoopFuser::getInstance()->getOffset(); \
   auto __fusible_start_index__ = START; \
   auto __fusible_end_index__ = END; \
   int __fusible_scan_pos__ = 0; \
   LoopFuser::getInstance()->registerAction( \
      __fusible_start_index__, __fusible_end_index__, __fusible_scan_pos__, \
      [=] FUSIBLE_DEVICE(int INDEX, bool, int VAL, int, int)->bool { SCANVAR[INDEX + __fusible_start_index__ - __fusible_offset__] = VAL; return true; }, \
      [=] FUSIBLE_DEVICE(int INDEX, bool /*__is_fused__*/, int /*__action_index__*/, int , int)->int { \
         INDEX += __fusible_start_index__ -  __fusible_offset__ ; \
         if (INDEX < __fusible_end_index__ ) { \


#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_END(INDEX, LENGTH, SCANVAR)  } return SCANVAR[INDEX];}, 2, __fusible_scan_pos__ , SCANVAR); }
#else /* CARE_HAVE_LOOP_FUSER */

#define FUSIBLE_LOOP_STREAM(INDEX, START, END) CARE_STREAM_LOOP(INDEX, START, END)
#define FUSIBLE_KERNEL CARE_PARALLEL_KERNEL
#define FUSIBLE_LOOP_STREAM_END  CARE_STREAM_LOOP_END
#define FUSIBLE_KERNEL_END CARE_PARALLEL_KERNEL_END
#define FUSIBLE_LOOPS_FENCEPOST
#define FUSIBLE_LOOPS_START
#define FUSIBLE_LOOPS_PRESERVE_ORDER_START
#define FUSIBLE_LOOPS_STOP
#define FUSIBLE_LOOP_SCAN(INDEX, START, END, POS, INIT_POS, BOOL_EXPR) SCAN_LOOP(INDEX, START, END, POS, INIT_POS, BOOL_EXPR)
#define FUSIBLE_LOOP_SCAN_END(LENGTH, POS, POS_STORE_DESTINATION) SCAN_LOOP_END(LENGTH, POS, POS_STORE_DESTINATION)
#define FUSIBLE_FREE(A) A.free()
#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN(INDX,START,END,SCANVAR) SCAN_COUNTS_TO_OFFSETS_LOOP(INDX, START, END, SCANVAR)

#define FUSIBLE_LOOP_COUNTS_TO_OFFSETS_SCAN_END(INDEX, LENGTH, SCANVAR) SCAN_COUNTS_TO_OFFSETS_LOOP_END(INDEX, LENGTH, SCANVAR)

#endif /* CARE_HAVE_LOOP_FUSER */


#endif // !defined(_CARE_LOOP_FUSER_H_)
