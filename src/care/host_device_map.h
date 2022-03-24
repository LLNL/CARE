//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2022 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////
#ifndef _CARE_HOST_DEVICE_MAP_H
#define _CARE_HOST_DEVICE_MAP_H
#include "care/config.h"
#include "care/atomic.h"

#include <map>
#include <care/KeyValueSorter.h>

namespace care {

   ///
   /// @author Peter Robinson
   ///
   /// @class host_device_map is a rudimentary associative map. On the host,
   /// it is backed by a std::map, and on the device it uses a
   /// care::KeyValueSorter.
   ///
   /// Restrictions when compared to std::map:
   ///   1. Total capacity must be declared up front.
   ///   2. a call to sort() must be performed before lookups will find elements inserted by any emplace calls since
   ///      the last sort() call.
   ///   3. size() will return number of emplaced elements at the time of the last sort().
   ///   3. Insertions can only be done via an emplace(Key,Value). Note that in the GPU version, this insertion WILL ALWAYS OCCUR even if identical elements previously existed, so it is the responsibility of the programmer to ensure uniqueness.
   ///   4. Lookups can only be done via an at(Key), and will only find values inserted before the last call to sort().
   ///   5. Iteration should be done using the CARE_STREAM_MAP_LOOP macro, as the API for iteration differs depending on
   ///      execution policy and the macro abstracts those differences away.
   ///   6. The sortedness of a map is only guaranteed after a sort() call.
   
   ///  "Enhancements" compared to std::map
   ///  1. Semantically each key and value is associated with an index similar to a vector, which provides
   ///     GPU friendly lookups on a per thread level.
   //   2. Keys and Values can be looked up by this index using key_at(n) and value_at(n). 
   //
   template <typename key_type,
             typename mapped_type,
             typename Exec>
   class host_device_map
   {
      public:
         host_device_map(size_t max_entries, mapped_type miss_signal);
         CARE_HOST_DEVICE inline void emplace(key_type key, mapped_type val) const;
         CARE_HOST_DEVICE inline mapped_type at(key_type key) const;
         void sort();
         void free();
         void clear();
         void reserve(int size);
         int size() const;
   };

#if !CARE_ENABLE_GPU_SIMULATION_MODE
   // ********************************************************************************
   // RAJA::seq_exec specialization.
   // ********************************************************************************
   template <typename key_type, typename mapped_type>
   class host_device_map< key_type, mapped_type, RAJA::seq_exec> {
      public:
         // constructor
         host_device_map(size_t max_entries, mapped_type miss_signal) : m_map(), m_max_size(max_entries), m_signal(miss_signal)  {
            m_map = new std::map<key_type, mapped_type>{};
            m_size = new int();
            *m_size = 0;
         }

        // emplace a key value pair
        inline void emplace(key_type key, mapped_type val) const {
           m_map->emplace(key, val);
           // TODO Add control for this check
           if (m_map->size() > m_max_size) {
              printf("[CARE] Warning: host_device_map exceeds max size %d > %d\n", (int)m_map->size(), m_max_size);
           }
        }

        // lookup a value
        inline mapped_type at(key_type key) const {
           auto search = m_map->find(key);
           if (search != m_map->end()) {
              return search->second;
           }
           else {
              return m_signal;
           }
        }

        // prepare for lookups, update our size from the map size
        void sort() {
           *m_size = m_map->size();
        } 
        

        // free any heap data
        void free() {
           delete m_map;
           delete m_size;
        }

        // return the number of inserted elements
        int size() const { return *m_size; }

        // clear any added elements
        void clear() { m_map->clear();}
        
        // preallocate buffers for adding up to size elements
        void reserve(int max_size) {
           m_max_size = max_size;
        }

        // iteration - meant to only be called by the CARE_MAP_LOOP macros
        inline typename std::map<key_type, mapped_type>::iterator begin () const { return m_map->begin(); }
        inline typename std::map<key_type, mapped_type>::iterator end () const { return m_map->end(); }
        

      private:
         // we do a heap allocated map to ensure no deep copies occur during lambda capture
         std::map<key_type, mapped_type> * m_map = nullptr;
         int * m_size = nullptr;
         int m_max_size;
         mapped_type m_signal;
   };
#endif // !CARE_ENABLE_GPU_SIMULATION_MODE

#if defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE

   // ********************************************************************************
   // RAJADeviceExec specialization.
   // ********************************************************************************
   template <typename key_type, typename mapped_type>
   class host_device_map<key_type, mapped_type, RAJADeviceExec>
   {
      public:
         using int_ptr = care::host_device_ptr<int>;
         // constructor
         host_device_map(size_t max_entries, mapped_type miss_signal) : m_max_size(max_entries), m_signal(miss_signal), m_gpu_map{max_entries}   {
            // m_size_ptr will be atomically incremented as elements are emplaced into the map
            m_size_ptr = int_ptr(1, "map_size");
            // set size to 0
            clear();
         }

        // emplace a key value pair, using return of atomic increment to provide the initial insertion index
        inline CARE_DEVICE void emplace(key_type key, mapped_type val) const {
           care::local_ptr<int> size_ptr = m_size_ptr;
           int index = ATOMIC_ADD(size_ptr[0], 1);
           // TODO Add control for this check
           if (size_ptr[0] > m_max_size) {
              printf("[CARE] Warning: host_device_map exceeds max size %d > %d\n", size_ptr[0], m_max_size);
           }
           LocalKeyValueSorter<key_type, mapped_type, RAJADeviceExec> const & local_map = m_gpu_map;
           local_map.setKey(index, key);
           local_map.setValue(index, val);
        }

        // lookups (valid after a sort() call) are done by binary searching the keys and using the
        // index of the located key to grab the appropriate value
        inline CARE_DEVICE mapped_type at(key_type key) const {
           int index = care::BinarySearch<key_type>(m_gpu_map.keys(),0,m_size,key);
           if (index >= 0) {
              return m_gpu_map.values()[index];
           }
           else {
              return m_signal;
           }
        }
        

        // call sort() after emplaces are all done and before lookups are needed
        void sort() {
           m_size = m_size_ptr.pick(0);
           // only sort on the subset of values added so far
           m_gpu_map.sortByKey(m_size);
        }


        // release any heap data.
        void free() {
           m_size_ptr.free();
           // KeyValueSorter will free its data during destruction.
        }

        /* initializes size value to 0, clears any added elements */
        void clear() {
           int_ptr size_ptr = m_size_ptr;
           CARE_PARALLEL_KERNEL{
              size_ptr[0] = 0;
           } CARE_PARALLEL_KERNEL_END
           m_size = 0;
        }
        
        // return the number of inserted elements
        int size() const {  return m_size; }
        
        // preallocate buffers for adding up to size elements
        void reserve(int max_size) { 
           if (m_max_size < max_size) {
              if (m_size == 0) {
                 m_gpu_map = std::move(KeyValueSorter<key_type, mapped_type, RAJADeviceExec>{max_size}); 
              }
              else {
                 // copy existing state into new map
                 KeyValueSorter<key_type, mapped_type, RAJADeviceExec> new_map{max_size};
                 auto & map = m_gpu_map;
                 CARE_STREAM_LOOP(i, 0, m_size) {
                    new_map.setKey(i, map.key(i));
                    new_map.setValue(i, map.value(i));
                 } CARE_STREAM_LOOP_END
                 m_gpu_map = std::move(new_map);
              }
           }
           m_max_size = max_size;
        }
        
        // iteration - only to be used by macro layer */ 
        struct iterator { 
           CARE_DEVICE iterator(key_type const key, mapped_type & val) : first(key), second(val) {}
           key_type const first;
           mapped_type &second;
           CARE_DEVICE iterator * operator ->() {return this;}
        };
           
        // lookups (valid after a sort() call) are done by binary searching the keys and using the
        // index of the located key to grab the appropriate value
        inline CARE_DEVICE mapped_type & value_at(int index) const {
           return m_gpu_map.values()[index];
        }
        
        // lookups (valid after a sort() call) are done by binary searching the keys and using the
        // index of the located key to grab the appropriate value
        inline CARE_DEVICE key_type const &  key_at(int index) const {
           return m_gpu_map.keys()[index];
        }

        inline CARE_DEVICE iterator iterator_at(int index) const { 
           return iterator(key_at(index),value_at(index));
        }


      private:
         int_ptr m_size_ptr = nullptr;
         int m_size = 0;
         int m_max_size;
         int m_signal;
         KeyValueSorter<key_type, mapped_type, RAJADeviceExec> m_gpu_map;
   };

#define CARE_STREAM_MAP_LOOP(INDX, START, ITER, MAP) \
   CARE_STREAM_LOOP(INDX,START,MAP.size()) { \
       auto ITER = MAP.iterator_at(INDX-START);

#define CARE_STREAM_MAP_LOOP_END } CARE_STREAM_LOOP_END
   
#else
#define CARE_STREAM_MAP_LOOP(INDEX, START, ITER, MAP) \
  int INDEX = START; \
  for (auto ITER = MAP.begin(); ITER != MAP.end(); ++ITER, ++INDEX) { \
     
#define CARE_STREAM_MAP_LOOP_END }
   
#endif // defined(CARE_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE

   // this implementation is used for benchmarking - may be appropriate choice depending on performance
   // of map on your system / compiler.

   // force the use of a key value sorter on the backend instead of std::map
   struct force_keyvaluesorter {};

   // ********************************************************************************
   // force_keyvaluesorter specialization. Host only.
   // ********************************************************************************
   template <typename key_type, typename mapped_type>
   class host_device_map<key_type, mapped_type, force_keyvaluesorter>
   {
      public:

         // constructor
         host_device_map(size_t max_entries, mapped_type signal) : m_max_size(max_entries), m_signal(signal) {
            
            // m_size_ptr will be atomically incremented as elements are emplaced into the map
            m_size_ptr = new int();
            // set size to 0
            clear();
            // back the map with a KeyValueSorter<key_type, mapped_type>
            m_map = KeyValueSorter<key_type, mapped_type, RAJA::seq_exec>{max_entries};
         }

        // emplace a key value pair,increment length
        inline void emplace(key_type key, mapped_type val) const {

           m_map.setKey(*m_size_ptr, key);
           m_map.setValue(*m_size_ptr, val);
           ++(*m_size_ptr);
        }

        // lookups (valid after a sort() call) are done by binary searching the keys and using the
        // index of the located key to grab the appropriate value
        inline mapped_type at(key_type key) const {
           int index = care::BinarySearch<key_type>(m_map.keys(),0,m_size,key);
           if (index == -1) {
              return m_signal;
           }
           else {
              mapped_type val = m_map.values()[index];
              return val;
           }
        }

        // call sort() after emplaces are all done and before lookups are needed
        void sort() {
           m_map.sortByKey();
           // cache the keys and values for lookups. Doing this outside of a kernel context is important
           // so that the primary m_map object (not the lambda-captured copy) has initialized keys.
           m_map.initializeKeys();
           m_map.initializeValues();
           m_size = *m_size_ptr;
        }

        // release any heap data
        void free() {
            // KeyValueSorter will free its data during destruction.
            delete m_size_ptr;
        }
        
        // return the number of inserted elements
        int size() {  return m_size; }
        
        // clear any added elements
        void clear() {
           *m_size_ptr = 0;
           m_size = 0;
        }
        
        // preallocate buffers for adding up to size elements
        void reserve(int max_size) { 
           if (m_max_size < max_size) {
              if (m_size == 0) {
                 m_map = std::move(KeyValueSorter<key_type, mapped_type, RAJA::seq_exec>{max_size}); 
              }
              else {
                 // copy existing state into new map
                 KeyValueSorter<key_type, mapped_type, RAJA::seq_exec> new_map{max_size};
                 auto & map = m_map;
                 CARE_SEQUENTIAL_LOOP(i, 0, m_size) {
                    new_map.setKey(i, map.key(i));
                    new_map.setValue(i, map.value(i));
                 } CARE_SEQUENTIAL_LOOP_END
                 m_map = std::move(new_map);
              }
              m_max_size = max_size;
           }
        }

      private:
         mutable int * m_size_ptr = nullptr;
         mutable int  m_size = 0;
         mutable int m_max_size;
         KeyValueSorter<key_type, mapped_type, RAJA::seq_exec> m_map;
         /* hasBeenSorted may be used in the future to enable an implicit sort on lambda capture */
         bool hasBeenSorted = false;
         int m_signal;
   };

}

#endif
