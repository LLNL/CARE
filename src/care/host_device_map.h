//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////
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
         host_device_map(size_t max_entries);
         host_device_map() noexcept;
         host_device_map(host_device_map const & other) noexcept;
         host_device_map(host_device_map && other) noexcept;
         host_device_map& operator=(host_device_map&& other) noexcept;
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
         // default constructor
         host_device_map() noexcept {};
         
         // constructor taking max number of entries
         host_device_map(size_t max_entries) : host_device_map{} {
            m_max_size = max_entries;
            m_map = new std::map<key_type, mapped_type>{};
            m_size = new int();
            *m_size = 0;
            m_iterator = new typename std::map<key_type, mapped_type>::iterator{};
            *m_iterator = m_map->begin();
            m_next_iterator_index = new int();
            *m_next_iterator_index = 0;
         }

         // constructor that also takes the miss signal
         host_device_map(size_t max_entries, mapped_type miss_signal) : host_device_map{max_entries} {
            m_signal = miss_signal;
         }

         // copy constructor 
         host_device_map(host_device_map const & other) noexcept :
            m_map(other.m_map),
            m_size(other.m_size),
            m_iterator(other.m_iterator),
            m_next_iterator_index(other.m_next_iterator_index),
            m_max_size(other.m_max_size),
            m_signal(other.m_signal)
	 {
	 }

         // move constructor
         host_device_map(host_device_map && other) noexcept  {
            delete m_map;
            delete m_size;
            delete m_iterator;
            delete m_next_iterator_index;
            m_map = other.m_map;
            m_size = other.m_size;
            m_iterator = other.m_iterator;
            m_next_iterator_index = other.m_next_iterator_index;
            m_max_size = other.m_max_size;
            m_signal = other.m_signal;
            other.m_map = nullptr;
            other.m_size = nullptr;
            other.m_iterator = nullptr;
            other.m_next_iterator_index = nullptr;
         }

         host_device_map& operator=(host_device_map && other) noexcept  {
            delete m_map;
            delete m_size;
            delete m_iterator;
            delete m_next_iterator_index;
            m_map = other.m_map;
            m_size = other.m_size;
            m_iterator = other.m_iterator;
            m_next_iterator_index = other.m_next_iterator_index;
            m_max_size = other.m_max_size;
            m_signal = other.m_signal;
            other.m_map = nullptr;
            other.m_size = nullptr;
            other.m_iterator = nullptr;
            other.m_next_iterator_index = nullptr;
            return *this;
         }

        // emplace a key value pair
        inline void emplace(key_type key, mapped_type val) const {
           m_map->emplace(key, val);
           // TODO Add control for this check
           if (m_map->size() > (size_t)m_max_size) {
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
           *m_iterator = m_map->begin();
           *m_next_iterator_index = 0;
        } 
        

        // free any heap data
        void free() {
           delete m_map;
           delete m_size;
           delete m_iterator;
           delete m_next_iterator_index;
        }

        // return the number of inserted elements
        int size() const { return *m_size; }

        // clear any added elements
        void clear() { 
           m_map->clear();
           *m_size = 0;
           *m_iterator = m_map->begin();
           *m_next_iterator_index = 0;
        }
        
        // preallocate buffers for adding up to size elements
        void reserve(int max_size) {
           m_max_size = max_size;
        }

        // iteration - meant to only be called by the CARE_MAP_LOOP macros
        inline typename std::map<key_type, mapped_type>::iterator iterator_at(int index) const {
           if (index != *m_next_iterator_index) {
              *m_iterator = m_map->begin();
              for (*m_next_iterator_index = 0; *m_next_iterator_index < index ; ++(*m_next_iterator_index)) {
                 ++(*m_iterator);
              }
           }
           ++(*m_next_iterator_index);
           return (*m_iterator)++;
        }

      private:
         // we do a heap allocated map to ensure no deep copies occur during lambda capture
         std::map<key_type, mapped_type> * m_map = nullptr;
         typename std::map<key_type, mapped_type>::iterator * m_iterator = nullptr;
         int * m_next_iterator_index = nullptr;
         int * m_size = nullptr;
         int m_max_size = 0;
         mapped_type m_signal {};
   };
#endif // !CARE_ENABLE_GPU_SIMULATION_MODE

#if defined(CARE_PARALLEL_DEVICE) || CARE_ENABLE_GPU_SIMULATION_MODE

   // ********************************************************************************
   // RAJADeviceExec specialization.
   // ********************************************************************************
   template <typename key_type, typename mapped_type>
   class host_device_map<key_type, mapped_type, RAJADeviceExec>
   {
      public:
         using int_ptr = care::host_device_ptr<int>;
         
         // default constructor
         CARE_HOST_DEVICE host_device_map() noexcept {};
         
         // constructor taking max_entries
         host_device_map(size_t max_entries) : m_max_size(max_entries), m_signal{}, m_gpu_map{max_entries}   {
            // m_size_ptr[0] will be atomically incremented as elements are emplaced into the map
            m_size_ptr = int_ptr(1, "map_size");
            // set size to 0
            clear();
         }

         // constructor that also takes the miss signal
         host_device_map(size_t max_entries, mapped_type miss_signal) : host_device_map{max_entries} {
            m_signal = miss_signal;
         }

         // copy constructor 
         CARE_HOST_DEVICE host_device_map(host_device_map const & other) noexcept :
            m_max_size{other.m_max_size},
            m_signal{other.m_signal},
            m_gpu_map{other.m_gpu_map},
            m_size_ptr{other.m_size_ptr},
            m_size{other.m_size}
	 {
	 }

         // move constructor
         CARE_HOST_DEVICE host_device_map(host_device_map&& other) noexcept { 
            m_max_size = other.m_max_size;
            m_signal = other.m_signal;
            m_gpu_map = std::move(other.m_gpu_map);
            m_size_ptr.free();
            m_size_ptr = other.m_size_ptr;
            other.m_size_ptr = nullptr;
            m_size = other.m_size;
         }

         // move assignment
         host_device_map& operator=(host_device_map && other) noexcept {
            m_max_size = other.m_max_size;
            m_signal = other.m_signal;
            m_gpu_map = std::move(other.m_gpu_map);
            m_size_ptr.free();
            m_size_ptr = other.m_size_ptr;
            other.m_size_ptr = nullptr;
            m_size = other.m_size;
            return *this;
         }

        // emplace a key value pair, using return of atomic increment to provide the initial insertion index
        inline CARE_HOST_DEVICE void emplace(key_type key, mapped_type val) const {
           care::local_ptr<int> size_ptr = m_size_ptr;
           int index = ATOMIC_ADD(size_ptr[0], 1);
           // commenting out to avoid having printfs compiled into every kernel that uses emplace
           //if (size_ptr[0] > m_max_size) {
           //   printf("[CARE] Warning: host_device_map exceeds max size %d > %d\n", size_ptr[0], m_max_size);
           //}
           LocalKeyValueSorter<key_type, mapped_type, RAJADeviceExec> const & local_map = m_gpu_map;
           local_map.setKey(index, key);
           local_map.setValue(index, val);
        }

        // lookups (valid after a sort() call) are done by binary searching the keys and using the
        // index of the located key to grab the appropriate value
        inline CARE_HOST_DEVICE mapped_type at(key_type key) const {
           int index = care::BinarySearch<key_type>(m_gpu_map.keys(),0,m_size,key);
           if (index >= 0) {
              const care::local_ptr<mapped_type>& values = m_gpu_map.values();
              return values[index];
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
              KeyValueSorter<key_type, mapped_type, RAJADeviceExec> new_map{
                 static_cast<size_t>(max_size)};

              if (m_size > 0) {
                 // copy existing state into new map
                 auto & map = m_gpu_map;

                 CARE_STREAM_LOOP(i, 0, m_size) {
                    new_map.setKey(i, map.key(i));
                    new_map.setValue(i, map.value(i));
                 } CARE_STREAM_LOOP_END
              }

              m_gpu_map = std::move(new_map);
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
           const care::local_ptr<mapped_type>& values = m_gpu_map.values();
           return values[index];
        }
        
        // lookups (valid after a sort() call) are done by binary searching the keys and using the
        // index of the located key to grab the appropriate value
        inline CARE_DEVICE key_type const &  key_at(int index) const {
           const care::local_ptr<key_type>& keys = m_gpu_map.keys();
           return keys[index];
        }

        inline CARE_DEVICE iterator iterator_at(int index) const { 
           return iterator(key_at(index),value_at(index));
        }


      private:
         int_ptr m_size_ptr = nullptr;
         int m_size = 0;
         int m_max_size = 0;
         mapped_type m_signal {};
         KeyValueSorter<key_type, mapped_type, RAJADeviceExec> m_gpu_map{};
   };

#endif // defined(CARE_PARALLEL_DEVICE) || CARE_ENABLE_GPU_SIMULATION_MODE

#define CARE_STREAM_MAP_LOOP(INDX, ITER, MAP) \
   CARE_STREAM_LOOP(INDX,0,MAP.size()) { \
       auto ITER = MAP.iterator_at(INDX);

#define CARE_STREAM_MAP_LOOP_END } CARE_STREAM_LOOP_END
   
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
         // default constructor
         host_device_map() noexcept {};         
         
         // constructor
         host_device_map(size_t max_entries) : host_device_map{} {
            m_max_size = max_entries;
            // m_size_ptr will be atomically incremented as elements are emplaced into the map
            m_size_ptr = new int();
            // set size to 0
            clear();
            // back the map with a KeyValueSorter<key_type, mapped_type>
            m_map = KeyValueSorter<key_type, mapped_type, RAJA::seq_exec>{max_entries};
         }

         // constructor
         host_device_map(size_t max_entries, mapped_type signal) : host_device_map{max_entries} { 
            m_signal = signal;
         }
         
         // copy constructor 
         host_device_map(host_device_map const & other) noexcept :
            m_size_ptr(other.m_size_ptr),
            m_size(other.m_size),
            m_map(other.m_map),
            m_max_size(other.m_max_size),
            m_signal(other.m_signal)
	 {
	 }

         // move constructor
         host_device_map(host_device_map && other)  noexcept {
            delete m_size_ptr;
            m_size_ptr = other.m_size_ptr;
            m_size = other.m_size;
            m_map = std::move(other.m_map);
            m_max_size = other.m_max_size;
            m_signal = other.m_signal;
         }
         
         // move assignment
         host_device_map& operator=(host_device_map && other)  noexcept {
            delete m_size_ptr;
            m_size_ptr = other.m_size_ptr;
            m_size = other.m_size;
            m_map = std::move(other.m_map);
            m_max_size = other.m_max_size;
            m_signal = other.m_signal;
            return *this;
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
              KeyValueSorter<key_type, mapped_type, RAJA::seq_exec> new_map{
                 static_cast<size_t>(max_size)};

              if (m_size > 0) {
                 // copy existing state into new map
                 auto & map = m_map;

                 CARE_SEQUENTIAL_LOOP(i, 0, m_size) {
                    new_map.setKey(i, map.key(i));
                    new_map.setValue(i, map.value(i));
                 } CARE_SEQUENTIAL_LOOP_END
              }

              m_map = std::move(new_map);
           }

           m_max_size = max_size;
        }

      private:
         mutable int * m_size_ptr = nullptr;
         mutable int  m_size = 0;
         mutable int m_max_size = 0;
         KeyValueSorter<key_type, mapped_type, RAJA::seq_exec> m_map{};
         /* hasBeenSorted may be used in the future to enable an implicit sort on lambda capture */
         bool hasBeenSorted = false;
         mapped_type m_signal {};
   };

}

#endif
