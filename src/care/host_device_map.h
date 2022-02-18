//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2022 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////
#ifndef _CARE_DEVICE_UNORDERED_MAP_H
#define _CARE_DEVICE_UNORDERED_MAP_H
#include "care/config.h"
#include "care/atomic.h"

#include <unordered_map>
#include <care/KeyValueSorter.h>

namespace care {

   ///
   /// @author Peter Robinson
   ///
   /// @class host_device_map is a rudimentary associative map. On the host,
   /// it is backed by a std::unordered_map, and on the device it uses a
   /// care::KeyValueSorter.
   ///
   /// Restrictions when compared to std::map:
   ///   1. Total capacity must be declared up front.
   ///   2. a call to sort() must be performed before any lookups are done.
   ///   3. Insertions can only be done via an emplace(Key,Value).
   ///   4. Lookups can only be done via an at(Key)
   ///
   template <typename key_type,
             typename mapped_type,
             typename Exec>
   class host_device_map
   {
      public:
         host_device_map(size_t max_entries);
         CARE_HOST_DEVICE inline void emplace(key_type key, mapped_type val) const;
         CARE_HOST_DEVICE inline mapped_type at(key_type key) const;
         void sort();
         void free();
   };

// ********************************************************************************
// RAJA::seq_exec specialization.
// ********************************************************************************
   template <typename key_type, typename mapped_type>
   class host_device_map< key_type, mapped_type, RAJA::seq_exec> {
      public:
         // constructor
         host_device_map(size_t max_entries) : m_map()  {
            m_map = new std::unordered_map<key_type, mapped_type>{};
         }
        // emplace a key value pair
        inline void emplace(key_type key, mapped_type val) const {
           (*m_map)[key] = val;
        }
        // lookup a value
        inline mapped_type at(key_type key) const {
           return (*m_map).at(key);
        }
        // prepare for lookups (no-op for std::unordered map implementation)
        void sort() { }

        // delete the map
        void free() { delete m_map; }

      private:
         // we do a heap allocated map to ensure no deep copies occur during lambda capture
         std::unordered_map<key_type, mapped_type> * m_map;
   };


#ifdef CARE_GPUCC

// ********************************************************************************
// RAJADeviceExec specialization.
// ********************************************************************************
   template <typename key_type, typename mapped_type>
   class host_device_map<key_type, mapped_type, RAJADeviceExec>
   {
      public:
         using int_ptr = care::host_device_ptr<int>;
         // constructor
         host_device_map(size_t max_entries)  {
            // m_size will be atomically incremented as elements are emplaced into the map
            m_size = int_ptr(1, "map_size");
            int_ptr size = m_size;
            CARE_SEQUENTIAL_LOOP(i, 0, 1) {
               size[0] = 0;
            } CARE_SEQUENTIAL_LOOP_END
            // back the map with a KeyValueSorter<key_type, mapped_type>
            m_gpu_map = KeyValueSorter<key_type, mapped_type, RAJADeviceExec>{max_entries};
         }

        // emplace a key value pair, using return of atomic increment to provide the initial insertion index
        inline CARE_DEVICE void emplace(key_type key, mapped_type val) const {
           int index = ATOMIC_ADD(m_size[0], 1);
           LocalKeyValueSorter<key_type, mapped_type, RAJADeviceExec> const & local_map = m_gpu_map;
           local_map.setKey(index, key);
           local_map.setValue(index, val);
        }

        // lookups (valid after a sort() call) are done by binary searching the keys and using the
        // index of the located key to grab the appropriate value
        inline CARE_DEVICE mapped_type at(key_type key) const {
           return m_gpu_map.values()[care::BinarySearch<key_type>(m_gpu_map.keys(),0,m_length,key)];
        }

        // call sort() after emplaces are all done and before lookups are needed
        void sort() {
           m_gpu_map.sortByKey();
           m_length = m_size.pick(0);
        }

        // release any heap data.
        void free() {
           m_size.free();
           // KeyValueSorter will free its data during destruction.
        }

      private:
         int_ptr m_size = nullptr;
         int m_length = -1;
         KeyValueSorter<key_type, mapped_type, RAJADeviceExec> m_gpu_map;
   };
#endif

   // this implementation is used for benchmarking - may be appropriate choice depending on performance
   // of unordered_map on your system / compiler.

   // force the use of a key value sorter on the backend instead of std::unordered_map
   struct force_keyvaluesorter {};

// ********************************************************************************
// force_keyvaluesorter specialization. Host only.
// ********************************************************************************
   template <typename key_type, typename mapped_type>
   class host_device_map<key_type, mapped_type, force_keyvaluesorter>
   {
      public:

         // constructor
         host_device_map(size_t max_entries)  {
            m_length = new int();
            *m_length = 0;
            // back the map with a KeyValueSorter<key_type, mapped_type>
            m_map = KeyValueSorter<key_type, mapped_type, RAJA::seq_exec>{max_entries};
         }

        // emplace a key value pair,increment length
        inline void emplace(key_type key, mapped_type val) const {
           m_map.setKey(*m_length, key);
           m_map.setValue(*m_length, val);
           ++(*m_length);
        }

        // lookups (valid after a sort() call) are done by binary searching the keys and using the
        // index of the located key to grab the appropriate value
        inline mapped_type at(key_type key) const {
           int index = care::BinarySearch<key_type>(m_map.keys(),0,*m_length,key);
           mapped_type val = m_map.values()[index];
           return val;
        }

        // call sort() after emplaces are all done and before lookups are needed
        void sort() {
           m_map.sortByKey();
           // cache the keys and values for lookups. Doing this outside of a kernel context is important
           // so that the primary m_map object (not the lambda-capgtured copy) has initialized keys.
           m_map.initializeKeys();
           m_map.initializeValues();
        }

        // release any heap data
        void free() {
            // KeyValueSorter will free its data during destruction.
            delete m_length;
        }

      private:
         mutable int *  m_length = 0;
         KeyValueSorter<key_type, mapped_type, RAJA::seq_exec> m_map;
         bool hasBeenSorted = false;
   };


}

#endif
