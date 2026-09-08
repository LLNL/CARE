//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// This header includes the implementations of KeyValueSorter.
// In very large monolithic codes, including this in too many compilation
// units can cause linking issues (particularly device link issues) due to
// the file size. In that case, external template instantiation should be
// used (this requires CARE_ENABLE_EXTERN_INSTANTIATE to be turned ON in the cmake
// configuration): this file should only be included in the compilation unit containing
// the instantiation and KeyValueSorter.h (along with the extern template
// declarations) should be included everywhere else.

#ifndef _CARE_KEY_VALUE_SORTER_IMPL_H_
#define _CARE_KEY_VALUE_SORTER_IMPL_H_

#include "care/algorithm.h"
#include "care/KeyValueSorter_decl.h"

// Other CARE headers
#include "care/LoopFuser.h"
// Other library headers
#ifdef CARE_GPUCC
#if defined(__CUDACC__)
#include "cub/cub.cuh"
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
#endif

#if defined(__HIPCC__)
#include "hipcub/hipcub.hpp"
#endif
#endif

namespace care {

// TODO openMP parallel implementation

#if defined(CARE_PARALLEL_DEVICE) || CARE_ENABLE_GPU_SIMULATION_MODE
///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes keys and values by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to copy
/// @param[in] arr - input array
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
CARE_INLINE void setKeyValueArraysFromArray(host_device_ptr<KeyType> & keys,
                                            host_device_ptr<ValueType> & values,
                                            const size_t len, const ValueType* arr)
{
   // TODO: this requires key types to be constructable from size_t -
   // maybe only enable this for integral types?

   CARE_SEQUENTIAL_LOOP(i, 0, len) {
      keys[i] = (KeyType)i;
      values[i] = arr[i];
   } CARE_SEQUENTIAL_LOOP_END
}

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes the KeyValueSorter by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to copy
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
CARE_INLINE void setKeyValueArraysFromManagedArray(host_device_ptr<KeyType> & keys,
                                                   host_device_ptr<ValueType> & values,
                                                   const size_t len, const host_device_ptr<const ValueType>& arr)
{
   // TODO: this requires key types to be constructable from size_t -
   // maybe only enable this for integral types?

   FUSIBLE_LOOP_STREAM(i, 0, len) {
      keys[i] = (KeyType) i;
      values[i] = arr[i];
   } FUSIBLE_LOOP_STREAM_END
}

///////////////////////////////////////////////////////////////////////////
/// @author Jeff Keasler, Alan Dayton
/// @brief Eliminates duplicate values from sorted key/value arrays
/// @param[out] newKeys New key array with duplicates removed
/// @param[out] newValues New value array with duplicates removed
/// @param[in] oldKeys Old key array
/// @param[in] oldValues Old value array (sorted)
/// @param[in] oldLen Length of the old arrays
/// @return Length of the new key/value arrays
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
CARE_INLINE size_t eliminateKeyValueDuplicates(host_device_ptr<KeyType>& newKeys,
                                               host_device_ptr<ValueType>& newValues,
                                               const host_device_ptr<const KeyType>& oldKeys,
                                               const host_device_ptr<const ValueType>& oldValues,
                                               const size_t oldLen)
{
   // Save values that are not duplicates and their corresponding keys
   int newSize = 0;

   SCAN_LOOP(i, 0, oldLen, idx, newSize, (i == 0) || (oldValues[i] != oldValues[i-1])) {
      newKeys[idx] = oldKeys[i];
      newValues[idx] = oldValues[i];
   } SCAN_LOOP_END(oldLen, idx, newSize)

   // Update space for the key value pairs without duplicates
   newKeys.realloc(newSize);
   newValues.realloc(newSize);

   return (size_t)newSize;
}

template <typename KeyType, typename ValueType, typename SizeType>
CARE_INLINE void IntersectKeyValueSorters(RAJADeviceExec exec,
                                          KeyValueSorter<KeyType, ValueType, RAJADeviceExec> sorter1, SizeType size1,
                                          KeyValueSorter<KeyType, ValueType, RAJADeviceExec> sorter2, SizeType size2,
                                          host_device_ptr<KeyType>& matches1,
                                          host_device_ptr<KeyType>& matches2,
                                          SizeType & numMatches)
{
   SizeType smaller = (size1 < size2) ? size1 : size2;
   SizeType start1 = 0;
   SizeType start2 = 0;

   numMatches = 0;
   if (smaller == 0) {
      matches1 = nullptr;
      matches2 = nullptr;
      return;
   }

   matches1.alloc(smaller);
   matches1.namePointer("matches1");
   matches2.alloc(smaller);
   matches2.namePointer("matches2");

   host_device_ptr<KeyType> smallerMatches, largerMatches;
   host_device_ptr<KeyType> smallerKeys, largerKeys;
   SizeType larger, smallStart, largeStart;
   host_device_ptr<const ValueType> smallerArray, largerArray;
   if (smaller == size1) {
      smallerArray = sorter1.values();
      largerArray = sorter2.values();
      smallerKeys = sorter1.keys();
      largerKeys = sorter2.keys();
      larger = size2;
      smallStart = start1;
      largeStart = start2;
      smallerMatches = matches1;
      largerMatches = matches2;
   }
   else {
      smallerArray = sorter2.values();
      largerArray = sorter1.values();
      smallerKeys = sorter2.keys();
      largerKeys = sorter1.keys();
      larger = size1;
      smallStart = start2;
      largeStart = start1;
      smallerMatches = matches2;
      largerMatches = matches1;
   }

   host_device_ptr<int> searches(smaller + 1);
   host_device_ptr<int> matched(smaller + 1);
   CARE_STREAM_LOOP(i, 0, smaller + 1) {
      if (i == smaller) {
         searches[i] = -1;
      }
      else {
         // to be consistent with CPU algorithm, find the first match
         int match = care::BinarySearch<ValueType>(largerArray, largeStart, larger, smallerArray[i + smallStart]);
         while (match > largeStart && largerArray[match - 1] == largerArray[match]) {
            --match;
         }
         searches[i] = match;
      }
      matched[i] = i != smaller && searches[i] > -1;
   } CARE_STREAM_LOOP_END

   care::exclusive_scan(RAJADeviceExec{}, matched, nullptr, smaller + 1, 0, true);

   CARE_STREAM_LOOP(i, 0, smaller) {
      if (searches[i] > -1) {
         smallerMatches[matched[i]] = smallerKeys[i + smallStart];
         largerMatches[matched[i]] = largerKeys[searches[i]];
      }
   } CARE_STREAM_LOOP_END
   numMatches = matched.pick(smaller);
   searches.free();
   matched.free();

   if (numMatches == 0) {
      matches1.free();
      matches2.free();
   }
   else {
      /* reduce the size of the matches arrays*/
      matches1.realloc(numMatches);
      matches2.realloc(numMatches);
   }
}

#endif // defined(CARE_PARALLEL_DEVICE) || CARE_ENABLE_GPU_SIMULATION_MODE

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes keys and values by copying elements from the array
/// @param[out] keyValues - The key value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
CARE_INLINE void setKeyValueArraysFromArray(host_device_ptr<_kv<KeyType,ValueType>> & keyValues,
                                            const size_t len, const ValueType* arr)
{
   // TODO: this requires key types to be constructable from a size_t -
   // maybe only enable this for integral types?

   CARE_SEQUENTIAL_LOOP(i, 0, (int) len) {
      keyValues[i].key = (KeyType)i;
      keyValues[i].value = arr[i];
   } CARE_SEQUENTIAL_LOOP_END
}

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes the KeyValueSorter by copying elements from the array
/// @param[out] keyValues - The key value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
CARE_INLINE void setKeyValueArraysFromManagedArray(host_device_ptr<_kv<KeyType, ValueType> > & keyValues,
                                                   const size_t len, const host_device_ptr<const ValueType>& arr)
{
   // TODO: this requires key types to be constructable from a size_t -
   // maybe only enable this for integral types?

   FUSIBLE_LOOP_STREAM(i, 0, (int)len) {
      keyValues[i].key = (KeyType)i;
      keyValues[i].value = arr[i];
   } FUSIBLE_LOOP_STREAM_END
}

///////////////////////////////////////////////////////////////////////////
/// @author Jeff Keasler, Alan Dayton
/// @brief Eliminates duplicate values
/// First does a stable sort based on the values, which preserves the
///    ordering in case of a tie. Then duplicates are removed. The final
///    step is to unsort.
/// @param[in/out] keyValues - The key value array to eliminate duplicates in
/// @param[in/out] len - original length of key value array/new length of array
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
CARE_INLINE size_t eliminateKeyValueDuplicates(host_device_ptr<_kv<KeyType, ValueType> > & keyValues, const size_t len)
{
   size_t newSize = len;
   if (len > 1) {
      CHAIDataGetter<_kv<KeyType, ValueType>, RAJA::seq_exec> getter {};
      _kv<KeyType, ValueType> * rawData = getter.getRawArrayData(keyValues);

      // First do a stable sort by value (preserve the original order
      // in the case of a tie)
      std::stable_sort(rawData, rawData + len);

      // Then eliminate duplicates
      size_t lsize = len - 1;  /* adjust search range */
      size_t put = 0;
      size_t get = 0;

      while (get < lsize) {
         if (put != get) {
            memcpy(&rawData[put], &rawData[get], sizeof(struct _kv<KeyType, ValueType>));
         }

         if (rawData[get].value == rawData[get+1].value) {
            ++get;
            ++put;

            while (get < lsize && rawData[get].value == rawData[get+1].value) {
               ++get;
            }
            ++get;
         }
         else {
            ++get;
            ++put;
         }
      }

      if (rawData[lsize].value != rawData[lsize-1].value) {
         memmove(&rawData[put++], &rawData[lsize], sizeof(struct _kv<KeyType, ValueType>));
      }

      lsize = put;

      // Then sort by key to get the original ordering
      std::sort(rawData, rawData + lsize, cmpKeys<_kv<KeyType,ValueType>>);

      // Reallocate memory
      keyValues.realloc(lsize);

      newSize = lsize;
   }

   return newSize;
}

///////////////////////////////////////////////////////////////////////////
/// @author Alan Dayton
/// @brief Initializes the keys
/// The keys are stored in the managed array of _kv structs. To get the
/// keys separately, they must be copied into their own array.
/// @param[out] keys - The key array
/// @param[in] keyValues - The key value array
/// @param[in/out] len - length of key value array
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
CARE_INLINE void initializeKeyArray(host_device_ptr<KeyType>& keys,
                                    const host_device_ptr<const _kv<KeyType, ValueType> >& keyValues, const size_t len)
{
   CARE_STREAM_LOOP(i, 0, len) {
      keys[i] = keyValues[i].key;
   } CARE_STREAM_LOOP_END

   return;
}

///////////////////////////////////////////////////////////////////////////
/// @author Alan Dayton
/// @brief Initializes the values
/// The values are stored in the managed array of _kv structs. To get the
///    values separately, they must be copied into their own array.
/// @param[out] values - The values array
/// @param[in] keyValues - The key value array
/// @param[in/out] len - length of key value array
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
CARE_INLINE void initializeValueArray(host_device_ptr<ValueType>& values,
                                      const host_device_ptr<const _kv<KeyType, ValueType> >& keyValues, const size_t len)
{
   CARE_STREAM_LOOP(i, 0, len) {
      values[i] = keyValues[i].value;
   } CARE_STREAM_LOOP_END

   return;
}



#if !CARE_ENABLE_GPU_SIMULATION_MODE
// This assumes arrays have been sorted and unique. If they are not uniqued the GPU
// and CPU versions may have different behaviors (the index they match to may be different,
// with the GPU implementation matching whatever binary search happens to land on, and the
// CPU version matching the first instance.

template <typename KeyType, typename ValueType>
CARE_INLINE void IntersectKeyValueSorters(RAJA::seq_exec /* exec */,
                                          KeyValueSorter<KeyType, ValueType, RAJA::seq_exec> sorter1, int size1,
                                          KeyValueSorter<KeyType, ValueType, RAJA::seq_exec> sorter2, int size2,
                                          host_device_ptr<KeyType>& matches1,
                                          host_device_ptr<KeyType>& matches2,
                                          int & numMatches)
{
   numMatches = 0 ;
   const int smaller = (size1 < size2) ? size1 : size2 ;

   if (smaller == 0) {
      matches1 = nullptr ;
      matches2 = nullptr ;
      return ;
   }
   else {
      matches1.alloc(smaller);
      matches1.namePointer("matches1");
      matches2.alloc(smaller);
      matches2.namePointer("matches2");
   }

   /* This algorithm assumes that the nodelists are sorted */


   int i = 0 ;
   int j = 0 ;
   host_ptr<KeyType> host_matches1 = matches1 ;
   host_ptr<KeyType> host_matches2 = matches2 ;
   /* keys() and values() will allocate managed arrays for the keys and values,
    * respectively, if they were not previously allocated.
    * Check to see whether they were previously allocated. */
   bool sorter1KeysAllocated = sorter1.keysAllocated() ;
   bool sorter2KeysAllocated = sorter2.keysAllocated() ;
   bool sorter1ValuesAllocated = sorter1.valuesAllocated() ;
   bool sorter2ValuesAllocated = sorter2.valuesAllocated() ;
   host_ptr<KeyType const> host_sorter1_key = sorter1.keys() ;
   host_ptr<KeyType const> host_sorter2_key = sorter2.keys() ;
   host_ptr<ValueType const> host_sorter1_value = sorter1.values() ;
   host_ptr<ValueType const> host_sorter2_value = sorter2.values() ;

   for (;; ) {
      if ((i >= size1) || (j >= size2)) {
         break ;
      }
      while ((i < size1) && (host_sorter1_value[i] < host_sorter2_value[j])) {
         i++ ;
      }
      if (i >= size1) {
         break ;
      }
      while ((j < size2) && (host_sorter2_value[j] < host_sorter1_value[i])) {
         j++ ;
      }
      if (j >= size2) {
         break ;
      }
      if (host_sorter1_value[i] == host_sorter2_value[j]) {
         host_matches1[numMatches] = host_sorter1_key[i] ;
         host_matches2[numMatches] = host_sorter2_key[j] ;
         numMatches++ ;
         i++ ;
         j++ ;
      }
      else if (host_sorter1_value[i] < host_sorter2_value[j]) {
         i++ ;
      }
      else if (host_sorter2_value[j] < host_sorter1_value[i]) {
         j++ ;
      }
   }

   /* change the size of the array */
   /* (reallocing to a size of zero should be the same as freeing
    * the object, but insight doesn't seem to think so... hence
    * the extra check here with an explicit free */
   if (numMatches == 0) {
      matches1.free();
      matches2.free();
   }
   else {
      matches1.realloc(numMatches);
      matches2.realloc(numMatches);
   }

   /* If the keys/values arrays were not previously allocated, free them. */
   if (!sorter1KeysAllocated) {
      sorter1.freeKeys() ;
   }
   if (!sorter2KeysAllocated) {
      sorter2.freeKeys() ;
   }
   if (!sorter1ValuesAllocated) {
      sorter1.freeValues() ;
   }
   if (!sorter2ValuesAllocated) {
      sorter2.freeValues() ;
   }
}
#endif // !CARE_ENABLE_GPU_SIMULATION_MODE

} // namespace care

#endif // !defined(_CARE_KEY_VALUE_SORTER_IMPL_H_)
