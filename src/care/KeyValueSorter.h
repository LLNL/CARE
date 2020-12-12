//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// This header includes the implementations of KeyValueSorter.
// In very large monolithic codes, including this in too many compilation
// units can cause linking issues (particularly device link issues) due to
// the file size. In that case, external template instantiation should be
// used: this file should only be included in the compilation unit containing
// the instantiation and KeyValueSorter_decl.h (along with the extern template
// declarations) should be included everywhere else.

#ifndef _CARE_KEY_VALUE_SORTER_H_
#define _CARE_KEY_VALUE_SORTER_H_

#include "care/algorithm.h"
#include "care/KeyValueSorter_decl.h"

namespace care {

// TODO openMP parallel implementation
#ifdef CARE_GPUCC

///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson, Alan Dayton
/// @brief ManagedArray API to cub::DeviceRadixSort::SortPairs
/// @param[in, out] keys   - The array to sort
/// @param[in, out] values - The array that is sorted simultaneously
/// @param[in]      start  - The index to start sorting at
/// @param[in]      len    - The number of elements to sort
/// @param[in]      noCopy - Whether or not to copy the result into the
///                             original arrays or simply replace the
///                             original arrays. Should be false if only
///                             sorting part of the arrays or you will
///                             have bugs!
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyT, typename ValueT, typename Exec>
void sortKeyValueArrays(host_device_ptr<KeyT> & keys,
                        host_device_ptr<ValueT> & values,
                        const size_t start, const size_t len,
                        const bool noCopy) {
   // Allocate space for the result
   host_device_ptr<KeyT> keyResult{len};
   host_device_ptr<ValueT> valueResult{len};

   // Get the raw data to pass to cub
   CHAIDataGetter<ValueT, Exec> valueGetter {};
   CHAIDataGetter<KeyT, Exec> keyGetter {};

   auto * rawKeyData = keyGetter.getRawArrayData(keys) + start;
   auto * rawValueData = valueGetter.getRawArrayData(values) + start;

   auto * rawKeyResult = keyGetter.getRawArrayData(keyResult);
   auto * rawValueResult = valueGetter.getRawArrayData(valueResult);

   // Get the temp storage length
   char * d_temp_storage = nullptr;
   size_t temp_storage_bytes = 0;

   // When called with a nullptr for temp storage, this returns how much
   // temp storage should be allocated.
   if (len > 0) {
#if defined(__CUDACC__)
      cub::DeviceRadixSort::SortPairs((void *)d_temp_storage, temp_storage_bytes,
                                      rawKeyData, rawKeyResult,
                                      rawValueData, rawValueResult,
                                      len);
#elif defined(__HIPCC__)
      hipcub::DeviceRadixSort::SortPairs((void *)d_temp_storage, temp_storage_bytes,
                                      rawKeyData, rawKeyResult,
                                      rawValueData, rawValueResult,
                                      len);
#endif
   }

   // Allocate the temp storage and get raw data to pass to cub
   host_device_ptr<char> tmpManaged {temp_storage_bytes};

   CHAIDataGetter<char, Exec> charGetter {};
   d_temp_storage = charGetter.getRawArrayData(tmpManaged);

   // Now sort
   if (len > 0) {
#if defined(__CUDACC__)
      cub::DeviceRadixSort::SortPairs((void *)d_temp_storage, temp_storage_bytes,
                                      rawKeyData, rawKeyResult,
                                      rawValueData, rawValueResult,
                                      len);
#elif defined(__HIPCC__)
      hipcub::DeviceRadixSort::SortPairs((void *)d_temp_storage, temp_storage_bytes,
                                      rawKeyData, rawKeyResult,
                                      rawValueData, rawValueResult,
                                      len);
#endif
   }

   // Get the result
   if (noCopy) {
      if (len > 0) {
         keys.free(); 
         values.free();
      }

      keys = keyResult;
      values = valueResult;
   }
   else {
      CARE_STREAM_LOOP(i, start, start + len) {
         keys[i] = keyResult[i];
         values[i] = valueResult[i];
      } CARE_STREAM_LOOP_END

      if (len > 0) {
         keyResult.free();
         valueResult.free();
      }
   }

   if (len > 0) {
      tmpManaged.free();
   }
}

template <typename T>
void IntersectKeyValueSorters(RAJADeviceExec exec, KeyValueSorter<T, RAJADeviceExec> sorter1, int size1,
                              KeyValueSorter<T, RAJADeviceExec> sorter2, int size2,
                              host_device_ptr<int> &matches1, host_device_ptr<int>& matches2,
                              int & numMatches) {
 
   
   int smaller = (size1 < size2) ? size1 : size2 ;
   int start1 = 0;
   int start2 = 0;

   numMatches = 0 ;
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

   host_device_ptr<int> smallerMatches, largerMatches;
   host_device_ptr<size_t> smallerKeys, largerKeys;
   int larger, smallStart, largeStart;
   host_device_ptr<const T> smallerArray, largerArray;
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

   host_device_ptr<int> searches{smaller+1};
   host_device_ptr<int> matched{smaller+1};
   CARE_STREAM_LOOP(i, 0, smaller+1) {
      searches[i] = i != smaller ? care::BinarySearch<T>(largerArray, largeStart, larger, smallerArray[i+smallStart]) : -1;
      matched[i] = i != smaller && searches[i] > -1;
   } CARE_STREAM_LOOP_END

   exclusive_scan<int, RAJADeviceExec>(matched, nullptr, smaller+1, RAJA::operators::plus<int>{}, 0, true);

   CARE_STREAM_LOOP(i, 0, smaller) {
      if (searches[i] > -1) {
         smallerMatches[matched[i]] = smallerKeys[i+smallStart];
         largerMatches[matched[i]] = largerKeys[searches[i]];
      }
   } CARE_STREAM_LOOP_END
   numMatches =  matched.pick(smaller);
   searches.free();
   matched.free();
   
   /* change the size of the array */
   if (numMatches == 0) {
      matches1.free();
      matches2.free();
   }
   else {
      matches1.realloc(numMatches);
      matches2.realloc(numMatches);
   }

}
#endif // defined(CARE_GPUCC)

// This assumes arrays have been sorted and unique. If they are not uniqued the GPU
// and CPU versions may have different behaviors (the index they match to may be different, 
// with the GPU implementation matching whatever binary search happens to land on, and the// CPU version matching the first instance. 

template <typename T>
void IntersectKeyValueSorters(RAJA::seq_exec exec, 
                              KeyValueSorter<T, RAJA::seq_exec> sorter1, int size1,
                              KeyValueSorter<T, RAJA::seq_exec> sorter2, int size2,
                              host_device_ptr<int> &matches1, host_device_ptr<int>& matches2, int & numMatches) {

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
   host_ptr<int> host_matches1 = matches1 ;
   host_ptr<int> host_matches2 = matches2 ;
   /* keys() and values() will allocate managed arrays for the keys and values,
    * respectively, if they were not previously allocated.
    * Check to see whether they were previously allocated. */
   bool sorter1KeysAllocated = sorter1.keysAllocated() ;
   bool sorter2KeysAllocated = sorter2.keysAllocated() ;
   bool sorter1ValuesAllocated = sorter1.valuesAllocated() ;
   bool sorter2ValuesAllocated = sorter2.valuesAllocated() ;
   host_ptr<size_t const> host_sorter1_key = sorter1.keys() ;
   host_ptr<size_t const> host_sorter2_key = sorter2.keys() ;
   host_ptr<T const> host_sorter1_value = sorter1.values() ;
   host_ptr<T const> host_sorter2_value = sorter2.values() ;

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

} // namespace care

#endif // !defined(_CARE_KEY_VALUE_SORTER_H_)

