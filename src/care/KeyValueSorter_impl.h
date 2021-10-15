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
CARE_INLINE void sortKeyValueArrays(host_device_ptr<KeyT> & keys,
                                    host_device_ptr<ValueT> & values,
                                    const size_t start, const size_t len,
                                    const bool noCopy)
{
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

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes keys and values by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename T>
CARE_INLINE void setKeyValueArraysFromArray(host_device_ptr<size_t> & keys,
                                            host_device_ptr<T> & values,
                                            const size_t len, const T* arr)
{
   CARE_SEQUENTIAL_LOOP(i, 0, len) {
      keys[i] = i;
      values[i] = arr[i];
   } CARE_SEQUENTIAL_LOOP_END
}

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes the KeyValueSorter by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename T>
CARE_INLINE void setKeyValueArraysFromManagedArray(host_device_ptr<size_t> & keys,
                                                   host_device_ptr<T> & values,
                                                   const size_t len, const host_device_ptr<const T>& arr)
{
   FUSIBLE_LOOP_STREAM(i, 0, len) {
      keys[i] = (size_t) i;
      values[i] = arr[i];
   } FUSIBLE_LOOP_STREAM_END
}

///////////////////////////////////////////////////////////////////////////
/// @author Jeff Keasler, Alan Dayton
/// @brief Eliminates duplicate values
/// Remove duplicate values from old key/value arrays.
/// Old key/value arrays should already be sorted by value.
/// New key/value arrays should be allocated to the old size.
/// @param[out] newKeys New key array with duplicates removed
/// @param[out] newValues New value array with duplicates removed
/// @param[in] oldKeys Old key array (key-value pairs sorted by value)
/// @param[in] oldValues Old value array (sorted)
/// @param[in] oldLen Length of old key/value array and initial length for new
/// @return newLen Length of new key/value arrays
///////////////////////////////////////////////////////////////////////////
template <typename T>
CARE_INLINE size_t eliminateKeyValueDuplicates(host_device_ptr<size_t>& newKeys,
                                               host_device_ptr<T>& newValues,
                                               const host_device_ptr<const size_t>& oldKeys,
                                               const host_device_ptr<const T>& oldValues,
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

template <typename T>
CARE_INLINE void IntersectKeyValueSorters(RAJADeviceExec exec,
                                          KeyValueSorter<T, RAJADeviceExec> sorter1, int size1,
                                          KeyValueSorter<T, RAJADeviceExec> sorter2, int size2,
                                          host_device_ptr<int> &matches1,
                                          host_device_ptr<int>& matches2,
                                          int & numMatches)
{
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

   care::exclusive_scan(RAJADeviceExec{}, matched, nullptr, smaller+1, 0, true);

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

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes keys and values by copying elements from the array
/// @param[out] keyValues - The key value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename T>
CARE_INLINE void setKeyValueArraysFromArray(host_device_ptr<_kv<T> > & keyValues,
                                            const size_t len, const T* arr)
{
   CARE_SEQUENTIAL_LOOP(i, 0, (int) len) {
      keyValues[i].key = i;
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
template <typename T>
CARE_INLINE void setKeyValueArraysFromManagedArray(host_device_ptr<_kv<T> > & keyValues,
                                                   const size_t len, const host_device_ptr<const T>& arr)
{
   FUSIBLE_LOOP_STREAM(i, 0, (int)len) {
      keyValues[i].key = i;
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
template <typename T>
CARE_INLINE size_t eliminateKeyValueDuplicates(host_device_ptr<_kv<T> > & keyValues, const size_t len)
{
   size_t newSize = len;
   if (len > 1) {
      CHAIDataGetter<_kv<T>, RAJA::seq_exec> getter {};
      _kv<T> * rawData = getter.getRawArrayData(keyValues);

      // First do a stable sort by value (preserve the original order
      // in the case of a tie)
      std::sort(rawData, rawData + len, cmpValsStable<T>);
      // TODO: investigate performance of std::stable_sort
      // std::stable_sort(rawData, rawData + len);

      // Then eliminate duplicates
      size_t lsize = len - 1;  /* adjust search range */
      size_t put = 0;
      size_t get = 0;

      while (get < lsize) {
         if (put != get) {
            memcpy(&rawData[put], &rawData[get], sizeof(struct _kv<T>));
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
         memmove(&rawData[put++], &rawData[lsize], sizeof(struct _kv<T>));
      }

      lsize = put;

      // Then sort by key to get the original ordering
      std::sort(rawData, rawData + lsize, cmpKeys<T>);

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
template <typename T>
CARE_INLINE void initializeKeyArray(host_device_ptr<size_t>& keys,
                                    const host_device_ptr<const _kv<T> >& keyValues, const size_t len)
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
template <typename T>
CARE_INLINE void initializeValueArray(host_device_ptr<T>& values,
                                      const host_device_ptr<const _kv<T> >& keyValues, const size_t len)
{
   CARE_STREAM_LOOP(i, 0, len) {
      values[i] = keyValues[i].value;
   } CARE_STREAM_LOOP_END

   return;
}



// This assumes arrays have been sorted and unique. If they are not uniqued the GPU
// and CPU versions may have different behaviors (the index they match to may be different, 
// with the GPU implementation matching whatever binary search happens to land on, and the// CPU version matching the first instance. 

template <typename T>
CARE_INLINE void IntersectKeyValueSorters(RAJA::seq_exec /* exec */, 
                                          KeyValueSorter<T, RAJA::seq_exec> sorter1, int size1,
                                          KeyValueSorter<T, RAJA::seq_exec> sorter2, int size2,
                                          host_device_ptr<int> &matches1,
                                          host_device_ptr<int>& matches2,
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

#endif // !defined(_CARE_KEY_VALUE_SORTER_IMPL_H_)

