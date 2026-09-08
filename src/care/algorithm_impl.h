//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_ALGORITHM_IMPL_H
#define CARE_ALGORITHM_IMPL_H

// This header includes the implementations of the CARE algorithms.
// In very large monolithic codes, including this in too many compilation
// units can cause linking issues (particularly device link issues) due to
// the file size. In that case, external template instantiation should be
// used (this requires CARE_ENABLE_EXTERN_INSTANTIATE to be turned ON in the cmake
// configuration): this file should only be included in the compilation unit containing
// the instantiation and algorithm.h (along with the extern template
// declarations) should be included everywhere else.

#include "care/algorithm_decl.h"
#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/scan.h"
#include "care/compress_algorithm_impl.h"

// Other library headers
#if defined(__CUDACC__)
#include "cub/cub.cuh"
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
#endif

#if defined(__HIPCC__)
#include "hipcub/hipcub.hpp"
#endif

namespace care {


/************************************************************************
 * Function  : IntersectArrays<A,RAJAExec>
 * Author(s) : Peter Robinson, based on IntersectGlobalIDArrays by Al Nichols
 * Purpose   : Given two arrays of unique elements of type A sorted in  ascending order, this
 *             routine returns the number of matching elements, and
 *             two arrays of indices: the indices in the first array
 *             at which intersection occurs, and the corresponding set
 *             of indices in the second array.
 *             This is the parallel overload of this method.
 * Note      : matches are given as offsets from start1 and start2. So, if a match occurs at arr1[2] with
 *             start1=0, then matches1 will contain 2. However, if start1 was 1, then matches will contain 2-start1=1.
 ************************************************************************/
#ifdef CARE_PARALLEL_DEVICE
template <typename T>
CARE_INLINE void IntersectArrays(RAJADeviceExec,
                                 care::host_device_ptr<const T> arr1, int size1, int start1,
                                 care::host_device_ptr<const T> arr2, int size2, int start2,
                                 care::host_device_ptr<int> &matches1,
                                 care::host_device_ptr<int> &matches2,
                                 int *numMatches)
{
   *numMatches = 0 ;
   int smaller = (size1 < size2) ? size1 : size2 ;

   if (smaller <= 0) {
      matches1 = nullptr ;
      matches2 = nullptr ;
      return ;
   }
   else {
      matches1.alloc(smaller);
      matches2.alloc(smaller);
   }

   /* This algorithm assumes that the nodelists are sorted and unique */
#ifdef CARE_DEBUG
   bool checkIsSorted = true ;
#else
   bool checkIsSorted = false;
#endif

   if (checkIsSorted) {
      const char* funcname = "IntersectArrays" ;

      // allowDuplicates is false for these checks by default.
      care::host_device_ptr<const T> slice1(arr1.slice(start1));
      care::host_device_ptr<const T> slice2(arr2.slice(start2));

      checkSorted<T>(slice1, size1, funcname, "arr1") ;
      checkSorted<T>(slice2, size2, funcname, "arr2") ;
   }

   care::host_device_ptr<int> smallerMatches, largerMatches;
   int larger, smallStart, largeStart;
   care::host_device_ptr<const T> smallerArray, largerArray;

   if (smaller == size1) {
      smallerArray = arr1;
      largerArray = arr2;
      larger = size2;
      smallStart = start1;
      largeStart = start2;
      smallerMatches = matches1;
      largerMatches = matches2;
   }
   else {
      smallerArray = arr2;
      largerArray = arr1;
      larger = size1;
      smallStart = start2;
      largeStart = start1;
      smallerMatches = matches2;
      largerMatches = matches1;
   }

   /* this avoid thrust and the inherent memory allocation overhead associated with it */
   care::host_device_ptr<int> searches(smaller + 1,"IntersectArrays searches");
   care::host_device_ptr<int> matched(smaller + 1, "IntersectArrays matched");

   CARE_STREAM_LOOP(i, 0, smaller + 1) {
      searches[i] = i != smaller ? BinarySearch<T>(largerArray, largeStart, larger, smallerArray[i + smallStart]) : -1;
      matched[i] = i != smaller && searches[i] > -1;
   } CARE_STREAM_LOOP_END

   care::exclusive_scan(RAJAExec{}, matched, nullptr, smaller + 1, 0, true);

   CARE_STREAM_LOOP(i, 0, smaller) {
      if (searches[i] > -1) {
         // matches reported relative to smallStart and largeStart
         smallerMatches[matched[i]] = i;
         largerMatches[matched[i]] = searches[i] - largeStart;
      }
   } CARE_STREAM_LOOP_END

   *numMatches = matched.pick(smaller);

   searches.free(); 
   matched.free();

   /* change the size of the array */
   /* (reallocing to a size of zero should be the same as freeing
    * the object, but insight doesn't seem to think so... hence
    * the extra check here with an explicit free */
   if (*numMatches == 0) {
      matches1.free();
      matches2.free();
      matches1 = nullptr;
      matches2 = nullptr;
   }
   else {
      matches1.realloc(*numMatches);
      matches2.realloc(*numMatches);
   }
}

template <typename T>
CARE_INLINE void IntersectArrays(RAJADeviceExec exec,
                                 care::host_device_ptr<T> arr1, int size1, int start1,
                                 care::host_device_ptr<T> arr2, int size2, int start2,
                                 care::host_device_ptr<int> &matches1,
                                 care::host_device_ptr<int> &matches2,
                                 int *numMatches)
{
   IntersectArrays<T>(exec,
                      care::host_device_ptr<const T>(arr1), size1, start1,
                      care::host_device_ptr<const T>(arr2), size2, start2,
                      matches1, matches2,
                      numMatches);
}

#endif // defined(CARE_PARALLEL_DEVICE)

/************************************************************************
 * Function  : IntersectArraysSeq
 * Author(s) : Benjamin Liu, after Peter Robinson, Al Nichols
 * Purpose   : Helper function for IntersectArrays<A>(RAJA::seq_exec,...)
 * Note      : matches1 and matches2 should already be allocated to the
 *             smaller of size1 and size2 and should be reallocated to
 *             *numMatches after this call.
 ************************************************************************/
template <typename T>
CARE_INLINE void IntersectArraysSeq(const T *arr1, int size1, int start1,
                                    const T *arr2, int size2, int start2,
                                    int *matches1,
                                    int *matches2,
                                    int *numMatches)
{
   const T * a1 = arr1 + start1;
   const T * a2 = arr2 + start2;

   /* This algorithm assumes that the nodelists are sorted and unique */
#ifdef CARE_DEBUG
   bool checkIsSorted = true ;
#else
   bool checkIsSorted = false;
#endif

   if (checkIsSorted) {
      const char* funcname = "IntersectArrays" ;

      // allowDuplicates is false for this check by default
      checkSorted<T>(a1, size1, funcname, "arr1") ;
      checkSorted<T>(a2, size2, funcname, "arr2") ;
   }

   int i, j;
   i = j = 0 ;

   while (i < size1 && j < size2) {
      if (a1[i] < a2[j]) {
         ++i;
      }
      else if (a2[j] < a1[i]) {
         ++j;
      }
      else {
         matches1[(*numMatches)] = i;
         matches2[(*numMatches)] = j;
         (*numMatches)++;
         ++i;
         ++j;
      }
   }
}

/************************************************************************
 * Function  : IntersectArrays<A,RAJA::seq_exec>
 * Author(s) : Peter Robinson, based on IntersectGlobalIDArrays by Al Nichols
 * Purpose   : Given two arrays of unique elements of type A sorted in  ascending order, this
 *             routine returns the number of matching elements, and
 *             two arrays of indices: the indices in the first array
 *             at which intersection occurs, and the corresponding set
 *             of indices in the second array. 
 *             This is the sequential overload of this method, with a care::host_ptr API.
 * Note      : matches are given as offsets from start1 and start2. So, if a match occurs at arr1[2] with
 *             start1=0, then matches1 will contain 2. However, if start1 was 1, then matches will contain 2-start1=1.
 * Note      : The raw pointers contained in matches1 and matches2
 *             should be deallocated by the caller using free.
 ************************************************************************/
template <typename T>
CARE_INLINE void IntersectArrays(RAJA::seq_exec,
                                 care::host_ptr<const T> arr1, int size1, int start1,
                                 care::host_ptr<const T> arr2, int size2, int start2,
                                 care::host_ptr<int> &matches1,
                                 care::host_ptr<int> &matches2,
                                 int *numMatches)
{
   *numMatches = 0 ;
   int smaller = (size1 < size2) ? size1 : size2 ;

   if (smaller <= 0) {
      matches1 = nullptr ;
      matches2 = nullptr ;
      return ;
   }
   else {
      matches1 = (int*) std::malloc(smaller * sizeof(int));
      matches2 = (int*) std::malloc(smaller * sizeof(int));
   }

   IntersectArraysSeq(arr1.cdata(), size1, start1, arr2.cdata(), size2, start2,
                      matches1.data(), matches2.data(), numMatches) ;

   /* change the size of the array */
   if (*numMatches == 0) {
      std::free(matches1.data());
      std::free(matches2.data());
      matches1 = nullptr;
      matches2 = nullptr;
   }
   else {
      matches1 = (int*) std::realloc(matches1.data(), *numMatches * sizeof(int));
      matches2 = (int*) std::realloc(matches2.data(), *numMatches * sizeof(int));
   }
}

template <typename T>
CARE_INLINE void IntersectArrays(RAJA::seq_exec exec,
                                 care::host_ptr<T> arr1, int size1, int start1,
                                 care::host_ptr<T> arr2, int size2, int start2,
                                 care::host_ptr<int> &matches1,
                                 care::host_ptr<int> &matches2,
                                 int *numMatches)
{
   IntersectArrays<T>(exec,
                      care::host_ptr<const T>(arr1), size1, start1,
                      care::host_ptr<const T>(arr2), size2, start2,
                      matches1, matches2, numMatches);
}

/************************************************************************
 * Function  : IntersectArrays<A,RAJA::seq_exec>
 * Author(s) : Peter Robinson
 * Purpose   : Given two arrays of unique elements of type A sorted in  ascending order, this
 *             routine returns the number of matching elements, and
 *             two arrays of indices: the indices in the first array
 *             at which intersection occurs, and the corresponding set
 *             of indices in the second array.
 *             This is the sequential overload of this method, with a host_device pointer API.
 * Note      : matches are given as offsets from start1 and start2. So, if a match occurs at arr1[2] with
 *             start1=0, then matches1 will contain 2. However, if start1 was 1, then matches will contain 2-start1=1.
 ************************************************************************/
template <typename T>
CARE_INLINE void IntersectArrays(RAJA::seq_exec exec,
                                 care::host_device_ptr<const T> arr1, int size1, int start1,
                                 care::host_device_ptr<const T> arr2, int size2, int start2,
                                 care::host_device_ptr<int> &matches1,
                                 care::host_device_ptr<int> &matches2,
                                 int *numMatches)
{
   *numMatches = 0 ;
   int smaller = (size1 < size2) ? size1 : size2 ;

   if (smaller <= 0) {
      matches1 = nullptr ;
      matches2 = nullptr ;
      return ;
   }
   else {
      matches1.alloc(smaller);
      matches2.alloc(smaller);
   }

   IntersectArraysSeq(arr1.cdata(), size1, start1, arr2.cdata(), size2, start2,
                      matches1.data(), matches2.data(), numMatches) ;

   /* change the size of the array */
   if (*numMatches == 0) {
      matches1.free();
      matches2.free();
   }
   else {
      matches1.realloc(*numMatches);
      matches2.realloc(*numMatches);
   }

   return;
}

template <typename T>
CARE_INLINE void IntersectArrays(RAJA::seq_exec exec,
                                 care::host_device_ptr<T> arr1, int size1, int start1,
                                 care::host_device_ptr<T> arr2, int size2, int start2,
                                 care::host_device_ptr<int> &matches1,
                                 care::host_device_ptr<int> &matches2,
                                 int *numMatches)
{
   IntersectArrays(exec,
                   care::host_device_ptr<const T>(arr1), size1, start1,
                   care::host_device_ptr<const T>(arr2), size2, start2,
                   matches1, matches2, numMatches);
}


#ifdef CARE_PARALLEL_DEVICE
/************************************************************************
 * Function  : uniqArray
 * Author(s) : Peter Robinson
 * Purpose   : GPU version of uniqArray, implements uniq using an exclusive
 *             scan.
  ************************************************************************/
template <typename T>
CARE_INLINE void uniqArray(RAJADeviceExec, care::host_device_ptr<const T>  Array, size_t len,
                           care::host_device_ptr<T> & outArray, int & outLen)
{
   care::host_device_ptr<int> uniq(len+1,"uniqArray uniq");
   fill_n(uniq, len+1, 0);
   CARE_STREAM_LOOP(i, 0, len) {
      uniq[i] = (int) ((i == len-1) || (Array[i] < Array[i+1] || Array[i+1] < Array[i])) ;
   } CARE_STREAM_LOOP_END

   care::exclusive_scan(RAJADeviceExec{}, uniq, nullptr, len+1, 0, true);
   int numUniq;
   uniq.pick(len, numUniq);
   care::host_device_ptr<T> & tmp = outArray;
   tmp.alloc(numUniq);
   CARE_STREAM_LOOP(i, 0, len) {
      if ((i == len-1) || (Array[i] < Array[i+1] || Array[i+1] < Array[i])) {
         tmp[uniq[i]] = Array[i];
      }
   } CARE_STREAM_LOOP_END
   uniq.free();
   outLen = numUniq;
   return;
}

/************************************************************************
 * Function  : uniqArray
 * Author(s) : Peter Robinson
 * Purpose   : GPU version of uniqArray, implements uniq using an exclusive
 *             scan.
  ************************************************************************/
template <typename T>
CARE_INLINE int uniqArray(RAJADeviceExec exec, care::host_device_ptr<T> & Array, size_t len, bool noCopy)
{
   care::host_device_ptr<T> tmp;
   int newLen;
   uniqArray<T>(exec, Array, len, tmp, newLen);
   if (noCopy) {
      Array.free();
      Array = tmp;
   }
   else {
      ArrayCopy<T>(exec, Array, tmp, newLen);
      tmp.free();
   }
   return newLen;
}

#endif // defined(CARE_PARALLEL_DEVICE)

/************************************************************************
 * Function  : uniqArray
 * Author(s) : Peter Robinson
 * Purpose   : CPU version of uniqArray.
  ************************************************************************/
template <typename T>
CARE_INLINE void uniqArray(RAJA::seq_exec, care::host_device_ptr<const T> Array, size_t len,
                           care::host_device_ptr<T> & outArray, int & newLen)
{
   CHAIDataGetter<const T, RAJA::seq_exec> getter {};
   auto * rawData = getter.getConstRawArrayData(Array);
   newLen = 0 ;
   care::host_ptr<T> arrout = nullptr ;
   outArray = nullptr;

   if (len != 0) {
      size_t i=0  ;

      /* alloc some space, we realloc later */
      outArray = care::host_device_ptr<T>(len,"uniq_outArray");
      arrout = outArray;

      while (i < len) {
         /* copy the unique value into the array */
         arrout[newLen] = (T)rawData[i] ;

         /* skip over all the redundant elements */
         while ((i<len) && ((T)rawData[i] == arrout[newLen])) {
            i++ ;
         }

         ++newLen;
      }
      outArray.realloc(newLen);
   }
   return;
}

/************************************************************************
 * Function  : uniqArray
 * Author(s) : Peter Robinson
 * Purpose   : CPU version of uniqArray, with in-place semantics. Set noCopy to true
 *             if you don't care about data left at the end of the array after the uniq.
  ************************************************************************/
template <typename T>
CARE_INLINE int uniqArray(RAJA::seq_exec exec, care::host_device_ptr<T> & Array, size_t len, bool noCopy)
{
   int newLength = 0;
   if (len > 0) {
      care::host_device_ptr<T> tmp;
      uniqArray<T>(exec, Array, len, tmp, newLength);
      if (noCopy) {
         Array.free();
         Array = tmp;
      }
      else {
         ArrayCopy<T>(RAJA::seq_exec {}, Array, reinterpret_cast<care::host_device_ptr<const T> &>(tmp), newLength);
         tmp.free();
      }
   }
   return newLength;
}

#if defined(CARE_PARALLEL_DEVICE)
#if defined(CARE_GPUCC)

/************************************************************************
 * Function  : sortArray
 * Author(s) : Peter Robinson
 * Purpose   : GPU version of sortArray.
  ************************************************************************/

// TODO: Use if constexpr and std::is_arithmetic_v when c++17 support is required
template <typename T>
CARE_INLINE
std::enable_if_t<std::is_arithmetic<typename CHAIDataGetter<T, RAJADeviceExec>::raw_type>::value, void>
sortArray(RAJADeviceExec, care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy)
{
   radixSortArray(Array, len, start, noCopy);
}

#if defined(__HIPCC__) || (defined(__CUDACC__) && defined(CUB_MAJOR_VERSION) && defined(CUB_MINOR_VERSION) && (CUB_MAJOR_VERSION >= 2 || (CUB_MAJOR_VERSION == 1 && CUB_MINOR_VERSION >= 14)))
template <typename T>
CARE_INLINE
std::enable_if_t<!std::is_arithmetic<typename CHAIDataGetter<T, RAJADeviceExec>::raw_type>::value, void>
sortArray(RAJADeviceExec, care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy)
{
   mergeSortArray(Array, len, start, noCopy);
}
#endif

/************************************************************************
 * Function  : radixSortArray
 * Author(s) : Peter Robinson
 * Purpose   : ManagedArray API to [hip]cub::DeviceRadixSort::SortKeys
  ************************************************************************/
template <typename T>
CARE_INLINE void radixSortArray(care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy)
{
   CHAIDataGetter<T, RAJADeviceExec> getter {};
   CHAIDataGetter<char, RAJADeviceExec> charGetter {};
   care::host_device_ptr<T> result(len,"radix_sort_result");
   const auto * rawData = getter.getConstRawArrayData(Array) + start;
   auto * rawResult = getter.getRawArrayData(result);
   // get the temp storage length
   char * d_temp_storage = nullptr;
   size_t temp_storage_bytes = 0;
   if (len > 0) {
#if defined(__CUDACC__)
      cub::DeviceRadixSort::SortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, rawResult, len);
#elif defined(__HIPCC__)
      hipcub::DeviceRadixSort::SortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, rawResult, len);
#endif
   }
   // allocate the temp storage

   care::host_device_ptr<char> tmpManaged(temp_storage_bytes, "radix_sort_tmpManaged");
   d_temp_storage = charGetter.getRawArrayData(tmpManaged);

   // do the sort
   if (len > 0) {
#if defined(CHAI_THIN_GPU_ALLOCATE)
      chai::ArrayManager::getInstance()->setExecutionSpace(chai::GPU);
#endif

#if defined(__CUDACC__)
      cub::DeviceRadixSort::SortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, rawResult, len);
#elif defined(__HIPCC__)
      hipcub::DeviceRadixSort::SortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, rawResult, len);
#endif

#if defined(CHAI_THIN_GPU_ALLOCATE)
      chai::ArrayManager::getInstance()->setExecutionSpace(chai::NONE);
#endif
   }

   // cleanup
   if (noCopy) {
      if (len > 0) {
         Array.free();
      }
      Array = care::host_device_ptr<T>(chai::ManagedArray<T>(result));
   }
   else {
      ArrayCopy<T>(RAJADeviceExec{}, care::host_device_ptr<T>(Array), result, len, start, 0);
      if (len > 0) {
         result.free();
      }
   }
   if (len > 0) {
      tmpManaged.free();
   }
}

#if defined(__HIPCC__) || (defined(__CUDACC__) && defined(CUB_MAJOR_VERSION) && defined(CUB_MINOR_VERSION) && (CUB_MAJOR_VERSION >= 2 || (CUB_MAJOR_VERSION == 1 && CUB_MINOR_VERSION >= 14)))
/************************************************************************
 * Function  : mergeSortArray
 * Author(s) : Peter Robinson
 * Purpose   : ManagedArray API to [hip]cub::DeviceMergeSort::StableSortKeys.
  ************************************************************************/
template <typename T>
CARE_INLINE void mergeSortArray(care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy)
{
   CHAIDataGetter<T, RAJADeviceExec> getter {};
   CHAIDataGetter<char, RAJADeviceExec> charGetter {};
   auto * rawData = getter.getRawArrayData(Array) + start;
   // get the temp storage length
   char * d_temp_storage = nullptr;
   size_t temp_storage_bytes = 0;
   auto custom_comparator = [] CARE_HOST_DEVICE (const T & lhs, const T & rhs) {
      return lhs < rhs;
   };
   if (len > 0) {
#if defined(__CUDACC__)
      cub::DeviceMergeSort::StableSortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, len, custom_comparator);
#elif defined(__HIPCC__)
      hipcub::DeviceMergeSort::StableSortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, len, custom_comparator);
#endif
   }
   // allocate the temp storage

   care::host_device_ptr<char> tmpManaged(temp_storage_bytes, "merge_sort_tmpManaged");
   d_temp_storage = charGetter.getRawArrayData(tmpManaged);

   // do the sort
   if (len > 0) {
#if defined(CHAI_THIN_GPU_ALLOCATE)
      chai::ArrayManager::getInstance()->setExecutionSpace(chai::GPU);
#endif

#if defined(__CUDACC__)
      cub::DeviceMergeSort::StableSortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, len, custom_comparator);
#elif defined(__HIPCC__)
      hipcub::DeviceMergeSort::StableSortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, len, custom_comparator);
#endif

#if defined(CHAI_THIN_GPU_ALLOCATE)
      chai::ArrayManager::getInstance()->setExecutionSpace(chai::NONE);
#endif
   }

   // cleanup
   if (len > 0) {
      tmpManaged.free();
   }
}
#endif
#else // defined(CARE_GPUCC)

// TODO openMP parallel implementation
template <typename T>
CARE_INLINE void sortArray(RAJADeviceExec, care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy)
{
   sortArray(RAJA::seq_exec{}, Array, len, start, noCopy);
}

#endif // defined(CARE_GPUCC)

template <typename T>
CARE_INLINE void sortArray(RAJADeviceExec exec, care::host_device_ptr<T> & Array, size_t len)
{
   sortArray(exec, Array, len, 0, false);
}

#endif // defined(CARE_PARALLEL_DEVICE)

/************************************************************************
 * Function  : sortArray
 * Author(s) : Peter Robinson
 * Purpose   : CPU version of sortArray. Calls std::sort
  ************************************************************************/
template <typename T>
CARE_INLINE void sortArray(RAJA::seq_exec, care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy)
{
   CHAIDataGetter<T, RAJA::seq_exec> getter {};
   auto * rawData = getter.getRawArrayData(Array)+start;
   std::sort(rawData, rawData+len);
   (void) noCopy;
}

template <typename T>
CARE_INLINE void sortArray(RAJA::seq_exec, care::host_device_ptr<T> &Array, size_t len)
{
   CHAIDataGetter<T, RAJA::seq_exec> getter {};
   auto * rawData = getter.getRawArrayData(Array);
   std::sort(rawData, rawData+len);
}

/************************************************************************
* Function  : sort_uniq(<T>_ptr)
* Author(s) : Peter Robinson
* Purpose   : Sorts and uniques an array.
* @param    : e Execution policy
* @param    : array: pointer to an array to sort and uniq
* @param    : len: length of the array
* @param    : noCopy. If true, implementation is free to store a completely new array at pointer
*             If false, implementation will not mess with the underlying allocation of the new array
**************************************************************************/
template <typename T, typename Exec>
CARE_INLINE void sort_uniq(Exec e, care::host_device_ptr<T> * array, int * len, bool noCopy)
{
   if (*len == 0) {
      if (noCopy && *array != nullptr) {
         array->free();
         *array = nullptr;
      }

      return;
   }

   /* first sort the array */
   sortArray<T>(e, *array, *len, 0, noCopy);
   /* then unique it */
   *len = uniqArray<T>(e, *array, *len, noCopy);
}


template <typename T>
CARE_INLINE void ExpandArrayInPlace(RAJA::seq_exec, care::host_device_ptr<T> array,
                                    care::host_device_ptr<int const> indexSet, int length)
{
   if (length > 0) {
      care::host_ptr<int const> host_indexSet = indexSet ;
      care::host_ptr<T> host_array =array;
      for (int i = length-1; i >= 0; --i) {
         int idx = host_indexSet[i] ;
         host_array[idx] = host_array[i] ;
      }
   }
}

#ifdef CARE_PARALLEL_DEVICE
template <typename T>
CARE_INLINE void ExpandArrayInPlace(RAJADeviceExec, care::host_device_ptr<T> array,
                                    care::host_device_ptr<int const> indexSet, int length)
{
   if (length > 0) {
      care::host_device_ptr<T> array_copy(length, "ExpandArrayInPlace array_copy");
      ArrayCopy<T>(array_copy, array, length);
      CARE_STREAM_LOOP(i, 0, length) {
         int nL = indexSet[i] ;
         array[nL] = array_copy[i] ;
      } CARE_STREAM_LOOP_END
      array_copy.free();
   }
}

#endif // defined(CARE_PARALLEL_DEVICE)

/************************************************************************
 * Function  : fill_n
 * Author(s) : Peter Robinson
 * Purpose   : Fills a ManagedArray with the value given.
 * ************************************************************************/
template <class T, class Size, class U>
CARE_INLINE void fill_n(care::host_device_ptr<T> arr, Size n, const U& val)
{
   CARE_STREAM_LOOP(i, 0, n) {
      arr[i] = val;
   } CARE_STREAM_LOOP_END
}

/************************************************************************
 * Function  : copy_n
 * Author(s) : Alan Dayton
 * Purpose   : Copies one ManagedArray into another.
 * ************************************************************************/
template <class T, class Size, class U>
CARE_INLINE void copy_n(care::host_device_ptr<const T> in, Size n,
                        care::host_device_ptr<U> out)
{
   CARE_STREAM_LOOP(i, 0, n) {
      out[i] = in[i];
   } CARE_STREAM_LOOP_END
}

/************************************************************************
 * Function  : copy_n
 * Author(s) : Alan Dayton
 * Purpose   : Copies one ManagedArray into another. Non-const overload.
 * ************************************************************************/
template <class T, class Size, class U>
CARE_INLINE void copy_n(care::host_device_ptr<T> in, Size n,
                        care::host_device_ptr<U> out)
{
   copy_n(care::host_device_ptr<const T>(in), n, out);
}

/************************************************************************
 * Function  : ArrayMin
 * Author(s) : Peter Robinson
 * Purpose   : Returns the minimum value in a ManagedArray
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE T ArrayMin(care::host_device_ptr<const T> arr, int n, T initVal, int startIndex)
{
   RAJAReduceMin<T> min { initVal };
   CARE_REDUCE_LOOP(k, startIndex, n) {
      min.min(arr[k]);
   } CARE_REDUCE_LOOP_END
   return (T)min;
}

template <typename T, typename Exec>
CARE_INLINE T ArrayMin(care::host_device_ptr<T> arr, int n, T initVal, int startIndex)
{
   return ArrayMin<T, Exec>((care::host_device_ptr<const T>)arr, n, initVal, startIndex);
}


/************************************************************************
 * Function  : ArrayMin
 * Author(s) : Benjamin Liu, after Peter Robinson
 * Purpose   : Returns the minimum value in a ManagedArray.
 *             Wraps above care::host_device_ptr implementation with a care::host_ptr API.
 * ************************************************************************/
template <typename T>
CARE_INLINE T ArrayMin(care::host_ptr<const T> arr, int n, T initVal, int startIndex)
{
   return ArrayMin<T, RAJA::seq_exec>(care::host_device_ptr<const T>(arr.cdata(), n, "ArrayMinTmp"), n, initVal, startIndex);
}

template <typename T>
CARE_INLINE T ArrayMin(care::host_ptr<T> arr, int n, T initVal, int startIndex)
{
   return ArrayMin<T>((care::host_ptr<const T>)arr, n, initVal, startIndex);
}

/************************************************************************
 * Function  : ArrayMinLoc
 * Author(s) : Peter Robinson
 * Purpose   : Returns the minimum value in a ManagedArray
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE T ArrayMinLoc(care::host_device_ptr<const T> arr, int n, T initVal, int & loc)
{
   RAJAReduceMinLoc<T> min { initVal, -1 };
   CARE_REDUCE_LOOP(k, 0, n) {
      min.minloc(arr[k], k);
   } CARE_REDUCE_LOOP_END
   loc = min.getLoc();
   return (T)min;
}

/************************************************************************
 * Function  : ArrayMax
 * Author(s) : Peter Robinson
 * Purpose   : Returns the maximum value in a ManagedArray
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE T ArrayMax(care::host_device_ptr<const T> arr, int n, T initVal, int startIndex)
{
   RAJAReduceMax<T> max { initVal };
   CARE_REDUCE_LOOP(k, startIndex, n) {
      max.max(arr[k]);
   } CARE_REDUCE_LOOP_END
   return (T)max;
}

template <typename T, typename Exec>
CARE_INLINE T ArrayMax(care::host_device_ptr<T> arr, int n, T initVal, int startIndex)
{
   return ArrayMax<T, Exec>((care::host_device_ptr<const T>)arr, n, initVal, startIndex);
}

/************************************************************************
 * Function  : ArrayMax
 * Author(s) : Peter Robinson
 * Purpose   : Returns the maximum value in a ManagedArray.
 *             Wraps above care::host_device_ptr implementation with a care::host_ptr API.
 * ************************************************************************/
template <typename T>
CARE_INLINE T ArrayMax(care::host_ptr<const T> arr, int n, T initVal, int startIndex)
{
   return ArrayMax<T,RAJA::seq_exec>(care::host_device_ptr<const T>(arr.cdata(), n, "ArrayMaxTmp"), n, initVal, startIndex);
}

template <typename T>
CARE_INLINE T ArrayMax(care::host_ptr<T> arr, int n, T initVal, int startIndex)
{
   return ArrayMax<T>((care::host_ptr<const T>)arr, n, initVal, startIndex);
}

/************************************************************************
 * Function  : ArrayMaxLoc
 * Author(s) : Peter Robinson
 * Purpose   : Returns the maximum value in a ManagedArray
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE T ArrayMaxLoc(care::host_device_ptr<const T> arr, int n, T initVal, int & loc)
{
   RAJAReduceMaxLoc<T> max { initVal, -1 };
   CARE_REDUCE_LOOP(k, 0, n) {
      max.maxloc(arr[k], k);
   } CARE_REDUCE_LOOP_END
   loc = max.getLoc();
   return (T)max;
}

/************************************************************************
 * Function  : ArrayFind
 * Author(s) : Rob Neely, Alan Dayton
 * Purpose   : Returns the index of the first element in the the array
 *             that equals the value, or -1 if no match is found.
 ************************************************************************/
template <typename T>
CARE_INLINE int ArrayFind(care::host_device_ptr<const T> arr, const int len, const T val, const int start)
{
   RAJAReduceMin<int> min { len };

   CARE_REDUCE_LOOP(i, start, len) {
      if (arr[i] == val) {
         min.min(i);
      }
   } CARE_REDUCE_LOOP_END

   int result = (int) min;

   if (result == len) {
      return -1 ;
   }
   else {
      return result;
   }
}

/************************************************************************
 * Function  : ArrayMinMax
 * Author(s) : Peter Robinson
 * Purpose   : Stores Minimum / Maximum values of arr (as a double) in outMin / outMax;
 *             If mask was such that no values were compared, returns 0, outMin will be -DBL_MAX, outMax will be DBL_MAX
 *             Otherwise, returns 1.
 * ************************************************************************/
template <typename T, typename ReducerType, typename RAJAExec>
CARE_INLINE int ArrayMinMax(care::host_device_ptr<const T> arr,
                            care::host_device_ptr<int const> mask,
                            int n, double *outMin, double *outMax)
{
   bool result = false;
   ReducerType minVal, maxVal;
   if (arr) {
      RAJAReduceMax<ReducerType> max { std::numeric_limits<ReducerType>::lowest() };
      RAJAReduceMin<ReducerType> min { std::numeric_limits<ReducerType>::max() };
      if (mask) {
         CARE_REDUCE_LOOP(i, 0, n) {
            if (mask[i]) {
               min.min((ReducerType) arr[i]);
               max.max((ReducerType) arr[i]);
            }
         } CARE_REDUCE_LOOP_END
         minVal = (ReducerType) min;
         maxVal = (ReducerType) max;
         if (minVal!= std::numeric_limits<ReducerType>::max() ||
             maxVal != std::numeric_limits<ReducerType>::lowest()) {
            result = true;
         }
      }
      else {
         CARE_REDUCE_LOOP(i, 0, n) {
            min.min((ReducerType)arr[i]);
            max.max((ReducerType)arr[i]);
         } CARE_REDUCE_LOOP_END
         minVal = (ReducerType) min;
         maxVal = (ReducerType) max;
         result = true;
      }
   }
   if (result) {
      *outMin = (double) minVal;
      *outMax = (double) maxVal;
   }
   else {
      *outMin = -DBL_MAX;
      *outMax = +DBL_MAX;
   }

   return (int) result;
}

template <typename T, typename ReducerType, typename RAJAExec>
CARE_INLINE int ArrayMinMax(care::host_device_ptr<T> arr,
                            care::host_device_ptr<int> mask,
                            int n, double *outMin, double *outMax)
{
 return ArrayMinMax<T, ReducerType, RAJAExec>((care::host_device_ptr<const T>)arr, (care::host_device_ptr<int const>)mask, n, outMin, outMax);
}

#if CARE_HAVE_LLNL_GLOBALID

template <typename Exec>
CARE_INLINE int ArrayMinMax(care::host_device_ptr<const globalID> arr,
                            care::host_device_ptr<int const> mask,
                            int n, double *outMin, double *outMax)
{
   return ArrayMinMax<globalID, GIDTYPE, Exec>(arr, mask, n, outMin, outMax);
}

#endif // CARE_HAVE_LLNL_GLOBALID

/************************************************************************
 * Function  : ArrayCount
 * Author(s) : Peter Robinson
 * Purpose   : Returns count of occurence of val in arr.
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE int ArrayCount(care::host_device_ptr<const T> arr, int length, T val)
{
   RAJAReduceSum<int> count { 0 };
   CARE_REDUCE_LOOP(k, 0, length) {
      count += (T) arr[k] == val;
   } CARE_REDUCE_LOOP_END
   return (int) count ;
}

/************************************************************************
 * Function  : ArraySum
 * Author(s) : Peter Robinson
 * Purpose   : Returns the sum of all values in a ManagedArray
 * ************************************************************************/
template <typename T, typename ReduceType, typename Exec, typename ReturnType>
CARE_INLINE ReturnType ArraySum(care::host_device_ptr<const T> arr, int n, T initVal)
{
   ReduceType iVal = initVal;
   RAJAReduceSum<ReduceType> sum { iVal };
   CARE_REDUCE_LOOP(k, 0, n) {
      sum += arr[k];
   } CARE_REDUCE_LOOP_END
   return (ReturnType) (ReduceType) sum;
}

/************************************************************************
 * Function  : ArraySumSubset
 * Author(s) : Peter Robinson
 * Purpose   : Returns the sum of values in arr at indices in subset.
 * Note      : length n refers to length of subset, not array
 * ************************************************************************/
template <typename T, typename ReduceType, typename Exec, typename ReturnType>
CARE_INLINE ReturnType ArraySumSubset(care::host_device_ptr<const T> arr,
                             care::host_device_ptr<int const> subset, int n, T initVal)
{
   ReduceType iVal = initVal;
   RAJAReduceSum<ReduceType> sum { iVal };
   CARE_REDUCE_LOOP(k, 0, n) {
      sum += arr[subset[k]];
   } CARE_REDUCE_LOOP_END
   return (ReturnType) (ReduceType) sum;
}

/************************************************************************
 * Function  : ArrayMaskedSumSubset
 * Author(s) : Peter Robinson
 * Purpose   : Returns the sum of values in arr at indices in subset.
 * ************************************************************************/
template <typename T, typename ReduceType, typename Exec, typename ReturnType>
CARE_INLINE ReturnType ArrayMaskedSumSubset(care::host_device_ptr<const T> arr,  
                                   care::host_device_ptr<int const> mask,
                                   care::host_device_ptr<int const> subset,
                                   int n, T initVal)
{
   ReduceType iVal =initVal;
   RAJAReduceSum<ReduceType> sum { iVal };
   CARE_REDUCE_LOOP(k, 0, n) {
      int ndx = subset[k];
      if (mask[ndx] == 0) {
         sum += arr[ndx];
      }
   } CARE_REDUCE_LOOP_END
   return (ReturnType) (ReduceType) sum;
}

/************************************************************************
 * Function  : ArrayMaskedSum
 * Author(s) : Peter Robinson
 * Purpose   : Returns the sum of values in arr at indices where mask is 0.
 * ************************************************************************/
template<typename T, typename ReduceType, typename Exec, typename ReturnType>
CARE_INLINE ReturnType ArrayMaskedSum(care::host_device_ptr<const T> arr,
                             care::host_device_ptr<int const> mask,
                             int n, T initVal)
{
   ReduceType iVal = initVal;
   RAJAReduceSum<ReduceType> sum { iVal };

   CARE_STREAM_LOOP(i, 0, n) {
      sum += arr[i] * T(mask[i] == 0);
   } CARE_STREAM_LOOP_END

   return (ReturnType) (ReduceType) sum ;
}

/************************************************************************
 * Function  : FindIndexGT
 * Author(s) : Peter Robinson
 * Purpose   : Returns an index of an array where the value is greater than
 *             limit
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE int FindIndexGT(care::host_device_ptr<const T> arr, int n, T limit)
{
   RAJAReduceMin<int> minIndexAboveLimit {n};
   CARE_REDUCE_LOOP(i, 0, n) {
     if ( arr[i] > limit) {
          minIndexAboveLimit.min(i);
      }
   } CARE_REDUCE_LOOP_END

   int result = (int)minIndexAboveLimit;
   if (result >= n) {
      return -1; // care typically returns -1 for invalid value
   } else {
      return result;
   }

   /* above is supposed to be equivalent to below sequential code.
   int i ;
   for (i=0 ; i<n ; ++i) {
      if (arr[i] > limit) {
         return i ;
      }
   }
   return -1 ;
   */
}

/************************************************************************
 * Function  : FindIndexMax
 * Author(s) : Peter Robinson
 * Purpose   : Returns an index of an array where the value is greater than
 *             limit
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE int FindIndexMax(care::host_device_ptr<const T> arr, int n)
{
   RAJAReduceMaxLoc<T> maxLoc { std::numeric_limits<T>::lowest(), -1 };
   CARE_REDUCE_LOOP(i, 0, n) {
      maxLoc.maxloc(arr[i], i);
   } CARE_REDUCE_LOOP_END
   return maxLoc.getLoc();
}


/************************************************************************
 * Function  : ArrayCopy
 * Author(s) : Peter Robinson
 * Purpose   : Copies from one ManagedArray into another. from and to
 *             should not have the same or overlapping memory addresses.
 * ************************************************************************/
template <typename T>
CARE_INLINE void ArrayCopy(care::host_device_ptr<T> into,
                           care::host_device_ptr<const T> from,
                           int n, int start1, int start2)
{
   ArrayCopy(RAJAExec {}, into, from, n, start1, start2);
}

/************************************************************************
 * Function  : ArrayCopy
 * Author(s) : Peter Robinson
 * Purpose   : Copies from one ManagedArray into another. from and to
 *             should not have the same or overlapping memory addresses.
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE void ArrayCopy(Exec,
                           care::host_device_ptr<T> into,
                           care::host_device_ptr<const T> from,
                           int n, int start1, int start2)
{
   CARE_STREAM_LOOP(i, 0, n) {
      into[i+start1] = from[i+start2];
   } CARE_STREAM_LOOP_END
}

/************************************************************************
 * Function  : ArrayCopy
 * Author(s) : Peter Robinson
 * Purpose   : Copies from one ManagedArray into another. from and to
 *             should not have the same or overlapping memory addresses.
 * ************************************************************************/
template <typename T>
CARE_INLINE void ArrayCopy(RAJA::seq_exec,
                           care::host_device_ptr<T> into,
                           care::host_device_ptr<const T> from,
                           int n, int start1, int start2)
{
   CARE_SEQUENTIAL_LOOP(i, 0, n) {
      into[i+start1] = from[i+start2];
   } CARE_SEQUENTIAL_LOOP_END
}

/************************************************************************
 * Function  : ArrayDup
 * Author(s) : Peter Robinson
 * Purpose   : Duplicates a ManagedArray.
 * ************************************************************************/
template <typename T>
CARE_INLINE care::host_device_ptr<T> ArrayDup(care::host_device_ptr<const T> from, int len)
{
   return ArrayDup<T>(RAJAExec {}, from, len);
}

/************************************************************************
 * Function  : ArrayDup
 * Author(s) : Benjamin Liu
 * Purpose   : Duplicates a ManagedArray from raw pointer.
 * ************************************************************************/
template <typename T>
CARE_INLINE care::host_device_ptr<T> ArrayDup(const T* from, int len)
{
   return ArrayDup<T>(RAJAExec {}, from, len);
}

/************************************************************************
 * Function  : ArrayDup
 * Author(s) : Peter Robinson
 * Purpose   : Duplicates a ManagedArray.
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE care::host_device_ptr<T> ArrayDup(Exec, care::host_device_ptr<const T> from, int len)
{
   if (from == nullptr) { // don't make a new array for null input
     return nullptr;
   } else {
     care::host_device_ptr<T> newArray(len,"ArrayDup newArray");
     ArrayCopy<T>(Exec{}, newArray, from, len);
     return newArray;
   }
}

/************************************************************************
 * Function  : ArrayDup
 * Author(s) : Benjamin Liu
 * Purpose   : Duplicates a ManagedArray from raw pointer.
 * ************************************************************************/
template <typename T, typename Exec>
CARE_INLINE care::host_device_ptr<T> ArrayDup(Exec, const T* from, int len)
{
   if (from == nullptr) { // don't make a new array for null input
     return nullptr;
   } else {
     care::host_device_ptr<const T> wrappedArray(from, len, "ArrayDup wrappedArray");
     care::host_device_ptr<T> newArray(len,"ArrayDup newArray");
     ArrayCopy<T>(Exec{}, newArray, wrappedArray, len);
     wrappedArray.freeDeviceMemory(nullptr, 0);
     return newArray;
   }
}

/************************************************************************
 * Function  : ArrayDup
 * Author(s) : Benjamin Liu
 * Purpose   : Duplicates a ManagedArray from raw pointer.
 * ************************************************************************/
template <typename T>
CARE_INLINE care::host_device_ptr<T> ArrayDup(RAJA::seq_exec, const T* from, int len)
{
   if (from == nullptr) { // don't make a new array for null input
     return nullptr;
   } else {
     care::host_device_ptr<T> newArray(len,"ArrayDup newArray");
     CARE_SEQUENTIAL_LOOP(i, 0, len) {
        newArray[i] = from[i];
     } CARE_SEQUENTIAL_LOOP_END
     return newArray;
   }
}

//******************************************************************************
// If subset is defined, calls SumIntArraySubset, otherwise calls
// SumIntArray.
// @author Peter Robinson
//
template<typename T, typename ReduceType, typename Exec, typename ReturnType>
CARE_INLINE ReturnType SumArrayOrArraySubset(care::host_device_ptr<const T> arr,
                                    care::host_device_ptr<int const> subset, int n)
{
   if (subset) {
      return ArraySumSubset<T, ReduceType, Exec, ReturnType>(arr, subset, n, T(0));
   }
   else {
      return ArraySum<T, ReduceType, Exec, ReturnType>(arr, n, T(0));
   }
}

//******************************************************************************
// Based on whether mask and/or subset is defined, calls the appropriate
// sum array function.
// @author Peter Robinson
// @param arr Array of length > n
// @param mask Array of same length as arr
// @param subset Array of length n.
//
template<typename T, typename ReduceType, typename Exec, typename ReturnType>
CARE_INLINE ReturnType PickAndPerformSum(care::host_device_ptr<const T> arr,
                                care::host_device_ptr<int const> mask,
                                care::host_device_ptr<int const> subset, int n)
{
   if (mask) {
      if (subset) {
         return ArrayMaskedSumSubset<T, ReduceType, Exec, ReturnType>(arr, mask, subset, n, T(0));
      }
      else {
         return ArrayMaskedSum<T, ReduceType, Exec, ReturnType>(arr, mask, n, T(0));
      }
   }
   else {
      return SumArrayOrArraySubset<T, ReduceType, Exec, ReturnType>(arr, subset, n);
   }
}

//******************************************************************************
// Return the index of the minimum value of an array.
// @author Peter Robinson
// @param arr Data array of length >= n
// @param thresholds : If thresholds is not nullptr, only look at indices where
//                     thresholds is above cutoff.
// @cutoff           : The cutoff value described above.
// @thresholdIndex   : (out) the index of the threshold array used for the min vale.
template<typename T, typename Exec>
CARE_INLINE int FindIndexMinAboveThresholds(care::host_device_ptr<const T> arr, int n,
                                            care::host_device_ptr<double const> thresholds,
                                            double cutoff,
                                            int * thresholdIndex)
{
   int ndx = -1;
   if (thresholds) {
      RAJAReduceMinLoc<T> min { std::numeric_limits<T>::max(), ndx };
      CARE_REDUCE_LOOP(i, 0, n) {
         if (thresholds[i] > cutoff) {
            min.minloc(arr[i], i);
         }
      } CARE_REDUCE_LOOP_END
      ndx = min.getLoc();
      *thresholdIndex = ndx;
   }
   else {
      ArrayMinLoc<T, Exec>(arr, n, std::numeric_limits<T>::max(), ndx);
   }

   return ndx ;
}

//******************************************************************************
// Return the subset index minimum value of an array at indices defined by subset.
// @author Peter Robinson
// @param arr Data array of length >= max value of subset[0:lenset].
// @param subset: Indices of arr to find minvalue.
// @param lenset: Length of subset.
// @returns     : The index of subset where min value was found in arr.
//
template<typename T, typename Exec>
CARE_INLINE int FindIndexMinSubset(care::host_device_ptr<const T> arr,
                                   care::host_device_ptr<int const> subset, int lenset)
{
   RAJAReduceMinLoc<T> min { std::numeric_limits<T>::max(), -1 };
   CARE_REDUCE_LOOP(i, 0, lenset) {
      int curr = subset[i] ;
      min.minloc(arr[curr], curr);
   } CARE_REDUCE_LOOP_END
   return min.getLoc();
}

//******************************************************************************
// Return the index of the minimum value of a subset of an array, optionally
// where a corresponding thresholds array is above a cutoff value.
// @author Peter Robinson
// @param arr        : Data array of length >= max(subset[0:lenset])
// @param subset     : The subset of arr to find the min value for.
// @param lenset     : Length of subset.
// @param thresholds : If thresholds is not nullptr, only look at indices where
//                     thresholds is above cutoff. length >= max(subset[0:lenset]).
//                     Indexing of thresholds corresponds to the the subset,
//                     not of the original array arr.
// @cutoff           : The cutoff value described above.
// @thresholdIndex   : (out) the index of the threshold array used for the min vale.
//
template<typename T, typename Exec>
CARE_INLINE int FindIndexMinSubsetAboveThresholds(care::host_device_ptr<const T> arr,
                                                  care::host_device_ptr<int const> subset,
                                                  int lenset,
                                                  care::host_device_ptr<double const> thresholds,
                                                  double cutoff,
                                                  int * thresholdIndex)
{
   int ndx = -1 ;

   if (thresholds) {
      RAJAReduceMinLoc<T> min { std::numeric_limits<T>::max(), -1 };
      RAJAReduceMinLoc<T> thresholdmin { std::numeric_limits<T>::max(), -1 };

      CARE_REDUCE_LOOP(i, 0, lenset) {
         if (thresholds[i] > cutoff) { // if threshold were sized as arr, this would be thresholds[curr]
            const int curr = subset[i] ;
            min.minloc(arr[curr], curr);
            thresholdmin.minloc(arr[curr], i);
         }
      } CARE_REDUCE_LOOP_END

      *thresholdIndex = thresholdmin.getLoc();
      ndx = min.getLoc();
   }
   else {
      ndx = FindIndexMinSubset<T, Exec>(arr, subset, lenset);
   }

   return ndx ;
}

//******************************************************************************
// Return the index of the minimum value of the subset of the array.
// If that index happens to be masked off by a given mask, then
// returns -1. Designed for global reductions, where the mask is
// 1 for indices owned by other processors.
// @author Peter Robinson
// @param arr Data array of length >= n
// @param mask nullptr or Same length as arr, 1 for indices to ignore, 0 otherwise
// @param subset nullptr or length n, the indices of arr to include in the min search
//
// @note: threshold had length and indexing corresponding to the subset while mask
//        has length and index corresponding to arr
// @note: even if an element is masked off, thresholdIndex may still be set (if a threshold
//        is provided), just the returned value will be -1.
template<typename T, typename Exec>
CARE_INLINE int PickAndPerformFindMinIndex(care::host_device_ptr<const T> arr,
                                           care::host_device_ptr<int const> mask,
                                           care::host_device_ptr<int const> subset, int n,
                                           care::host_device_ptr<double const> thresholds,
                                           double cutoff,
                                           int *thresholdIndex)
{
   int minIndex;
   if (subset) {
      minIndex = FindIndexMinSubsetAboveThresholds<T, Exec>(arr, subset, n,
                                                            thresholds, cutoff,
                                                            thresholdIndex);
   }
   else {
      minIndex = FindIndexMinAboveThresholds<T, Exec>(arr, n, thresholds, cutoff,
                                                      thresholdIndex);
   }

   if (mask && n > 0) {
      if (minIndex >= 0 && mask.pick(minIndex) == 1) {
         minIndex = -1;
      }
   }

   return minIndex;
}

//******************************************************************************
// Return the index of the maximum value of an array.
// @author Peter Robinson
// @param arr Data array of length >= n
// @param thresholds : If thresholds is not nullptr, only look at indices where
//                     thresholds is above cutoff.
// @cutoff           : The cutoff value described above.
// @thresholdIndex   : (out) the index of the threshold array used for the max vale.
template<typename T, typename Exec>
CARE_INLINE int FindIndexMaxAboveThresholds(care::host_device_ptr<const T> arr, int n,
                                            care::host_device_ptr<double const> thresholds,
                                            double cutoff,
                                            int * thresholdIndex)
{
   int ndx = -1;

   if (thresholds) {
      RAJAReduceMaxLoc<T> max { std::numeric_limits<T>::lowest(), ndx };

      CARE_REDUCE_LOOP(i, 0, n) {
         if (thresholds[i] > cutoff) {
            max.maxloc(arr[i], i);
         }
      } CARE_REDUCE_LOOP_END

      ndx = max.getLoc();
      *thresholdIndex = ndx;
   }
   else {
      ArrayMaxLoc<T, Exec>(arr, n, std::numeric_limits<T>::lowest(), ndx);
   }

   return ndx ;
}

//******************************************************************************
// Return the subset index maximum value of an array at indices defined by subset.
// @author Peter Robinson
// @param arr Data array of length >= max value of subset[0:lenset].
// @param subset: Indices of arr to find maxvalue.
// @param lenset: Length of subset.
// @returns     : The index of subset where max value was found in arr.
//
template<typename T, typename Exec>
CARE_INLINE int FindIndexMaxSubset(care::host_device_ptr<const T> arr,
                                   care::host_device_ptr<int const> subset, int lenset)
{
   RAJAReduceMaxLoc<T> max { std::numeric_limits<T>::lowest(), -1 };
   CARE_REDUCE_LOOP(i, 0, lenset) {
      int curr = subset[i] ;
      max.maxloc(arr[curr], curr);
   } CARE_REDUCE_LOOP_END
   return max.getLoc();
}

//******************************************************************************
// Return the index of the maximum value of a subset of an array, optionally
// where a corresponding thresholds array is above a cutoff value.
// @author Peter Robinson
// @param arr        : Data array of length >= max(subset[0:lenset])
// @param subset     : The subset of arr to find the max value for.
// @param lenset     : Length of subset.
// @param thresholds : If thresholds is not nullptr, only look at indices where
//                     thresholds is above cutoff. length >= max(subset[0:lenset]).
//                     Indexing corresponds to subset, the the array arr.
// @cutoff           : The cutoff value described above.
// @thresholdIndex   : (out) the index of the threshold array used for the max vale.
//
template<typename T, typename Exec>
CARE_INLINE int FindIndexMaxSubsetAboveThresholds(care::host_device_ptr<const T> arr,
                                                  care::host_device_ptr<int const> subset,
                                                  int lenset,
                                                  care::host_device_ptr<double const> thresholds,
                                                  double cutoff,
                                                  int * thresholdIndex)
{
   int ndx = -1;

   if (thresholds) {
      RAJAReduceMaxLoc<T> max { std::numeric_limits<T>::lowest(), -1 };
      RAJAReduceMaxLoc<T> thresholdmax { std::numeric_limits<T>::lowest(), -1 };

      CARE_REDUCE_LOOP(i, 0, lenset) {
         if (thresholds[i] > cutoff) {
            const int curr = subset[i] ;
            max.maxloc(arr[curr], curr);
            thresholdmax.maxloc(arr[curr], i);
         }
      } CARE_REDUCE_LOOP_END

      *thresholdIndex = thresholdmax.getLoc();
      ndx = max.getLoc();
   }
   else {
      ndx = FindIndexMaxSubset<T, Exec>(arr, subset, lenset);
   }

   return ndx ;
}

//******************************************************************************
// Return the index of the maximum value of the subset of the array.
// If that index happens to be masked off by a given mask, then
// returns -1. Designed for global reductions, where the mask is
// 1 for indices owned by other processors.
// @author Peter Robinson
// @param arr Data array of length >= n
// @param mask nullptr or Same length as arr, 1 for indices to ignore, 0 otherwise
// @param subset nullptr or length n, the indices of arr to include in the max search
//
// @note: threshold had length and indexing corresponding to the subset while mask
//        has length and index corresponding to arr
// @note: even if an element is masked off, thresholdIndex may still be set (if a threshold
//        is provided), just the returned value will be -1.
template<typename T, typename Exec>
CARE_INLINE int PickAndPerformFindMaxIndex(care::host_device_ptr<const T> arr,
                                           care::host_device_ptr<int const> mask,
                                           care::host_device_ptr<int const> subset, int n,
                                           care::host_device_ptr<double const> thresholds,
                                           double cutoff,
                                           int *thresholdIndex)
{
   int maxIndex;
   if (subset) {
      maxIndex = FindIndexMaxSubsetAboveThresholds<T, Exec>(arr, subset, n,
                                                            thresholds, cutoff,
                                                            thresholdIndex);
   }
   else {
      maxIndex = FindIndexMaxAboveThresholds<T, Exec>(arr, n, thresholds, cutoff,
                                                      thresholdIndex);
   }

   if (mask && n > 0) {
      if (maxIndex >= 0 && mask.pick(maxIndex) == 1) {
         maxIndex = -1;
      }
   }

   return maxIndex;
}

} // namespace care

#endif // CARE_ALGORITHM_IMPL_H
