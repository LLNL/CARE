//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_ALGORITHM_DECL_H
#define CARE_ALGORITHM_DECL_H

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/host_device_ptr.h"
#include "care/host_ptr.h"
#include "care/local_ptr.h"
#include "care/policies.h"
#include "care/CHAIDataGetter.h"

#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

namespace care {

template <typename T>
CARE_HOST_DEVICE CARE_INLINE T abs(const T a)
{
   return a > 0 ? a : -a ;
}

template <typename T>
CARE_HOST_DEVICE CARE_INLINE const T& max(const T& a, const T& b)
{
   return a > b ? a : b;
}

template <typename T>
CARE_HOST_DEVICE CARE_INLINE const T& min(const T& a, const T& b)
{
   return a < b ? a : b;
}

template <class T, class Size, class U>
void fill_n(care::host_device_ptr<T> arr, Size n, const U& val);

template <class T, class Size, class U>
void copy_n(care::host_device_ptr<const T> in, Size n, care::host_device_ptr<U> out);

template <class T, class Size, class U>
void copy_n(care::host_device_ptr<T> in, Size n, care::host_device_ptr<U> out);

template <typename T, typename Exec=RAJAExec>
T ArrayMin(care::host_device_ptr<const T> arr, int endIndex, T initVal, int startIndex = 0);

template <typename T, typename Exec=RAJAExec>
T ArrayMin(care::host_device_ptr<T> arr, int endIndex, T initVal, int startIndex = 0);

template <typename T>
CARE_HOST_DEVICE T ArrayMin(care::local_ptr<const T> arr, int endIndex, T initVal, int startIndex = 0);

template <typename T>
CARE_HOST_DEVICE T ArrayMin(care::local_ptr<T> arr, int endIndex, T initVal, int startIndex = 0);

template <typename T>
T ArrayMin(care::host_ptr<const T> arr, int n, T initVal, int startIndex = 0);

template <typename T>
T ArrayMin(care::host_ptr<T> arr, int n, T initVal, int startIndex = 0);

template <typename T, typename Exec=RAJAExec>
T ArrayMax(care::host_device_ptr<const T> arr, int n, T initVal, int startIndex = 0);

template <typename T, typename Exec=RAJAExec>
T ArrayMax(care::host_device_ptr<T> arr, int n, T initVal, int startIndex = 0);

template <typename T>
CARE_HOST_DEVICE T ArrayMax(care::local_ptr<const T> arr, int n, T initVal, int startIndex = 0);

template <typename T>
CARE_HOST_DEVICE T ArrayMax(care::local_ptr<T> arr, int n, T initVal, int startIndex = 0);

template <typename T>
T ArrayMax(care::host_ptr<const T> arr, int n, T initVal, int startIndex = 0);

template <typename T>
T ArrayMax(care::host_ptr<T> arr, int n, T initVal, int startIndex = 0);

template <typename T, typename ReducerType=T, typename Exec=RAJAExec>
int ArrayMinMax(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, int n, double *outMin, double *outMax);

template <typename T, typename ReducerType=T, typename Exec=RAJAExec>
int ArrayMinMax(care::host_device_ptr<T> arr, care::host_device_ptr<int> mask, int n, double *outMin, double *outMax);

#if CARE_HAVE_LLNL_GLOBALID

template <typename Exec=RAJAExec>
int ArrayMinMax(care::host_device_ptr<const globalID> arr, care::host_device_ptr<int const> mask, int n, double *outMin, double *outMax);

#endif // CARE_HAVE_LLNL_GLOBALID

template <typename T>
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<const T> arr, care::local_ptr<int const> mask, int n, double *outMin, double *outMax);

template <typename T>
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<T> arr, care::local_ptr<int> mask, int n, double *outMin, double *outMax);

template <typename T, typename Exec=RAJAExec>
T ArrayMinLoc(care::host_device_ptr<const T> arr, int n, T initVal, int & loc);

template <typename T, typename Exec=RAJAExec>
T ArrayMaxLoc(care::host_device_ptr<const T> arr, int n, T initVal, int & loc);

template <typename T>
int ArrayFind(care::host_device_ptr<const T> arr, const int len, const T val, const int start = 0) ;

template<typename T, typename ReduceType=T, typename Exec=RAJAExec>
T PickAndPerformSum(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, care::host_device_ptr<int const> subset, int n);

template<typename T, typename Exec=RAJAExec>
int FindIndexMinAboveThresholds(care::host_device_ptr<const T> arr, int n,
                                care::host_device_ptr<double const> thresholds,
                                double cutoff,
                                int * thresholdIndex);

template<typename T, typename Exec=RAJAExec>
int FindIndexMinSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int lenset);

template<typename T, typename Exec=RAJAExec>
int FindIndexMinSubsetAboveThresholds(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int lenset,
                                      care::host_device_ptr<double const> thresholds, double cutoff,
                                      int * thresholdIndex);

template<typename T, typename Exec=RAJAExec>
int PickAndPerformFindMinIndex(care::host_device_ptr<const T> arr,
                               care::host_device_ptr<int const> mask,
                               care::host_device_ptr<int const> subset, int n,
                               care::host_device_ptr<double const> thresholds,
                               double cutoff,
                               int *thresholdIndex);

template<typename T, typename Exec=RAJAExec>
int FindIndexMaxAboveThresholds(care::host_device_ptr<const T> arr, int n,
                                care::host_device_ptr<double const> thresholds,
                                double cutoff,
                                int * thresholdIndex);

template<typename T, typename Exec=RAJAExec>
int FindIndexMaxSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int lenset);

template<typename T, typename Exec=RAJAExec>
int FindIndexMaxSubsetAboveThresholds(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int lenset,
                                      care::host_device_ptr<double const> thresholds, double cutoff,
                                      int * thresholdIndex);

template<typename T, typename Exec=RAJAExec>
int PickAndPerformFindMaxIndex(care::host_device_ptr<const T> arr,
                               care::host_device_ptr<int const> mask,
                               care::host_device_ptr<int const> subset, int n,
                               care::host_device_ptr<double const> thresholds,
                               double cutoff,
                               int *thresholdIndex);

/* returns count of occurence of val in array */
template <typename T, typename Exec=RAJAExec>
int ArrayCount(care::host_device_ptr<const T> arr, int length, T val);

template <typename T, typename ReducerType=T, typename Exec=RAJAExec>
T ArraySum(care::host_device_ptr<const T> arr, int n, T initVal);

template <typename T, typename ReduceType=T, typename Exec=RAJAExec>
T ArraySumSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int n, T initVal);

template<typename T, typename Exec=RAJAExec>
T SumArrayOrArraySubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const>  subset, int n);

template <typename T, typename ReduceType=T, typename Exec=RAJAExec>
T ArrayMaskedSumSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, care::host_device_ptr<int const> subset, int n, T initVal);

template <typename T, typename ReduceType=T, typename Exec=RAJAExec>
T ArrayMaskedSum(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, int n, T initVal);

template <typename T, typename Exec=RAJAExec>
int FindIndexGT(care::host_device_ptr<const T> arr, int n, T limit);

template <typename T>
care::host_device_ptr<T> ArrayDup(care::host_device_ptr<const T> from, int len);

template <typename T>
care::host_device_ptr<T> ArrayDup(const T* from, int len);

template <typename T, typename Exec>
care::host_device_ptr<T> ArrayDup(Exec, care::host_device_ptr<const T> from, int len);

template <typename T, typename Exec>
care::host_device_ptr<T> ArrayDup(Exec, const T* from, int len);

template <typename T>
care::host_device_ptr<T> ArrayDup(RAJA::seq_exec, const T* from, int len);

template <typename T>
void ArrayCopy(care::host_device_ptr<T> into, care::host_device_ptr<const T> from, int n,
               int start1=0, int start2=0);

template <typename T, typename Exec>
void ArrayCopy(Exec,
               care::host_device_ptr<T> into, care::host_device_ptr<const T> from,
               int n, int start1=0, int start2=0);

template <typename T>
void ArrayCopy(RAJA::seq_exec,
               care::host_device_ptr<T> into, care::host_device_ptr<const T> from,
               int n, int start1=0, int start2=0);

template<typename T>
CARE_HOST_DEVICE void ArrayCopy(
                           care::local_ptr<T> into,
                           care::local_ptr<const T> from,
                           int n, int start1=0, int start2=0);

/************************************************************************
 * Function  : ArrayCopy
 * Author(s) : Peter Robinson
 * Purpose   : Copies from one local_ptr into another. from and to
 *             should not have the same or overlapping memory addresses.
 * ************************************************************************/
template<typename T>
CARE_HOST_DEVICE inline void ArrayCopy(
                           care::local_ptr<T> into,
                           care::local_ptr<const T> from,
                           int n, int start1, int start2)
{
   for (int i = 0; i < n; ++i)  {
      into[i+start1] = from[i+start2];
   }
}

template <typename T, typename Exec=RAJAExec >
int FindIndexMax(care::host_device_ptr<const T> arr, int n);

template <typename T>
CARE_HOST_DEVICE bool checkSorted(const T* array, const int len,
                                  const char* name, const char* argname,
                                  const bool allowDuplicates = false,
                                  const bool warnOnFailure = true);

template <typename T>
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<const T>& array, const int len,
                                  const char* name, const char* argname,
                                  const bool allowDuplicates = false,
                                  const bool warnOnFailure = true);

template<typename mapType>
CARE_HOST_DEVICE CARE_DLL_API int BinarySearch(const mapType *map,
                                               const int start,
                                               const int mapSize,
                                               const mapType num,
                                               bool returnUpperBound = false);

template<typename mapType>
CARE_HOST_DEVICE CARE_DLL_API int BinarySearch(const care::host_device_ptr<const mapType> & map,
                                               const int start,
                                               const int mapSize,
                                               const mapType num,
                                               bool returnUpperBound = false);

template<typename mapType>
CARE_HOST_DEVICE CARE_DLL_API int BinarySearch(const care::host_device_ptr<mapType> & map,
                                               const int start,
                                               const int mapSize,
                                               const mapType num,
                                               bool returnUpperBound = false);

#ifdef CARE_PARALLEL_DEVICE
template <typename T>
void IntersectArrays(RAJADeviceExec exec,
                     care::host_device_ptr<const T> arr1, int size1, int start1,
                     care::host_device_ptr<const T> arr2, int size2, int start2,
                     care::host_device_ptr<int> &matches1, care::host_device_ptr<int> &matches2,
                     int *numMatches);

template <typename T>
void IntersectArrays(RAJADeviceExec exec,
                     care::host_device_ptr<T> arr1, int size1, int start1,
                     care::host_device_ptr<T> arr2, int size2, int start2,
                     care::host_device_ptr<int> &matches1, care::host_device_ptr<int> &matches2,
                     int *numMatches);
#endif // defined(CARE_PARALLEL_DEVICE)

template <typename T>
void IntersectArrays(RAJA::seq_exec,
                     care::host_ptr<const T> arr1, int size1, int start1,
                     care::host_ptr<const T> arr2, int size2, int start2,
                     care::host_ptr<int> &matches1, care::host_ptr<int> &matches2,
                     int *numMatches);

template <typename T>
void IntersectArrays(RAJA::seq_exec exec,
                     care::host_ptr<T> arr1, int size1, int start1,
                     care::host_ptr<T> arr2, int size2, int start2,
                     care::host_ptr<int> &matches1, care::host_ptr<int> &matches2,
                     int *numMatches);

template <typename T>
void IntersectArrays(RAJA::seq_exec exec,
                     care::host_device_ptr<const T> arr1, int size1, int start1,
                     care::host_device_ptr<const T> arr2, int size2, int start2,
                     care::host_device_ptr<int> &matches1, care::host_device_ptr<int> &matches2,
                     int *numMatches);

template <typename T>
void IntersectArrays(RAJA::seq_exec exec,
                     care::host_device_ptr<T> arr1, int size1, int start1,
                     care::host_device_ptr<T> arr2, int size2, int start2,
                     care::host_device_ptr<int> &matches1, care::host_device_ptr<int> &matches2,
                     int *numMatches);

template <typename T>
void sortArray(RAJA::seq_exec, care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy) ;

template <typename T>
void sortArray(RAJA::seq_exec, care::host_device_ptr<T> &Array, size_t len) ;

#if defined(CARE_PARALLEL_DEVICE)
#if defined(CARE_GPUCC)

template <typename T>
std::enable_if_t<std::is_arithmetic<typename CHAIDataGetter<T, RAJADeviceExec>::raw_type>::value, void>
sortArray(RAJADeviceExec, care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy);

#if defined(__HIPCC__) || (defined(__CUDACC__) && defined(CUB_MAJOR_VERSION) && defined(CUB_MINOR_VERSION) && (CUB_MAJOR_VERSION >= 2 || (CUB_MAJOR_VERSION == 1 && CUB_MINOR_VERSION >= 14)))
template <typename T>
std::enable_if_t<!std::is_arithmetic<typename CHAIDataGetter<T, RAJADeviceExec>::raw_type>::value, void>
sortArray(RAJADeviceExec, care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy);
#endif

template <typename T>
void radixSortArray(care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy);

#else // defined(CARE_GPUCC)

template <typename T>
void sortArray(RAJADeviceExec, care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy);

#endif // defined(CARE_GPUCC)

template <typename T>
void sortArray(RAJADeviceExec, care::host_device_ptr<T> &Array, size_t len);

#endif // defined(CARE_PARALLEL_DEVICE)

template <typename T>
void uniqArray(RAJA::seq_exec, care::host_device_ptr<const T> Array, size_t len, care::host_device_ptr<T> & outArray, int & newLen);
template <typename T>
int uniqArray(RAJA::seq_exec exec, care::host_device_ptr<T> & Array, size_t len, bool noCopy=false);
#ifdef CARE_PARALLEL_DEVICE
template <typename T>
void uniqArray(RAJADeviceExec, care::host_device_ptr<const T>  Array, size_t len, care::host_device_ptr<T> & outArray, int & outLen);
template <typename T>
int uniqArray(RAJADeviceExec exec, care::host_device_ptr<T> & Array, size_t len, bool noCopy=false);
#endif // defined(CARE_PARALLEL_DEVICE)

template <typename T, typename Exec>
void sort_uniq(Exec e, care::host_device_ptr<T> * array, int * len, bool noCopy = false);

enum class compress_array { removed_list, mapping_list };

template <typename T>
void CompressArray(RAJA::seq_exec, care::host_device_ptr<T> & arr, const int arrLen,
                   care::host_device_ptr<int const> list, const int listLen, const care::compress_array listType, bool realloc=false);
#ifdef CARE_PARALLEL_DEVICE
template <typename T>
void CompressArray(RAJADeviceExec exec, care::host_device_ptr<T> & arr, const int arrLen,
                   care::host_device_ptr<int const> list, const int listLen, const care::compress_array listType, bool realloc=false);
#endif // defined(CARE_PARALLEL_DEVICE)
template <typename T>
void CompressArray(care::host_device_ptr<T> & arr, const int arrLen,
                   care::host_device_ptr<int const> list, const int listLen, const care::compress_array listType, bool realloc=false);

template <typename T>
CARE_HOST_DEVICE void InsertionSort(care::local_ptr<T> array, int len);

/************************************************************************
 * Function  : InsertionSort
 * Author(s) : Rob Neely
 * Purpose   : Simple insertion sort function.  Should only be used on
 *             small arrays - otherwise use the qsort function from the
 *             standard C library.  Sorts in ascending order.
 ************************************************************************/
template <typename T>
CARE_HOST_DEVICE inline void InsertionSort(care::local_ptr<T> array, int len)
{
   if (len <= 1) {
      return;
   }

   for (int i=1 ; i<len ; ++i) {
      T tmp = array[i] ;
      int j ;
      for (j=i-1 ; (j >= 0) && (array[j] > tmp) ; --j) {
         array[j+1] = array[j];
      }
      array[j+1] = tmp ;
   }
}

/************************************************************************
 * Function  : LocalSortPairs
 * Author(s) : Alan Dayton
 * Purpose   : Simple insertion simultaneous sort function.  Should only
 *             be used on small arrays - otherwise use the qsort function
 *             from the standard C library.  Sorts in ascending order.
 ************************************************************************/
template <class T, class U, class Comparator>
CARE_HOST_DEVICE void LocalSortPairs(int length,
                                     care::local_ptr<T> sortArray,
                                     care::local_ptr<U> pairArray,
                                     Comparator comparator);

template <class T, class U, class Comparator>
CARE_HOST_DEVICE void LocalSortPairs(int length,
                                     care::local_ptr<T> sortArray,
                                     care::local_ptr<U> pairArray,
                                     Comparator comparator)
{
   for (int i = 1; i < length; ++i) {
      const T currentSortArrayEntry = sortArray[i];
      const U currentPairArrayEntry = pairArray[i];

      int j = i - 1;

      while (j >= 0 && comparator(currentSortArrayEntry, sortArray[j])) {
         sortArray[j + 1] = sortArray[j];
         pairArray[j + 1] = pairArray[j];

         --j;
      }

      sortArray[j + 1] = currentSortArrayEntry;
      pairArray[j + 1] = currentPairArrayEntry;
   }
}

template <class T, class U>
CARE_HOST_DEVICE void LocalSortPairs(int length,
                                     care::local_ptr<T> sortArray,
                                     care::local_ptr<U> pairArray);

template <class T, class U>
CARE_HOST_DEVICE void LocalSortPairs(int length,
                                     care::local_ptr<T> sortArray,
                                     care::local_ptr<U> pairArray) {
   LocalSortPairs(length, sortArray, pairArray,
                  [] (const T& val1, const T& val2) { return val1 < val2; });
}

template <typename T>
CARE_HOST_DEVICE void sortLocal(care::local_ptr<T> array, int len);

/************************************************************************
 * Function  : sortLocal
 * Author(s) : Benjamin Liu
 * Purpose   : General sort routine to call from within RAJA loops.
 *             Sorts in ascending order.
 ************************************************************************/
template <typename T>
CARE_HOST_DEVICE inline void sortLocal(care::local_ptr<T> array, int len)
{  
   if (len > 1) {
#if defined(__CUDA_ARCH__)
      // TODO this should be replaced with a CUDA GPU sort implementation that
      // is reasonable for longer arrays.
      InsertionSort(array, len) ;
#elif defined(__HIP_DEVICE_COMPILE__)
      // TODO this should be replaced with a HIPCC GPU sort implementation that
      // is reasonable for longer arrays.
      InsertionSort(array, len) ;
#else
      // host compile case
      std::sort(array.data(), array.data()+len) ;
#endif
   }
}

template <typename T>
CARE_HOST_DEVICE void uniqLocal(care::local_ptr<T> array, int& len);

template <typename T>
void ExpandArrayInPlace(RAJA::seq_exec, care::host_device_ptr<T> array, care::host_device_ptr<int const> indexSet, int length);
#ifdef CARE_PARALLEL_DEVICE
template <typename T>
void ExpandArrayInPlace(RAJADeviceExec, care::host_device_ptr<T> array, care::host_device_ptr<int const> indexSet, int length);
#endif // defined(CARE_PARALLEL_DEVICE)

} // end namespace care

#endif // !defined(CARE_ALGORITHM_DECL_H)

