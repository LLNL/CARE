//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
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

template<typename T, typename ReduceType=T, typename Exec=RAJAExec, typename ReturnType=T>
ReturnType PickAndPerformSum(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, care::host_device_ptr<int const> subset, int n);

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

template <typename T, typename ReducerType=T, typename Exec=RAJAExec, typename ReturnType=T>
ReturnType ArraySum(care::host_device_ptr<const T> arr, int n, T initVal);

template <typename T, typename ReduceType=T, typename Exec=RAJAExec, typename ReturnType=T>
ReturnType ArraySumSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int n, T initVal);

template<typename T, typename ReduceType=T, typename Exec=RAJAExec, typename ReturnType=T>
ReturnType SumArrayOrArraySubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const>  subset, int n);

template <typename T, typename ReduceType=T, typename Exec=RAJAExec, typename ReturnType=T>
ReturnType ArrayMaskedSumSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, care::host_device_ptr<int const> subset, int n, T initVal);

template <typename T, typename ReduceType=T, typename Exec=RAJAExec, typename ReturnType=T>
ReturnType ArrayMaskedSum(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, int n, T initVal);

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

enum class compress_array { removed_list, mapping_list, remove_flag_list, keep_flag_list };

template <typename T>
int CompressArray(RAJA::seq_exec, care::host_device_ptr<T> & arr, const int arrLen,
                    care::host_device_ptr<int const> list, const int listLen, const care::compress_array listType, bool realloc=false);
#ifdef CARE_PARALLEL_DEVICE
template <typename T>
int CompressArray(RAJADeviceExec exec, care::host_device_ptr<T> & arr, const int arrLen,
                    care::host_device_ptr<int const> list, const int listLen, const care::compress_array listType, bool realloc=false);
#endif // defined(CARE_PARALLEL_DEVICE)
template <typename T>
int CompressArray(care::host_device_ptr<T> & arr, const int arrLen,
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


///////////////////////////////////////////////////////////////////////////
/// @author Ben Liu, Peter Robinson, Alan Dayton
/// @brief Checks whether an array of type T is sorted and optionally unique.
/// @param[in] array           - The array to check
/// @param[in] len             - The number of elements contained in the sorter
/// @param[in] name            - The name of the calling function
/// @param[in] argname         - The name of the sorter in the calling function
/// @param[in] allowDuplicates - Whether or not to allow duplicates
/// @param[in] warnOnFailure   - Whether to print a warning if array not sorted
/// @return true if sorted, false otherwise
///////////////////////////////////////////////////////////////////////////
template <typename T>
CARE_HOST_DEVICE CARE_INLINE bool checkSorted(const T* array, const int len,
                                              const char* name, const char* argname,
                                              const bool allowDuplicates,
                                              const bool warnOnFailure)
{
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
         if (warnOnFailure) {
            printf("care:%s: %s not in ascending order at index %d\n", name, argname, last + 1);
         }
         return false;
      }
   }

   return true;
}

template <typename T>
CARE_HOST_DEVICE CARE_INLINE bool checkSorted(const care::host_device_ptr<const T>& array,
                                              const int len,
                                              const char* name,
                                              const char* argname,
                                              const bool allowDuplicates,
                                              const bool warnOnFailure)
{
   return checkSorted<T>(array.data(), len, name, argname, allowDuplicates, warnOnFailure);
}

/************************************************************************
 * Function  : BinarySearch
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
 *             than num. A return value of -1 indicates that all values
 *             in map are smaller than or equal to num.
 *
 *             @NOTE: Intentionally implemented this using only the '<'
 *             operator to follow weak strict ordering semantics.
 *
 ************************************************************************/

template <typename T>
CARE_HOST_DEVICE CARE_INLINE int BinarySearch(const T *map, const int start,
                                              const int mapSize, const T num,
                                              bool returnUpperBound)
{
   int klo = start ;
   int khi = start + mapSize;
   int k = ((khi+klo) >> 1) + 1 ;

   if ((map == nullptr) || (mapSize == 0)) {
      return -1 ;
   }
#ifdef CARE_DEBUG
   const bool allowDuplicates = true;
   const bool warnOnFailure = true;
   checkSorted(&(map[start]), mapSize, "BinarySearch", "map", allowDuplicates, warnOnFailure) ;
#endif

   while (khi-klo > 1) {
      k = (khi+klo) >> 1 ;
      if (! (map[k] < num) && !(num < map[k])) {
         if (returnUpperBound) {
            khi = k+1;
            klo = k;
            continue;
         }
         else {
            return k ;
         }
      }
      else if (num < map[k]) {
         khi = k ;
      }
      else {
         klo = k ;
      }
   }
   if (returnUpperBound) {
      k = klo;
      // the lower option bounds num
      if (num < map[k]) {
         return k;
      }
      // the upper option is within the range of the map index set
      if (khi < start + mapSize) {
         // Note: fix for last test in TEST(algorithm, binarysearch). This algorithm has failed to pick up the upper
         // bound above 1 in the array {0, 1, 1, 1, 1, 1, 6}. Having 1 repeated confused the algorithm.
         while ((khi < start + mapSize) && (!(map[khi] <  num) && !(num < map[khi]))) {
            ++khi;
         }

         // the upper option bounds num
         if ((khi < start + mapSize) && (num < map[khi])) {
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
   --k;
   if (!(map[k] < num) && !(num < map[k])) {
      return k ;
   }
   else {
      return -1 ;
   }
}

template<typename mapType>
CARE_HOST_DEVICE CARE_INLINE int BinarySearch(const care::host_device_ptr<mapType>& map, const int start,
                                              const int mapSize, const mapType num,
                                              bool returnUpperBound)
{
   return BinarySearch<mapType>(map.data(), start, mapSize, num, returnUpperBound);
}

template<typename mapType>
CARE_HOST_DEVICE CARE_INLINE int BinarySearch(const care::host_device_ptr<const mapType>& map, const int start,
                                  const int mapSize, const mapType num,
                                  bool returnUpperBound)
{
   return BinarySearch<mapType>(map.data(), start, mapSize, num, returnUpperBound);
}


/************************************************************************
 * Function  : uniqLocal
 * Author(s) : Benjamin Liu
 * Purpose   : Remove duplicates in-place from an array that is sorted
 *             in ascending order and updates len.
 *             For calls from within RAJA loops.
 *             Does not reallocate array.
 ************************************************************************/
template <typename T>
CARE_HOST_DEVICE CARE_INLINE void uniqLocal(care::local_ptr<T> array, int& len)
{
   int origLen = len ;
   len = 0 ;

   int i = 0 ;
   while (i < origLen) {
      /* copy the unique value into the array */
      array[len] = array[i] ;
      /* skip over all the redundant elements */
      while (i < origLen && array[i] == array[len]) {
         ++i ;
      }
      ++len ;
   }
}

template <typename T>
CARE_HOST_DEVICE CARE_INLINE T ArrayMin(care::local_ptr<const T> arr, int n, T initVal, int startIndex)
{
   T min = initVal;
   for (int k = startIndex; k < n; ++k) {
      min = care::min(min, arr[k]);
   }
   return min;
}

template <typename T>
CARE_HOST_DEVICE CARE_INLINE T ArrayMin(care::local_ptr<T> arr, int n, T initVal, int startIndex)
{
   return ArrayMin<T>((care::local_ptr<const T>)arr, n, initVal, startIndex);
}

template <typename T>
CARE_HOST_DEVICE CARE_INLINE T ArrayMax(care::local_ptr<const T> arr, int n, T initVal, int startIndex)
{
   T max = initVal;
   for (int k = startIndex; k < n; ++k) {
      max = care::max(max, arr[k]);
   }
   return max;
}

template <typename T>
CARE_HOST_DEVICE CARE_INLINE T ArrayMax(care::local_ptr<T> arr, int n, T initVal, int startIndex)
{
   return ArrayMax<T>((care::local_ptr<const T>)arr, n, initVal, startIndex);
}

/************************************************************************
 * Function  : ArrayMinMax
 * Author(s) : Peter Robinson
 * Purpose   : Stores Minimum / Maximum values of arr (as a double) in outMin / outMax;
 *             If mask was such that no values were compared, returns 0, outMin will be -DBL_MAX, outMax will be DBL_MAX
 *             Otherwise, returns 1.
 *             care::local_ptr API to support calls from within RAJA contexts.
 * ************************************************************************/
template <typename T>
CARE_HOST_DEVICE CARE_INLINE int ArrayMinMax(care::local_ptr<const T> arr,
                                             care::local_ptr<int const> mask,
                                             int n, double *outMin, double *outMax)
{
   bool result = false;
   // a previous implementation had min and max as a templated type and then used std::numeric_limits<T>::lowest() and 
   // std::numeric_limits<ReducerType>::max() for initial values, but that is not valid on the device and results in 
   // warnings and undefined behavior at runtime.
   double min, max;
   if (arr) {
      max =  -DBL_MAX; 
      min =  DBL_MAX;
      if (mask) {
         for (int i = 0; i < n; ++i) {
            if (mask[i]) {
               min = care::min(min, (double)arr[i]);
               max = care::max(max, (double)arr[i]);
            }
         }
         if (min != DBL_MAX ||
             max != -DBL_MAX) {
            result = true;
         }
      }
      else {
         for (int i = 0; i < n; ++i) {
            min = care::min(min, (double)arr[i]);
            max = care::max(max, (double)arr[i]);
         }
         result = true;
      }
   }

   if (result) {
      *outMin = (double) min;
      *outMax = (double) max;
   }
   else {
      *outMin = -DBL_MAX;
      *outMax = +DBL_MAX;
   }
   return (int) result;
}

template <typename T>
CARE_HOST_DEVICE CARE_INLINE int ArrayMinMax(care::local_ptr<T> arr,
                                             care::local_ptr<int> mask,
                                             int n, double *outMin, double *outMax)
{
   return ArrayMinMax<T>((care::local_ptr<const T>)arr, (care::local_ptr<int const>)mask, n, outMin, outMax);
}



} // end namespace care

#endif // !defined(CARE_ALGORITHM_DECL_H)
