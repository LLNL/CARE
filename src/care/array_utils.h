//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_ARRAY_UTILS_H_
#define _CARE_ARRAY_UTILS_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/care.h"

// Other library headers
#ifdef __CUDACC__
#include "cub/cub.cuh"
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
#endif

#ifdef __HIPCC__
#include "hipcub/hipcub.hpp"
#endif

#define CARE_MAX(a,b) a > b ? a : b
#define CARE_MIN(a,b) a < b ? a : b

namespace care_utils {
template <typename T, typename Exec=RAJAExec>
void ArrayFill(care::host_device_ptr<T> arr, int n, T val) ;

template <typename T>
void ArrayFill(care::host_ptr<T> arr, int n, T val) ;

template <typename T, typename Exec=RAJAExec>
T ArrayMin(care::host_device_ptr<const T> arr, int endIndex, T initVal, int startIndex = 0);

template <typename T, typename Exec=RAJAExec>
T ArrayMin(care::host_device_ptr<T> arr, int endIndex, T initVal, int startIndex = 0);

template <typename T>
CARE_HOST_DEVICE T ArrayMin(care::local_ptr<const T> arr, int endIndex, T initVal, int startIndex = 0);

template <typename T>
CARE_HOST_DEVICE T ArrayMin(care::local_ptr<T> arr, int endIndex, T initVal, int startIndex = 0);

template <typename T, typename Exec=RAJAExec>
T ArrayMax(care::host_device_ptr<const T> arr, int n, T initVal);

template <typename T, typename Exec=RAJAExec>
T ArrayMax(care::host_device_ptr<T> arr, int n, T initVal);

template <typename T>
CARE_HOST_DEVICE inline T ArrayMax(care::local_ptr<const T> arr, int n, T initVal);

template <typename T>
CARE_HOST_DEVICE inline T ArrayMax(care::local_ptr<T> arr, int n, T initVal);

template <typename T>
T ArrayMax(care::host_ptr<const T> arr, int n, T initVal);

template <typename T>
T ArrayMax(care::host_ptr<T> arr, int n, T initVal);

template <typename T, typename ReducerType=T, typename Exec=RAJAExec>
int ArrayMinMax(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, int n, double *outMin, double *outMax);

template <typename T, typename ReducerType=T, typename Exec=RAJAExec>
int ArrayMinMax(care::host_device_ptr<T> arr, care::host_device_ptr<int> mask, int n, double *outMin, double *outMax);

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

template <typename T, typename Exec=RAJAExec >
care::host_device_ptr<T> ArrayDup(care::host_device_ptr<const T> from, int len);

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


template <typename T, typename Exec=RAJAExec >
int FindIndexMax(care::host_device_ptr<const T> arr, int n);



template <typename T>
CARE_HOST_DEVICE bool checkSorted(const T* array, const int len,
                                  const char* name, const char* argname,
                                  const bool allowDuplicates = false);


template <typename T>
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<const T>& array, const int len,
                                  const char* name, const char* argname,
                                  const bool allowDuplicates = false);

#if defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

template <typename T>
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<T>& array, const int len,
                                  const char* name, const char* argname,
                                  const bool allowDuplicates = false);

#endif // defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

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
CARE_HOST_DEVICE bool checkSorted(const T* array, const int len,
                                  const char* name, const char* argname,
                                  const bool allowDuplicates) {
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
         printf( "care:%s: %s not in ascending order at index %d", name, argname, last + 1);
         return false;
      }
   }

   return true;
}

template <typename T>
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<const T>& array, const int len,
                                  const char* name, const char* argname,
                                  const bool allowDuplicates)
{
   return checkSorted<const T>(array.data(), len, name, argname, allowDuplicates);
}

#if defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

template <typename T>
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<T>& array, const int len,
                                  const char* name, const char* argname,
                                  const bool allowDuplicates)
{
   return checkSorted(care::host_device_ptr<const T>(array), len, name, argname, allowDuplicates);
}

#endif // defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

template<typename mapType>
CARE_HOST_DEVICE int BinarySearch(const mapType *map, const int start,
                             const int mapSize, const mapType num,
                             bool returnUpperBound = false) ;

template<typename mapType>
CARE_HOST_DEVICE int BinarySearch(const care::host_device_ptr<const mapType>& map, const int start,
                                  const int mapSize, const mapType num,
                                  bool returnUpperBound = false) ;

template<typename mapType>
CARE_HOST_DEVICE int BinarySearch(const care::host_device_ptr<mapType>& map, const int start,
                                  const int mapSize, const mapType num,
                                  bool returnUpperBound = false) ;

template <typename ArrayType, typename Exec>
inline void IntersectArrays(Exec,
                            care::host_device_ptr<const ArrayType> arr1, int size1, int start1,
                            care::host_device_ptr<const ArrayType> arr2, int size2, int start2,
                            care::host_device_ptr<int> &matches1, care::host_device_ptr<int> &matches2,
                            int *numMatches);

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
#ifdef RAJA_PARALLEL_ACTIVE
template <typename ArrayType>
inline void IntersectArrays(RAJAExec,
                            care::host_device_ptr<const ArrayType> arr1, int size1, int start1,
                            care::host_device_ptr<const ArrayType> arr2, int size2, int start2,
                            care::host_device_ptr<int> &matches1, care::host_device_ptr<int> &matches2,
                            int *numMatches) {
   *numMatches = 0 ;
   int smaller = (size1 < size2) ? size1 : size2 ;

   if (smaller <= 0 || start1 >= size1 || start2 >= size2) {
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
      checkSorted<ArrayType>(care::host_device_ptr<const ArrayType>(arr1.slice(start1)), size1, funcname, "arr1") ;
      checkSorted<ArrayType>(care::host_device_ptr<const ArrayType>(arr2.slice(start2)), size2, funcname, "arr2") ;
   }

   care::host_device_ptr<int> smallerMatches, largerMatches;
   int larger, smallStart, largeStart;
   care::host_device_ptr<const ArrayType> smallerArray, largerArray;

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
      searches[i] = i != smaller ? BinarySearch<ArrayType>(largerArray, largeStart, larger, smallerArray[i + smallStart]) : -1;
      matched[i] = i != smaller && searches[i] > -1;
   } CARE_STREAM_LOOP_END

   exclusive_scan<int, RAJAExec>(matched, nullptr, smaller + 1, RAJA::operators::plus<int>{}, 0, true);

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

template <typename ArrayType>
inline void IntersectArrays(RAJAExec exec,
                            care::host_device_ptr<ArrayType> arr1, int size1, int start1,
                            care::host_device_ptr<ArrayType> arr2, int size2, int start2,
                            care::host_device_ptr<int> &matches1, care::host_device_ptr<int> &matches2,
                            int *numMatches) {
   IntersectArrays<ArrayType>(exec,
                              care::host_device_ptr<const ArrayType>(arr1), size1, start1,
                              care::host_device_ptr<const ArrayType>(arr2), size2, start2,
                              matches1, matches2,
                              numMatches);
}

#endif



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
template <typename ArrayType>
inline void IntersectArrays(RAJA::seq_exec,
                            care::host_ptr<const ArrayType> arr1, int size1, int start1,
                            care::host_ptr<const ArrayType> arr2, int size2, int start2,
                            care::host_ptr<int> &matches1, care::host_ptr<int> &matches2,
                            int *numMatches) {
   *numMatches = 0 ;
   int smaller = (size1 < size2) ? size1 : size2 ;

   if (smaller <= 0 || start1 >= size1 || start2 >= size2) {
      matches1 = nullptr ;
      matches2 = nullptr ;
      return ;
   }
   else {
      matches1 = (int*) std::malloc(smaller * sizeof(int));
      matches2 = (int*) std::malloc(smaller * sizeof(int));
   }

   /* This algorithm assumes that the nodelists are sorted and unique */
#ifdef CARE_DEBUG
   bool checkIsSorted = true ;
#else
   bool checkIsSorted = false;
#endif

   if (checkIsSorted) {
      const char* funcname = "IntersectArrays" ;

      // allowDuplicates is false for this check by default
      checkSorted<ArrayType>(&arr1[start1], size1, funcname, "arr1") ;
      checkSorted<ArrayType>(&arr2[start2], size2, funcname, "arr2") ;
   }

   int i, j;
   i = j = 0 ;

   /* the host arrays */
   const ArrayType * A1, *A2;
   int * m1 = matches1;
   int * m2 = matches2;

   A1 = arr1;
   A1 += start1;
   A2 = arr2;
   A2 += start2;

   while (i < size1 && j < size2) {
      if ((A1)[i] < A2[j]) {
         ++i;
      }
      else if (A2[j] < A1[i]) {
         ++j;
      }
      else {
         m1[(*numMatches)] = i;
         m2[(*numMatches)] = j;
         (*numMatches)++;
         ++i;
         ++j;
      }
   }

   /* change the size of the array */
   if (*numMatches == 0) {
      std::free((int*) matches1);
      std::free((int*) matches2);
      matches1 = nullptr;
      matches2 = nullptr;
   }
   else {
      matches1 = (int*) std::realloc((int*) matches1, *numMatches * sizeof(int));
      matches2 = (int*) std::realloc((int*) matches2, *numMatches * sizeof(int));
   }
}

template <typename ArrayType>
inline void IntersectArrays(RAJA::seq_exec exec,
                            care::host_ptr<ArrayType> arr1, int size1, int start1,
                            care::host_ptr<ArrayType> arr2, int size2, int start2,
                            care::host_ptr<int> &matches1, care::host_ptr<int> &matches2,
                            int *numMatches) {
   IntersectArrays<ArrayType>(exec,
                              care::host_ptr<const ArrayType>(arr1), size1, start1,
                              care::host_ptr<const ArrayType>(arr2), size2, start2,
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
template <typename ArrayType>
inline void IntersectArrays(RAJA::seq_exec exec,
                            care::host_device_ptr<const ArrayType> arr1, int size1, int start1,
                            care::host_device_ptr<const ArrayType> arr2, int size2, int start2,
                            care::host_device_ptr<int> &matches1, care::host_device_ptr<int> &matches2,
                            int *numMatches) {
   care::host_ptr<int> matches1_tmp, matches2_tmp;

   IntersectArrays<ArrayType>(exec,
                              care::host_ptr<const ArrayType>(arr1), size1, start1,
                              care::host_ptr<const ArrayType>(arr2), size2, start2,
                              matches1_tmp, matches2_tmp, numMatches);

   matches1 = care::host_device_ptr<int>((int *) matches1_tmp, *numMatches, "IntersectArrays matches1");
   matches2 = care::host_device_ptr<int>((int *) matches2_tmp, *numMatches, "IntersectArrays matches2");

   return;
}

template <typename ArrayType>
inline void IntersectArrays(RAJA::seq_exec exec,
                            care::host_device_ptr<ArrayType> arr1, int size1, int start1,
                            care::host_device_ptr<ArrayType> arr2, int size2, int start2,
                            care::host_device_ptr<int> &matches1, care::host_device_ptr<int> &matches2,
                            int *numMatches) {
   IntersectArrays(exec,
                   care::host_device_ptr<const ArrayType>(arr1), size1, start1,
                   care::host_device_ptr<const ArrayType>(arr2), size2, start2,
                   matches1, matches2, numMatches);
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
 *             than num.
 ************************************************************************/

template <typename T>
CARE_HOST_DEVICE inline int BinarySearch(const T *map, const int start,
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
   checkSorted(&(map[start]), mapSize, "BinarySearch", "map", allowDuplicates) ;
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
         // Note: fix for last test in TEST(array_utils, binarysearch). This algorithm has failed to pick up the upper
         // bound above 1 in the array {0, 1, 1, 1, 1, 1, 6}. Having 1 repeated confused the algorithm.
         while ((khi < start + mapSize) && (map[khi] == num)) {
            ++khi;
         }

         // the upper option bounds num
         if ((khi < start + mapSize) && (map[khi] > num)) {
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

template<typename mapType>
CARE_HOST_DEVICE inline int BinarySearch(const care::host_device_ptr<mapType>& map, const int start,
                                         const int mapSize, const mapType num,
                                         bool returnUpperBound)
{
   return BinarySearch<const mapType>(map.data(), start, mapSize, num, returnUpperBound);
}

template<typename mapType>
CARE_HOST_DEVICE inline int BinarySearch(const care::host_device_ptr<const mapType>& map, const int start,
                                         const int mapSize, const mapType num,
                                         bool returnUpperBound)
{
   return BinarySearch<const mapType>(map.data(), start, mapSize, num, returnUpperBound);
}


#ifdef RAJA_PARALLEL_ACTIVE
/************************************************************************
 * Function  : uniqArray
 * Author(s) : Peter Robinson
 * Purpose   : GPU version of uniqArray, implements uniq using an exclusive
 *             scan.
  ************************************************************************/
template <typename T>
inline void uniqArray(RAJAExec, care::host_device_ptr<T>  Array, size_t len, care::host_device_ptr<T> & outArray, int & outLen, bool noCopy=false) {
   care::host_device_ptr<int> uniq(len+1,"uniqArray uniq");
   ArrayFill<int, RAJAExec>(uniq, len+1, 0);
   CARE_STREAM_LOOP(i, 0, len) {
      uniq[i] = (int) ((i == len-1) || (Array[i] < Array[i+1] || Array[i+1] < Array[i])) ;
   } CARE_STREAM_LOOP_END

   exclusive_scan<int, RAJAExec>(uniq, nullptr, len+1, RAJA::operators::plus<int>{}, 0, true);
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
inline int uniqArray(RAJAExec exec, care::host_device_ptr<T> & Array, size_t len, bool noCopy=false) {
   care::host_device_ptr<T> tmp;
   int newLen;
   uniqArray(exec, Array, len, tmp, newLen);
   if (noCopy) {
      Array.free();
      Array = tmp;
   }
   else {
      ArrayCopy<T>(Array, tmp, newLen);
      tmp.free();
   }
   return newLen;
}

#endif //RAJA_PARALLEL_ACTIVE

/************************************************************************
 * Function  : uniqArray
 * Author(s) : Peter Robinson
 * Purpose   : CPU version of uniqArray.
  ************************************************************************/
template <typename T>
inline void uniqArray(RAJA::seq_exec, care::host_device_ptr<T> Array, size_t len, care::host_device_ptr<T> & outArray, int & newLen) {
   CHAIDataGetter<T, RAJA::seq_exec> getter {};
   T * rawData = getter.getRawArrayData(Array);
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
         arrout[newLen] = rawData[i] ;

         /* skip over all the redundant elements */
         while ((i<len) && (rawData[i] == arrout[newLen])) {
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
inline int uniqArray(RAJA::seq_exec exec, care::host_device_ptr<T> & Array, size_t len, bool noCopy=false) {
   int newLength = 0;
   if (len > 0) {
      care::host_device_ptr<T> tmp;
      uniqArray(exec, Array, len, tmp, newLength);
      if (noCopy) {
         Array.free();
         Array = tmp;
      }
      else {
         ArrayCopy<T>(RAJA::seq_exec {}, Array, tmp, newLength);
         tmp.free();
      }
   }
   return newLength;
}

#ifdef RAJA_GPU_ACTIVE
template <typename T, typename Exec>
inline void sortArray(Exec, care::host_device_ptr<T> &Array, size_t len, int start, bool noCopy)
{
   printf("%s:%s","sortArray(Exec, care::host_device_ptr, len, start, noCopy)", "unspecialized version should not be called");
}

template <typename T, typename Exec>
inline void sortArray(Exec, care::host_device_ptr<T> &Array, size_t len)
{
   printf("%s:%s","sortArray(Exec, care::host_device_ptr, len)", "unspecialized version should not be called");
}

template <typename T>
inline void radixSortArray(care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy);

/************************************************************************
 * Function  : sortArray
 * Author(s) : Peter Robinson
 * Purpose   : GPU version of sortArray. Defaults to thrust::sort, but
 *             specialized to cub::DeviceRadix sort for data types that
 *             cub supports.
  ************************************************************************/
template <typename T>
inline void sortArray(RAJAExec, care::host_device_ptr<T> &Array, size_t len, int start, bool noCopy) ;

template <>
inline void sortArray(RAJAExec, care::host_device_ptr<int> & Array, size_t len, int start, bool noCopy) {
   radixSortArray(Array, len, start, noCopy);
}

template <>
inline void sortArray(RAJAExec, care::host_device_ptr<float> & Array, size_t len, int start, bool noCopy) {
   radixSortArray(Array, len, start, noCopy);
}

template <>
inline void sortArray(RAJAExec, care::host_device_ptr<double> & Array, size_t len, int start, bool noCopy) {
   radixSortArray(Array, len, start, noCopy);
}

#if CARE_HAVE_LLNL_GLOBALID

template <>
inline void sortArray(RAJAExec, care::host_device_ptr<globalID> & Array, size_t len, int start, bool noCopy) {
   radixSortArray(Array, len, start, noCopy);
}

#endif // CARE_HAVE_LLNL_GLOBALID

// This must be explicitly specialized for NVCC to link to the correct version.
template <typename T>
inline void sortArray(RAJAExec, care::host_device_ptr<T> &Array, size_t len);

template <>
inline void sortArray(RAJAExec, care::host_device_ptr<int> & Array, size_t len) {
   radixSortArray(Array, len, 0, false);
}

template <>
inline void sortArray(RAJAExec, care::host_device_ptr<float> & Array, size_t len) {
   radixSortArray(Array, len, 0, false);
}

template <>
inline void sortArray(RAJAExec, care::host_device_ptr<double> & Array, size_t len) {
   radixSortArray(Array, len, 0, false);
}

#if CARE_HAVE_LLNL_GLOBALID

template <>
inline void sortArray(RAJAExec, care::host_device_ptr<globalID> & Array, size_t len) {
   radixSortArray(Array, len, 0, false);
}

#endif // CARE_HAVE_LLNL_GLOBALID

/************************************************************************
 * Function  : radixSortArray
 * Author(s) : Peter Robinson
 * Purpose   : ManagedArray API to cub::DeviceRadixSort::SortKeys.
  ************************************************************************/
template <typename T>
inline void radixSortArray(care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy) {
   CHAIDataGetter<T, RAJAExec> getter {};
   CHAIDataGetter<char, RAJAExec> charGetter {};
   care::host_device_ptr<T> result(len,"radix_sort_result");
   auto * rawData = getter.getRawArrayData(Array) + start;
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
#if defined(__CUDACC__)
      cub::DeviceRadixSort::SortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, rawResult, len);
#elif defined(__HIPCC__)
      hipcub::DeviceRadixSort::SortKeys((void *)d_temp_storage, temp_storage_bytes, rawData, rawResult, len);
#endif   
   }
   // cleanup
   if (noCopy) {
      if (len > 0) {
         Array.free();
      }
      Array = result;
   }
   else {
      ArrayCopy<T>(Array, result, len, start, 0);
      if (len > 0) {
         result.free();
      }
   }
   if (len > 0) {
      tmpManaged.free();
   }
}
#endif // RAJA_GPU_ACTIVE

/************************************************************************
 * Function  : sortArray
 * Author(s) : Peter Robinson
 * Purpose   : CPU version of sortArray. Calls std::sort
  ************************************************************************/
template <typename T>
inline void sortArray(RAJA::seq_exec, care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy) {
   CHAIDataGetter<T, RAJA::seq_exec> getter {};
   T * rawData = getter.getRawArrayData(Array)+start;
   std::sort(rawData, rawData+len);
   noCopy = noCopy ;
}

template <typename T>
inline void sortArray(RAJA::seq_exec, care::host_device_ptr<T> &Array, size_t len)
{
   CHAIDataGetter<T, RAJA::seq_exec> getter {};
   T * rawData = getter.getRawArrayData(Array);
   std::sort(rawData, rawData+len);
}

/************************************************************************
* Function  : sort_uniq(<T>_ptr)
* Author(s) : Peter Robinson
* Purpose   : Sorts and uniques an array.
**************************************************************************/
template <typename T, typename Exec>
inline void sort_uniq(Exec e, care::host_device_ptr<T> * array, int * len, bool noCopy = false) {
   if ((*len) == 0) {
      if ((*array) != nullptr) {
         array->free();
         *array = nullptr;
      }
      return  ;
   }
   /* first sort the array */
   sortArray<T>(e, *array, *len, 0, noCopy);
   /* then unique it */
   *len = uniqArray<T>(e, *array, *len, noCopy);
}

/************************************************************************
* Function  : CompressArray<T>
* Author(s) : Peter Robinson
* Purpose   : Removes items at indices defined in removed from arr.
*           : Thread safe version of CompressArray.
*           : Note that thread safe version only requires removed to be sorted.
*           : Note also that it's a error if any index in removed is
*           : not found (beyond the end of arr).
**************************************************************************/
#ifdef RAJA_PARALLEL_ACTIVE
template <typename T>
inline void CompressArray(RAJAExec exec, care::host_device_ptr<T> & arr, const int arrLen,
                          care::host_device_ptr<int const> removed, const int removedLen, bool noCopy=false) {
   //GPU VERSION
   care::host_device_ptr<T> tmp(arrLen-removedLen, "CompressArray_tmp");
   int numKept = 0;
   SCAN_LOOP(i, 0, arrLen, pos, numKept,
             -1 == BinarySearch<int>(removed, 0, removedLen, i)) {
      tmp[pos] = arr[i];
   } SCAN_LOOP_END(arrLen, pos, numKept)

#ifdef CARE_DEBUG
   int numRemoved = arrLen - numKept;
   if (removedLen != numRemoved) {
      printf("Warning in CompressArray<T>: did not remove expected number of members!\n");
   }
#endif
   if (noCopy) {
      arr.free();
      arr = tmp;
   }
   else {
      ArrayCopy<T>(exec, arr, tmp, numKept);
      tmp.free();
   }
}

#endif // RAJA_PARALLEL_ACTIVE

/************************************************************************
 * Function  : InsertionSort
 * Author(s) : Rob Neely
 * Purpose   : Simple insertion sort function.  Should only be used on
 *             small arrays - otherwise use the qsort function from the
 *             standard C library.  Sorts in ascending order.
 ************************************************************************/
template <typename T>
inline CARE_HOST_DEVICE void InsertionSort(care::local_ptr<T> array, int len)
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
 * Function  : sortLocal
 * Author(s) : Benjamin Liu
 * Purpose   : General sort routine to call from within RAJA loops.
 *             Sorts in ascending order.
 ************************************************************************/
template <typename T>
inline CARE_HOST_DEVICE void sortLocal(care::local_ptr<T> array, int len)
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

/************************************************************************
 * Function  : uniqLocal
 * Author(s) : Benjamin Liu
 * Purpose   : Remove duplicates in-place from an array that is sorted
 *             in ascending order and updates len.
 *             For calls from within RAJA loops.
 *             Does not reallocate array.
 ************************************************************************/
template <typename T>
inline CARE_HOST_DEVICE void uniqLocal(care::local_ptr<T> array, int& len)
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

/************************************************************************
* Function  : CompressArray<T>
* Author(s) : Peter Robinson
* Purpose   : Removes items at indices defined in removed from arr.
*             Sequential Version of CompressArray
*             Requires both arr and removed to be sorted.
*             Note that it's a error if any index in removed is
*             not found (beyond the end of arr).
**************************************************************************/
template <typename T>
inline void CompressArray(RAJA::seq_exec, care::host_device_ptr<T> & arr, const int arrLen,
                          care::host_device_ptr<int const> removed, const int removedLen, bool noCopy=false) {
   // CPU VERSION

   int readLoc;
   int writeLoc = 0, numRemoved = 0;
   care::host_ptr<int const> removedHost = removed ;
   care::host_ptr<T> arrHost = arr ;
#ifdef CARE_DEBUG
   if (removedHost[removedLen-1] > arrLen-1) {
      printf("Warning in CompressArray<T> seq_exec: asking to remove entries not in array!\n");
   }
#endif
   for (readLoc = 0; readLoc < arrLen; ++readLoc) {
      if ((numRemoved == removedLen) || (readLoc < removedHost[numRemoved])) {
         arrHost[writeLoc++] = arrHost[readLoc];
      }
      else if (readLoc == removedHost[numRemoved]) {
         ++numRemoved;
      }
#ifdef CARE_DEBUG
      else {
         printf("Warning in CompressArray<int> seq_exec: list of removed members not sorted!\n");
      }
#endif
   }
#ifdef CARE_DEBUG
   if ((removedLen != numRemoved) || (writeLoc != arrLen - removedLen)) {
      printf("CompressArray<T> seq_exec: did not remove expected number of members!\n");
   }
#endif
   noCopy = noCopy;
}

template <typename T>
inline void CompressArray(care::host_device_ptr<T> & arr, const int arrLen,
                          care::host_device_ptr<int const> removed, const int removedLen, bool noCopy=false) {
   return CompressArray(RAJAExec(), arr, arrLen, removed, removedLen, noCopy);
}

template <typename T>
inline void ExpandArrayInPlace(RAJA::seq_exec, care::host_device_ptr<T> array, care::host_device_ptr<int const> indexSet, int length)
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

#ifdef RAJA_PARALLEL_ACTIVE
template <typename T>
inline void ExpandArrayInPlace(RAJAExec, care::host_device_ptr<T> array, care::host_device_ptr<int const> indexSet, int length)
{
   if (length > 0) {
      care::host_device_ptr<T> array_copy(length, "ExpandArrayInPlace array_copy");
      ArrayCopy<double>(array_copy, array, length);
      CARE_STREAM_LOOP(i, 0, length) {
         int nL = indexSet[i] ;
         array[nL] = array_copy[i] ;
      } CARE_STREAM_LOOP_END
      array_copy.free();
   }
}

#endif


/************************************************************************
 * Function  : ArrayFill
 * Author(s) : Peter Robinson
 * Purpose   : Fills a ManagedArray with the value given.
 * ************************************************************************/
template <typename T, typename Exec>
inline void ArrayFill(care::host_device_ptr<T> arr, int n, T val) {
   CARE_STREAM_LOOP(i, 0, n) {
      arr[i] = val;
   } CARE_STREAM_LOOP_END
}

template <typename T>
void ArrayFill(care::host_ptr<T> arr, int n, T val)  {
   for (int i = 0; i < n; ++i) {
      arr[i] = val;
   }
}

/************************************************************************
 * Function  : ArrayMin
 * Author(s) : Peter Robinson
 * Purpose   : Returns the minimum value in a ManagedArray
 * ************************************************************************/
template <typename T, typename Exec>
inline T ArrayMin(care::host_device_ptr<const T> arr, int n, T initVal, int startIndex)  {
   RAJAReduceMin<T> min { initVal };
   CARE_REDUCE_LOOP(k, startIndex, n) {
      min.min(arr[k]);
   } CARE_REDUCE_LOOP_END
   return (T)min;
}

template <typename T, typename Exec>
inline T ArrayMin(care::host_device_ptr<T> arr, int n, T initVal, int startIndex)  {
   return ArrayMin<T, Exec>((care::host_device_ptr<const T>)arr, n, initVal, startIndex);
}

template <typename T>
inline CARE_HOST_DEVICE T ArrayMin(care::local_ptr<const T> arr, int n, T initVal, int startIndex)  {
   T min = initVal;
   for (int k = startIndex; k < n; ++k) {
      min = CARE_MIN(min, arr[k]);
   }
   return min;
}

template <typename T>
inline CARE_HOST_DEVICE T ArrayMin(care::local_ptr<T> arr, int n, T initVal, int startIndex)  {
   return ArrayMin<T>((care::local_ptr<const T>)arr, n, initVal, startIndex);
}

/************************************************************************
 * Function  : ArrayMinLoc
 * Author(s) : Peter Robinson
 * Purpose   : Returns the minimum value in a ManagedArray
 * ************************************************************************/
template <typename T, typename Exec>
inline T ArrayMinLoc(care::host_device_ptr<const T> arr, int n, T initVal, int & loc)  {
   RAJAReduceMinLoc<T> min { initVal, -1 };
   CARE_REDUCE_LOOP(k, 0, n) {
      min.minloc(arr[k], k);
   } CARE_REDUCE_LOOP_END
   loc = min.getLoc();
   return (T)min;
}

/************************************************************************
 * Function  : ArrayMaxLoc
 * Author(s) : Peter Robinson
 * Purpose   : Returns the maximum value in a ManagedArray
 * ************************************************************************/
template <typename T, typename Exec>
inline T ArrayMaxLoc(care::host_device_ptr<const T> arr, int n, T initVal, int & loc)  {
   RAJAReduceMaxLoc<T> max { initVal, -1 };
   CARE_REDUCE_LOOP(k, 0, n) {
      max.maxloc(arr[k], k);
   } CARE_REDUCE_LOOP_END
   loc = max.getLoc();
   return (T)max;
}

/************************************************************************
 * Function  : ArrayMax
 * Author(s) : Peter Robinson
 * Purpose   : Returns the maximum value in a ManagedArray
 * ************************************************************************/
template <typename T, typename Exec>
inline T ArrayMax(care::host_device_ptr<const T> arr, int n, T initVal)  {
   RAJAReduceMax<T> max { initVal };
   CARE_REDUCE_LOOP(k, 0, n) {
      max.max(arr[k]);
   } CARE_REDUCE_LOOP_END
   return (T)max;
}

template <typename T, typename Exec>
inline T ArrayMax(care::host_device_ptr<T> arr, int n, T initVal)  {
   return ArrayMax<T, Exec>((care::host_device_ptr<const T>)arr, n, initVal);;
}

/************************************************************************
 * Function  : ArrayMax
 * Author(s) : Peter Robinson
 * Purpose   : Returns the maximum value in a ManagedArray
 * ************************************************************************/
template <typename T>
CARE_HOST_DEVICE inline T ArrayMax(care::local_ptr<const T> arr, int n, T initVal)  {
   T max = initVal;
   for (int k = 0; k < n; ++k) {
      max = CARE_MAX(max, arr[k]);
   }
   return max;
}

template <typename T>
CARE_HOST_DEVICE inline T ArrayMax(care::local_ptr<T> arr, int n, T initVal)  {
   return ArrayMax<T>((care::local_ptr<const T>)arr, n, initVal);
}

/************************************************************************
 * Function  : ArrayFind
 * Author(s) : Rob Neely, Alan Dayton
 * Purpose   : Returns the index of the first element in the the array
 *             that equals the value, or -1 if no match is found.
 ************************************************************************/
template <typename T>
int ArrayFind(care::host_device_ptr<const T> arr, const int len, const T val, const int start)
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
 * Function  : ArrayMax
 * Author(s) : Peter Robinson
 * Purpose   : Returns the maximum value in a ManagedArray.
 *             Wraps above care::host_device_ptr implementation with a care::host_ptr API.
 * ************************************************************************/
template <typename T>
inline T ArrayMax(care::host_ptr<const T> arr, int n, T initVal)  {
   return ArrayMax<T, RAJA::seq_exec>(care::host_device_ptr<const T>((const T *) arr, n, "ArrayMaxTmp"), n, initVal);
}

template <typename T>
inline T ArrayMax(care::host_ptr<T> arr, int n, T initVal)  {
   return ArrayMax<T>((care::host_ptr<const T>)arr, n, initVal);
}

/************************************************************************
 * Function  : ArrayMinMax
 * Author(s) : Peter Robinson
 * Purpose   : Stores Minimum / Maximum values of arr (as a double) in outMin / outMax;
 *             If mask was such that no values were compared, returns 0, outMin will be -DBL_MAX, outMax will be DBL_MAX
 *             Otherwise, returns 1.
 * ************************************************************************/
template <typename T, typename ReducerType, typename RAJAExec>
int ArrayMinMax(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, int n, double *outMin, double *outMax) {
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
inline int ArrayMinMax(care::host_device_ptr<T> arr, care::host_device_ptr<int> mask, int n, double *outMin, double *outMax) {
 return ArrayMinMax<T, ReducerType, RAJAExec>((care::host_device_ptr<const T>)arr, (care::host_device_ptr<int const>)mask, n, outMin, outMax);
}

#if CARE_HAVE_LLNL_GLOBALID

template <typename Exec=RAJAExec>
int ArrayMinMax(care::host_device_ptr<const globalID> arr, care::host_device_ptr<int const> mask, int n, double *outMin, double *outMax);

template <typename Exec>
inline int ArrayMinMax(care::host_device_ptr<const globalID> arr, care::host_device_ptr<int const> mask, int n, double *outMin, double *outMax) {
   return ArrayMinMax<globalID, GIDTYPE, Exec>(arr, mask, n, outMin, outMax);
}

#endif // CARE_HAVE_LLNL_GLOBALID

/************************************************************************
 * Function  : ArrayMinMax
 * Author(s) : Peter Robinson
 * Purpose   : Stores Minimum / Maximum values of arr (as a double) in outMin / outMax;
 *             If mask was such that no values were compared, returns 0, outMin will be -DBL_MAX, outMax will be DBL_MAX
 *             Otherwise, returns 1.
 *             care::local_ptr API to support calls from within RAJA contexts.
 * ************************************************************************/
template <typename T>
CARE_HOST_DEVICE inline int ArrayMinMax(care::local_ptr<const T> arr, care::local_ptr<int const> mask, int n, double *outMin, double *outMax) {
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
               min = CARE_MIN(min, (double)arr[i]);
               max = CARE_MAX(max, (double)arr[i]);
            }
         }
         if (min != DBL_MAX ||
             max != -DBL_MAX) {
            result = true;
         }
      }
      else {
         for (int i = 0; i < n; ++i) {
            min = CARE_MIN(min, (double)arr[i]);
            max = CARE_MAX(max, (double)arr[i]);
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
CARE_HOST_DEVICE inline int ArrayMinMax(care::local_ptr<T> arr, care::local_ptr<int> mask, int n, double *outMin, double *outMax)
{
   return ArrayMinMax<T>((care::local_ptr<const T>)arr, (care::local_ptr<int const>)mask, n, outMin, outMax);
}

/************************************************************************
 * Function  : ArrayCount
 * Author(s) : Peter Robinson
 * Purpose   : Returns count of occurence of val in arr.
 * ************************************************************************/
template <typename T, typename Exec>
inline int  ArrayCount(care::host_device_ptr<const T> arr, int length, T val)  {
   RAJAReduceSum<int> count { 0 };
   CARE_REDUCE_LOOP(k, 0, length) {
      count += (int) arr[k] == val;
   } CARE_REDUCE_LOOP_END
   return (int) count ;
}

/************************************************************************
 * Function  : ArraySum
 * Author(s) : Peter Robinson
 * Purpose   : Returns the sum of all values in a ManagedArray
 * ************************************************************************/
template <typename T, typename ReduceType, typename Exec>
inline T ArraySum(care::host_device_ptr<const T> arr, int n, T initVal)  {
   ReduceType iVal = initVal;
   RAJAReduceSum<ReduceType> sum { iVal };
   CARE_REDUCE_LOOP(k, 0, n) {
      sum += arr[k];
   } CARE_REDUCE_LOOP_END
   return (T) (ReduceType) sum;
}

/************************************************************************
 * Function  : ArraySumSubset
 * Author(s) : Peter Robinson
 * Purpose   : Returns the sum of values in arr at indices in subset.
 * Note      : length n refers to length of subset, not array
 * ************************************************************************/
template <typename T, typename ReduceType, typename Exec>
inline T ArraySumSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int n, T initVal) {
   ReduceType iVal = initVal;
   RAJAReduceSum<ReduceType> sum { iVal };
   CARE_REDUCE_LOOP(k, 0, n) {
      sum += arr[subset[k]];
   } CARE_REDUCE_LOOP_END
   return (T) (ReduceType) sum;
}

/************************************************************************
 * Function  : ArrayMaskedSumSubset
 * Author(s) : Peter Robinson
 * Purpose   : Returns the sum of values in arr at indices in subset.
 * ************************************************************************/
template <typename T, typename ReduceType, typename Exec>
inline T ArrayMaskedSumSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, care::host_device_ptr<int const> subset, int n, T initVal) {
   ReduceType iVal =initVal;
   RAJAReduceSum<ReduceType> sum { iVal };
   CARE_REDUCE_LOOP(k, 0, n) {
      int ndx = subset[k];
      if (mask[ndx] == 0) {
         sum += arr[ndx];
      }
   } CARE_REDUCE_LOOP_END
   return (T) (ReduceType) sum;
}

/************************************************************************
 * Function  : ArrayMaskedSum
 * Author(s) : Peter Robinson
 * Purpose   : Returns the sum of values in arr at indices where mask is 0.
 * ************************************************************************/
template<typename T, typename ReduceType, typename Exec>
T ArrayMaskedSum(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, int n, T initVal)
{
   ReduceType iVal = initVal;
   RAJAReduceSum<ReduceType> sum { iVal };

   CARE_STREAM_LOOP(i, 0, n) {
      sum += arr[i] * T(mask[i] == 0);
   } CARE_STREAM_LOOP_END

   return (T) (ReduceType) sum ;
}

/************************************************************************
 * Function  : FindIndexGT
 * Author(s) : Peter Robinson
 * Purpose   : Returns an index of an array where the value is greater than
 *             limit
 * ************************************************************************/
template <typename T, typename Exec>
inline int FindIndexGT(care::host_device_ptr<const T> arr, int n, T limit) {
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
inline int FindIndexMax(care::host_device_ptr<const T> arr, int n) {
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
template<typename T>
inline void ArrayCopy(care::host_device_ptr<T> into, care::host_device_ptr<const T> from,
                      int n, int start1, int start2) {
   ArrayCopy(RAJAExec {}, into, from, n, start1, start2);
}

/************************************************************************
 * Function  : ArrayCopy
 * Author(s) : Peter Robinson
 * Purpose   : Copies from one ManagedArray into another. from and to
 *             should not have the same or overlapping memory addresses.
 * ************************************************************************/
template<typename T, typename Exec>
inline void ArrayCopy(Exec,
                      care::host_device_ptr<T> into, care::host_device_ptr<const T> from,
                      int n, int start1, int start2) {
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
template<typename T>
inline void ArrayCopy(RAJA::seq_exec,
                      care::host_device_ptr<T> into, care::host_device_ptr<const T> from,
                      int n, int start1, int start2) {
   CARE_SEQUENTIAL_LOOP(i, 0, n) {
      into[i+start1] = from[i+start2];
   } CARE_SEQUENTIAL_LOOP_END
}

/************************************************************************
 * Function  : ArrayDup
 * Author(s) : Peter Robinson
 * Purpose   : Duplicates a ManagedArray.
 * ************************************************************************/
template <typename T, typename Exec>
inline care::host_device_ptr<T> ArrayDup(care::host_device_ptr<const T> from, int len) {
   if (from == nullptr) { // don't make a new array for null input
     return nullptr;
   } else {
     care::host_device_ptr<T> newArray(len,"ArrayDup newArray");
     CARE_STREAM_LOOP(i, 0, len) {
        newArray[i] = from[i];
     } CARE_STREAM_LOOP_END
     return newArray;
   }
}

//******************************************************************************
// If subset is defined, calls SumIntArraySubset, otherwise calls
// SumIntArray.
// @author Peter Robinson
//
template<typename T, typename Exec>
T SumArrayOrArraySubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const>  subset, int n) {
   if (subset) {
      return ArraySumSubset<T, T, Exec>(arr, subset, n, T(0));
   }
   else {
      return ArraySum<T, T, Exec>(arr, n, T(0));
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
template<typename T, typename ReduceType=T, typename Exec=RAJAExec>
inline T PickAndPerformSum(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> mask, care::host_device_ptr<int const> subset, int n)
{
   if (mask) {
      if (subset) {
         return ArrayMaskedSumSubset<T, ReduceType, Exec>(arr, mask, subset, n, T(0));
      }
      else {
         return ArrayMaskedSum<T, ReduceType, Exec>(arr, mask, n, T(0));
      }
   }
   else {
      return SumArrayOrArraySubset<T, Exec>(arr, subset, n);
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
int FindIndexMinAboveThresholds(care::host_device_ptr<const T> arr, int n,
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
int FindIndexMinSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int lenset)
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
int FindIndexMinSubsetAboveThresholds(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int lenset,
                                      care::host_device_ptr<double const> thresholds, double cutoff,
                                      int * thresholdIndex)
{
   int  ndx = -1 ;
   if (thresholds) {
      RAJAReduceMinLoc<T> min { std::numeric_limits<T>::max(), -1 };
      RAJAReduceMinLoc<T> thresholdmin { std::numeric_limits<T>::max(), -1 };
      CARE_REDUCE_LOOP(i, 0, lenset) {
         int curr = subset[i] ;
         if (thresholds[i] > cutoff) { // if threshold were sized as arr, this would be thresholds[curr]
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
int PickAndPerformFindMinIndex(care::host_device_ptr<const T> arr,
                               care::host_device_ptr<int const> mask,
                               care::host_device_ptr<int const> subset, int n,
                               care::host_device_ptr<double const> thresholds,
                               double cutoff,
                               int *thresholdIndex)
{
   int minIndex;
   if (subset) {
      minIndex =  FindIndexMinSubsetAboveThresholds<T, Exec>(arr, subset, n,
                                                             thresholds, cutoff,
                                                             thresholdIndex);
   }
   else {
      minIndex =   FindIndexMinAboveThresholds<T, Exec>(arr, n, thresholds, cutoff,
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
int FindIndexMaxAboveThresholds(care::host_device_ptr<const T> arr, int n,
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
int FindIndexMaxSubset(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int lenset)
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
int FindIndexMaxSubsetAboveThresholds(care::host_device_ptr<const T> arr, care::host_device_ptr<int const> subset, int lenset,
                                      care::host_device_ptr<double const> thresholds, double cutoff,
                                      int * thresholdIndex)
{
   int  ndx = -1 ;
   ndx = -1;
   if (thresholds) {
      RAJAReduceMaxLoc<T> max { std::numeric_limits<T>::lowest(), -1 };
      RAJAReduceMaxLoc<T> thresholdmax { std::numeric_limits<T>::lowest(), -1 };
      CARE_REDUCE_LOOP(i, 0, lenset) {
         int curr = subset[i] ;

         if (thresholds[i] > cutoff) {
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
int PickAndPerformFindMaxIndex(care::host_device_ptr<const T> arr,
                               care::host_device_ptr<int const> mask,
                               care::host_device_ptr<int const> subset, int n,
                               care::host_device_ptr<double const> thresholds,
                               double cutoff,
                               int *thresholdIndex)
{
   int maxIndex;
   if (subset) {
      maxIndex =  FindIndexMaxSubsetAboveThresholds<T, Exec>(arr, subset, n,
                                                             thresholds, cutoff,
                                                             thresholdIndex);
   }
   else {
      maxIndex =   FindIndexMaxAboveThresholds<T, Exec>(arr, n, thresholds, cutoff,
                                                        thresholdIndex);
   }
   if (mask && n > 0) {
      if (maxIndex >= 0 && mask.pick(maxIndex) == 1) {
         maxIndex = -1;
      }
   }
   return maxIndex;
}

} // end namespace care_utils

#endif // !defined(_CARE_ARRAY_UTILS_H_)

