//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef CARE_ALGORITHM_DECL_H
#define CARE_ALGORITHM_DECL_H

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/host_device_ptr.h"
#include "care/host_ptr.h"
#include "care/local_ptr.h"
#include "care/policies.h"

#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

#define CARE_MAX(a,b) a > b ? a : b
#define CARE_MIN(a,b) a < b ? a : b

namespace care {
template <class T, class Size, class U>
void fill_n(care::host_device_ptr<T> arr, Size n, const U& val);

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

template<typename T>
CARE_HOST_DEVICE void ArrayCopy(
                           care::local_ptr<T> into,
                           care::local_ptr<const T> from,
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

#ifdef CARE_GPUCC
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
#endif // defined(CARE_GPUCC)

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

// TODO openMP parallel implementation
#if defined(CARE_GPUCC)

template <typename T>
void sortArray(RAJADeviceExec, care::host_device_ptr<T> &Array, size_t len, int start, bool noCopy) ;

template <typename T>
void radixSortArray(care::host_device_ptr<T> & Array, size_t len, int start, bool noCopy);

template <typename T>
void sortArray(RAJADeviceExec, care::host_device_ptr<T> &Array, size_t len);

#endif // defined(CARE_GPUCC)

// TODO should this have an unused noCopy parameter?
template <typename T>
void uniqArray(RAJA::seq_exec, care::host_device_ptr<T> Array, size_t len, care::host_device_ptr<T> & outArray, int & newLen);
template <typename T>
int uniqArray(RAJA::seq_exec exec, care::host_device_ptr<T> & Array, size_t len, bool noCopy=false);
#ifdef CARE_GPUCC
template <typename T>
void uniqArray(RAJADeviceExec, care::host_device_ptr<T>  Array, size_t len, care::host_device_ptr<T> & outArray, int & outLen, bool noCopy=false);
template <typename T>
int uniqArray(RAJADeviceExec exec, care::host_device_ptr<T> & Array, size_t len, bool noCopy=false);
#endif // defined(CARE_GPUCC)

template <typename T, typename Exec>
void sort_uniq(Exec e, care::host_device_ptr<T> * array, int * len, bool noCopy = false);

template <typename T>
void CompressArray(RAJA::seq_exec, care::host_device_ptr<T> & arr, const int arrLen,
                   care::host_device_ptr<int const> removed, const int removedLen, bool noCopy=false);
#ifdef CARE_GPUCC
template <typename T>
void CompressArray(RAJADeviceExec exec, care::host_device_ptr<T> & arr, const int arrLen,
                   care::host_device_ptr<int const> removed, const int removedLen, bool noCopy=false);
#endif // defined(CARE_GPUCC)
template <typename T>
void CompressArray(care::host_device_ptr<T> & arr, const int arrLen,
                   care::host_device_ptr<int const> removed, const int removedLen, bool noCopy=false);

template <typename T>
CARE_HOST_DEVICE void InsertionSort(care::local_ptr<T> array, int len);

template <typename T>
CARE_HOST_DEVICE void sortLocal(care::local_ptr<T> array, int len);

template <typename T>
CARE_HOST_DEVICE void uniqLocal(care::local_ptr<T> array, int& len);

template <typename T>
void ExpandArrayInPlace(RAJA::seq_exec, care::host_device_ptr<T> array, care::host_device_ptr<int const> indexSet, int length);
#ifdef CARE_GPUCC
template <typename T>
void ExpandArrayInPlace(RAJADeviceExec, care::host_device_ptr<T> array, care::host_device_ptr<int const> indexSet, int length);
#endif // defined(CARE_GPUCC)

} // end namespace care

#endif // !defined(CARE_ALGORITHM_DECL_H)

