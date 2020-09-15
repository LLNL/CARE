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

#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

#define CARE_MAX(a,b) a > b ? a : b
#define CARE_MIN(a,b) a < b ? a : b

namespace care {
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

#ifdef CARE_GPU_ACTIVE

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

// This must be explicitly specialized for NVCC to link to the correct version.
template <typename T>
inline void sortArray(RAJAExec, care::host_device_ptr<T> &Array, size_t len);

#endif // CARE_GPU_ACTIVE

#if CARE_HAVE_LLNL_GLOBALID

template <typename Exec=RAJAExec>
int ArrayMinMax(care::host_device_ptr<const globalID> arr, care::host_device_ptr<int const> mask, int n, double *outMin, double *outMax);

#endif // CARE_HAVE_LLNL_GLOBALID

} // end namespace care

#include "care/array_utils.inl"

#endif // !defined(_CARE_ARRAY_UTILS_H_)

