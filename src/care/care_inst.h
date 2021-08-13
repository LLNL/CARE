#ifndef __CARE_INST_H
#define __CARE_INST_H

// CARE config header
#include "care/config.h"

#ifdef CARE_ENABLE_EXTERN_INSTANTIATE

///////////////////////////////////////////////////////////////////////////////

#define CARE_TEMPLATE_ARRAY_TYPE int
#include "care/KeyValueSorter_inst.h"

#define CARE_TEMPLATE_ARRAY_TYPE float
#include "care/KeyValueSorter_inst.h"

#define CARE_TEMPLATE_ARRAY_TYPE double
#include "care/KeyValueSorter_inst.h"

#if CARE_HAVE_LLNL_GLOBALID
#define CARE_TEMPLATE_ARRAY_TYPE globalID
#include "care/KeyValueSorter_inst.h"
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_EXTERN
#undef CARE_EXTERN
#endif

#ifdef CARE_INSTANTIATE
#define CARE_EXTERN
#else
#define CARE_EXTERN extern
#endif

namespace care {

///////////////////////////////////////////////////////////////////////////////

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const int*, const int, const char*, const char*, const bool) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const float*, const int, const char*, const char*, const bool) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const double*, const int, const char*, const char*, const bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const globalID*, const int, const char*, const char*, const bool) ;
#endif

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<const int>&, const int, const char*, const char*, const bool) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<const float>&, const int, const char*, const char*, const bool) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<const double>&, const int, const char*, const char*, const bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<const globalID>&, const int, const char*, const char*, const bool) ;
#endif

#if defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<int>&, const int, const char*, const char*, const bool) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<float>&, const int, const char*, const char*, const bool) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<double>&, const int, const char*, const char*, const bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool checkSorted(const care::host_device_ptr<globalID>&, const int, const char*, const char*, const bool) ;
#endif

#endif // defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJADeviceExec, care::host_device_ptr<const int>, int, int, care::host_device_ptr<const int>, int, int, care::host_device_ptr<int> &, care::host_device_ptr<int> &, int *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJADeviceExec, care::host_device_ptr<const globalID>, int, int, care::host_device_ptr<const globalID>, int, int, care::host_device_ptr<int> &, care::host_device_ptr<int> &, int *) ;
#endif

CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJADeviceExec, care::host_device_ptr<int>, int, int, care::host_device_ptr<int>, int, int, care::host_device_ptr<int> &, care::host_device_ptr<int> &, int *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJADeviceExec, care::host_device_ptr<globalID>, int, int, care::host_device_ptr<globalID>, int, int, care::host_device_ptr<int> &, care::host_device_ptr<int> &, int *) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJA::seq_exec, care::host_ptr<const int>, int, int, care::host_ptr<const int>, int, int, care::host_ptr<int> &, care::host_ptr<int> &, int *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJA::seq_exec, care::host_ptr<const globalID>, int, int, care::host_ptr<const globalID>, int, int, care::host_ptr<int> &, care::host_ptr<int> &, int *) ;
#endif

CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJA::seq_exec, care::host_ptr<int>, int, int, care::host_ptr<int>, int, int, care::host_ptr<int> &, care::host_ptr<int> &, int *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJA::seq_exec, care::host_ptr<globalID>, int, int, care::host_ptr<globalID>, int, int, care::host_ptr<int> &, care::host_ptr<int> &, int *) ;
#endif

CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJA::seq_exec, care::host_device_ptr<const int>, int, int, care::host_device_ptr<const int>, int, int, care::host_device_ptr<int> &, care::host_device_ptr<int> &, int *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJA::seq_exec, care::host_device_ptr<const globalID>, int, int, care::host_device_ptr<const globalID>, int, int, care::host_device_ptr<int> &, care::host_device_ptr<int> &, int *) ;
#endif

CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJA::seq_exec, care::host_device_ptr<int>, int, int, care::host_device_ptr<int>, int, int, care::host_device_ptr<int> &, care::host_device_ptr<int> &, int *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void IntersectArrays(RAJA::seq_exec, care::host_device_ptr<globalID>, int, int, care::host_device_ptr<globalID>, int, int, care::host_device_ptr<int> &, care::host_device_ptr<int> &, int *) ;
#endif

///////////////////////////////////////////////////////////////////////////////

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int BinarySearch(const int *, const int, const int, const int, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int BinarySearch(const globalID *, const int, const int, const globalID, bool) ;
#endif

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int BinarySearch(const care::host_device_ptr<int>&, const int, const int, const int, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int BinarySearch(const care::host_device_ptr<globalID>&, const int, const int, const globalID, bool) ;
#endif

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int BinarySearch(const care::host_device_ptr<const int>&, const int, const int, const int, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int BinarySearch(const care::host_device_ptr<const globalID>&, const int, const int, const globalID, bool) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
void uniqArray(RAJADeviceExec, care::host_device_ptr<int>, size_t, care::host_device_ptr<int> &, int &, bool) ;
CARE_EXTERN template CARE_DLL_API
void uniqArray(RAJADeviceExec, care::host_device_ptr<float>, size_t, care::host_device_ptr<float> &, int &, bool) ;
CARE_EXTERN template CARE_DLL_API
void uniqArray(RAJADeviceExec, care::host_device_ptr<double>, size_t, care::host_device_ptr<double> &, int &, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void uniqArray(RAJADeviceExec, care::host_device_ptr<globalID>, size_t, care::host_device_ptr<globalID> &, int &, bool) ;
#endif

CARE_EXTERN template CARE_DLL_API
int uniqArray(RAJADeviceExec, care::host_device_ptr<int> &, size_t, bool) ;
CARE_EXTERN template CARE_DLL_API
int uniqArray(RAJADeviceExec, care::host_device_ptr<float> &, size_t, bool) ;
CARE_EXTERN template CARE_DLL_API
int uniqArray(RAJADeviceExec, care::host_device_ptr<double> &, size_t, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int uniqArray(RAJADeviceExec, care::host_device_ptr<globalID> &, size_t, bool) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
void uniqArray(RAJA::seq_exec, care::host_device_ptr<int>, size_t, care::host_device_ptr<int> &, int &) ;
CARE_EXTERN template CARE_DLL_API
void uniqArray(RAJA::seq_exec, care::host_device_ptr<float>, size_t, care::host_device_ptr<float> &, int &) ;
CARE_EXTERN template CARE_DLL_API
void uniqArray(RAJA::seq_exec, care::host_device_ptr<double>, size_t, care::host_device_ptr<double> &, int &) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void uniqArray(RAJA::seq_exec, care::host_device_ptr<globalID>, size_t, care::host_device_ptr<globalID> &, int &) ;
#endif

CARE_EXTERN template CARE_DLL_API
int uniqArray(RAJA::seq_exec exec, care::host_device_ptr<int> &, size_t, bool) ;
CARE_EXTERN template CARE_DLL_API
int uniqArray(RAJA::seq_exec exec, care::host_device_ptr<float> &, size_t, bool) ;
CARE_EXTERN template CARE_DLL_API
int uniqArray(RAJA::seq_exec exec, care::host_device_ptr<double> &, size_t, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int uniqArray(RAJA::seq_exec exec, care::host_device_ptr<globalID> &, size_t, bool) ;
#endif

///////////////////////////////////////////////////////////////////////////////

// TODO openMP parallel implementation
#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
void sortArray(RAJADeviceExec, care::host_device_ptr<int> &, size_t, int, bool) ;
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJADeviceExec, care::host_device_ptr<float> &, size_t, int, bool) ;
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJADeviceExec, care::host_device_ptr<double> &, size_t, int, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJADeviceExec, care::host_device_ptr<globalID> &, size_t, int, bool) ;
#endif

CARE_EXTERN template CARE_DLL_API
void sortArray(RAJADeviceExec, care::host_device_ptr<int> &, size_t) ;
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJADeviceExec, care::host_device_ptr<float> &, size_t) ;
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJADeviceExec, care::host_device_ptr<double> &, size_t) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJADeviceExec, care::host_device_ptr<globalID> &, size_t) ;
#endif

CARE_EXTERN template CARE_DLL_API
void radixSortArray(care::host_device_ptr<int> &, size_t, int, bool) ;
CARE_EXTERN template CARE_DLL_API
void radixSortArray(care::host_device_ptr<float> &, size_t, int, bool) ;
CARE_EXTERN template CARE_DLL_API
void radixSortArray(care::host_device_ptr<double> &, size_t, int, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void radixSortArray(care::host_device_ptr<globalID> &, size_t, int, bool) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
void sortArray(RAJA::seq_exec, care::host_device_ptr<int> &, size_t, int, bool) ;
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJA::seq_exec, care::host_device_ptr<float> &, size_t, int, bool) ;
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJA::seq_exec, care::host_device_ptr<double> &, size_t, int, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJA::seq_exec, care::host_device_ptr<globalID> &, size_t, int, bool) ;
#endif

CARE_EXTERN template CARE_DLL_API
void sortArray(RAJA::seq_exec, care::host_device_ptr<int> &, size_t) ;
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJA::seq_exec, care::host_device_ptr<float> &, size_t) ;
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJA::seq_exec, care::host_device_ptr<double> &, size_t) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void sortArray(RAJA::seq_exec, care::host_device_ptr<globalID> &, size_t) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
void sort_uniq(RAJADeviceExec, care::host_device_ptr<int> *, int *, bool) ;
CARE_EXTERN template CARE_DLL_API
void sort_uniq(RAJADeviceExec, care::host_device_ptr<float> *, int *, bool) ;
CARE_EXTERN template CARE_DLL_API
void sort_uniq(RAJADeviceExec, care::host_device_ptr<double> *, int *, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void sort_uniq(RAJADeviceExec, care::host_device_ptr<globalID> *, int *, bool) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
void sort_uniq(RAJA::seq_exec, care::host_device_ptr<int> *, int *, bool) ;
CARE_EXTERN template CARE_DLL_API
void sort_uniq(RAJA::seq_exec, care::host_device_ptr<float> *, int *, bool) ;
CARE_EXTERN template CARE_DLL_API
void sort_uniq(RAJA::seq_exec, care::host_device_ptr<double> *, int *, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void sort_uniq(RAJA::seq_exec, care::host_device_ptr<globalID> *, int *, bool) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJADeviceExec, care::host_device_ptr<bool> &, const int, care::host_device_ptr<int const>, const int, bool) ;
CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJADeviceExec, care::host_device_ptr<int> &, const int, care::host_device_ptr<int const>, const int, bool) ;
CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJADeviceExec, care::host_device_ptr<float> &, const int, care::host_device_ptr<int const>, const int, bool) ;
CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJADeviceExec, care::host_device_ptr<double> &, const int, care::host_device_ptr<int const>, const int, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJADeviceExec, care::host_device_ptr<globalID> &, const int, care::host_device_ptr<int const>, const int, bool) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJA::seq_exec, care::host_device_ptr<bool> &, const int, care::host_device_ptr<int const>, const int, bool) ;
CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJA::seq_exec, care::host_device_ptr<int> &, const int, care::host_device_ptr<int const>, const int, bool) ;
CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJA::seq_exec, care::host_device_ptr<float> &, const int, care::host_device_ptr<int const>, const int, bool) ;
CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJA::seq_exec, care::host_device_ptr<double> &, const int, care::host_device_ptr<int const>, const int, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void CompressArray(RAJA::seq_exec, care::host_device_ptr<globalID> &, const int, care::host_device_ptr<int const>, const int, bool) ;
#endif

CARE_EXTERN template CARE_DLL_API
void CompressArray(care::host_device_ptr<bool> &, const int, care::host_device_ptr<int const>, const int, bool) ;
CARE_EXTERN template CARE_DLL_API
void CompressArray(care::host_device_ptr<int> &, const int, care::host_device_ptr<int const>, const int, bool) ;
CARE_EXTERN template CARE_DLL_API
void CompressArray(care::host_device_ptr<float> &, const int, care::host_device_ptr<int const>, const int, bool) ;
CARE_EXTERN template CARE_DLL_API
void CompressArray(care::host_device_ptr<double> &, const int, care::host_device_ptr<int const>, const int, bool) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void CompressArray(care::host_device_ptr<globalID> &, const int, care::host_device_ptr<int const>, const int, bool) ;
#endif

///////////////////////////////////////////////////////////////////////////////

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void InsertionSort(care::local_ptr<int>, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void InsertionSort(care::local_ptr<float>, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void InsertionSort(care::local_ptr<double>, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void InsertionSort(care::local_ptr<globalID>, int) ;
#endif

///////////////////////////////////////////////////////////////////////////////

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void sortLocal(care::local_ptr<int>, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void sortLocal(care::local_ptr<float>, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void sortLocal(care::local_ptr<double>, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void sortLocal(care::local_ptr<globalID>, int) ;
#endif

///////////////////////////////////////////////////////////////////////////////

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void uniqLocal(care::local_ptr<int>, int&) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void uniqLocal(care::local_ptr<float>, int&) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void uniqLocal(care::local_ptr<double>, int&) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE void uniqLocal(care::local_ptr<globalID>, int&) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJADeviceExec, care::host_device_ptr<bool>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJADeviceExec, care::host_device_ptr<int>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJADeviceExec, care::host_device_ptr<float>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJADeviceExec, care::host_device_ptr<double>, care::host_device_ptr<int const>, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJADeviceExec, care::host_device_ptr<globalID>, care::host_device_ptr<int const>, int) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJA::seq_exec, care::host_device_ptr<bool>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJA::seq_exec, care::host_device_ptr<int>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJA::seq_exec, care::host_device_ptr<float>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJA::seq_exec, care::host_device_ptr<double>, care::host_device_ptr<int const>, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void ExpandArrayInPlace(RAJA::seq_exec, care::host_device_ptr<globalID>, care::host_device_ptr<int const>, int) ;
#endif

///////////////////////////////////////////////////////////////////////////////

CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<bool>, int, const bool&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<int>, int, const int&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<float>, int, const int&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<float>, int, const float&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<double>, int, const int&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<double>, int, const float&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<double>, int, const double&) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<globalID>, int, const globalID&) ;
#endif

CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<bool>, size_t, const bool&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<int>, size_t, const int&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<float>, size_t, const int&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<float>, size_t, const float&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<double>, size_t, const int&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<double>, size_t, const float&) ;
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<double>, size_t, const double&) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void fill_n(care::host_device_ptr<globalID>, size_t, const globalID&) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
bool ArrayMin<bool, RAJADeviceExec>(care::host_device_ptr<const bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMin<int, RAJADeviceExec>(care::host_device_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMin<float, RAJADeviceExec>(care::host_device_ptr<const float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMin<double, RAJADeviceExec>(care::host_device_ptr<const double>, int, double, int) ;
// TODO GID not implemented

CARE_EXTERN template CARE_DLL_API
bool ArrayMin<bool, RAJADeviceExec>(care::host_device_ptr<bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMin<int, RAJADeviceExec>(care::host_device_ptr<int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMin<float, RAJADeviceExec>(care::host_device_ptr<float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMin<double, RAJADeviceExec>(care::host_device_ptr<double>, int, double, int) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
bool ArrayMin<bool, RAJA::seq_exec>(care::host_device_ptr<const bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMin<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMin<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMin<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int, double, int) ;
// TODO GID not implemented

CARE_EXTERN template CARE_DLL_API
bool ArrayMin<bool, RAJA::seq_exec>(care::host_device_ptr<bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMin<int, RAJA::seq_exec>(care::host_device_ptr<int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMin<float, RAJA::seq_exec>(care::host_device_ptr<float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMin<double, RAJA::seq_exec>(care::host_device_ptr<double>, int, double, int) ;
// TODO GID not implemented

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool ArrayMin(care::local_ptr<const bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMin(care::local_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE float ArrayMin(care::local_ptr<const float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE double ArrayMin(care::local_ptr<const double>, int, double, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE globalID ArrayMin(care::local_ptr<const globalID>, int, globalID, int) ;
#endif

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool ArrayMin(care::local_ptr<bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMin(care::local_ptr<int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE float ArrayMin(care::local_ptr<float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE double ArrayMin(care::local_ptr<double>, int, double, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE globalID ArrayMin(care::local_ptr<globalID>, int, globalID, int) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
bool ArrayMinLoc<bool, RAJADeviceExec>(care::host_device_ptr<const bool>, int, bool, int &) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinLoc<int, RAJADeviceExec>(care::host_device_ptr<const int>, int, int, int &) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMinLoc<float, RAJADeviceExec>(care::host_device_ptr<const float>, int, float, int &) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMinLoc<double, RAJADeviceExec>(care::host_device_ptr<const double>, int, double, int &) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
bool ArrayMinLoc<bool, RAJA::seq_exec>(care::host_device_ptr<const bool>, int, bool, int &) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinLoc<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int, int, int &) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMinLoc<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int, float, int &) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMinLoc<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int, double, int &) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
bool ArrayMaxLoc<bool, RAJADeviceExec>(care::host_device_ptr<const bool>, int, bool, int &) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMaxLoc<int, RAJADeviceExec>(care::host_device_ptr<const int>, int, int, int &) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMaxLoc<float, RAJADeviceExec>(care::host_device_ptr<const float>, int, float, int &) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMaxLoc<double, RAJADeviceExec>(care::host_device_ptr<const double>, int, double, int &) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
bool ArrayMaxLoc<bool, RAJA::seq_exec>(care::host_device_ptr<const bool>, int, bool, int &) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMaxLoc<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int, int, int &) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMaxLoc<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int, float, int &) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMaxLoc<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int, double, int &) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
bool ArrayMax<bool, RAJADeviceExec>(care::host_device_ptr<const bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMax<int, RAJADeviceExec>(care::host_device_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMax<float, RAJADeviceExec>(care::host_device_ptr<const float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMax<double, RAJADeviceExec>(care::host_device_ptr<const double>, int, double, int) ;
// TODO GID not implemented

CARE_EXTERN template CARE_DLL_API
bool ArrayMax<bool, RAJADeviceExec>(care::host_device_ptr<bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMax<int, RAJADeviceExec>(care::host_device_ptr<int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMax<float, RAJADeviceExec>(care::host_device_ptr<float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMax<double, RAJADeviceExec>(care::host_device_ptr<double>, int, double, int) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
bool ArrayMax<bool, RAJA::seq_exec>(care::host_device_ptr<const bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMax<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMax<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMax<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int, double, int) ;
// TODO GID not implemented

CARE_EXTERN template CARE_DLL_API
bool ArrayMax<bool, RAJA::seq_exec>(care::host_device_ptr<bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMax<int, RAJA::seq_exec>(care::host_device_ptr<int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMax<float, RAJA::seq_exec>(care::host_device_ptr<float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMax<double, RAJA::seq_exec>(care::host_device_ptr<double>, int, double, int) ;
// TODO GID not implemented

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool ArrayMax(care::local_ptr<const bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMax(care::local_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE float ArrayMax(care::local_ptr<const float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE double ArrayMax(care::local_ptr<const double>, int, double, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE globalID ArrayMax(care::local_ptr<const globalID>, int, globalID, int) ;
#endif

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE bool ArrayMax(care::local_ptr<bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMax(care::local_ptr<int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE float ArrayMax(care::local_ptr<float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE double ArrayMax(care::local_ptr<double>, int, double, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE globalID ArrayMax(care::local_ptr<globalID>, int, globalID, int) ;
#endif

CARE_EXTERN template CARE_DLL_API
bool ArrayMax(care::host_ptr<const bool>, int, bool, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMax(care::host_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMax(care::host_ptr<const float>, int, float, int) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMax(care::host_ptr<const double>, int, double, int) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

CARE_EXTERN template CARE_DLL_API
int ArrayFind(care::host_device_ptr<const bool>, const int, const bool, const int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayFind(care::host_device_ptr<const int>, const int, const int, const int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayFind(care::host_device_ptr<const float>, const int, const float, const int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayFind(care::host_device_ptr<const double>, const int, const double, const int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int ArrayFind(care::host_device_ptr<const globalID>, const int, const globalID, const int) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<int, int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<float, float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<double, double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, double *, double *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<globalID, GIDTYPE, RAJADeviceExec>(care::host_device_ptr<const globalID>, care::host_device_ptr<int const>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<RAJADeviceExec>(care::host_device_ptr<const globalID>, care::host_device_ptr<int const>, int, double *, double *) ;
#endif

CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<int, int, RAJADeviceExec>(care::host_device_ptr<int>, care::host_device_ptr<int>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<float, float, RAJADeviceExec>(care::host_device_ptr<float>, care::host_device_ptr<int>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<double, double, RAJADeviceExec>(care::host_device_ptr<double>, care::host_device_ptr<int>, int, double *, double *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<globalID, GIDTYPE, RAJADeviceExec>(care::host_device_ptr<globalID>, care::host_device_ptr<int>, int, double *, double *) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<int, int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<float, float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<double, double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, double *, double *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<globalID, GIDTYPE, RAJA::seq_exec>(care::host_device_ptr<const globalID>, care::host_device_ptr<int const>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<RAJA::seq_exec>(care::host_device_ptr<const globalID>, care::host_device_ptr<int const>, int, double *, double *) ;
#endif

CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<int, int, RAJA::seq_exec>(care::host_device_ptr<int>, care::host_device_ptr<int>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<float, float, RAJA::seq_exec>(care::host_device_ptr<float>, care::host_device_ptr<int>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<double, double, RAJA::seq_exec>(care::host_device_ptr<double>, care::host_device_ptr<int>, int, double *, double *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int ArrayMinMax<globalID, GIDTYPE, RAJA::seq_exec>(care::host_device_ptr<globalID>, care::host_device_ptr<int>, int, double *, double *) ;
#endif

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<const int>, care::local_ptr<int const>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<const float>, care::local_ptr<int const>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<const double>, care::local_ptr<int const>, int, double *, double *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<const globalID>, care::local_ptr<int const>, int, double *, double *) ;
#endif

CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<int>, care::local_ptr<int>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<float>, care::local_ptr<int>, int, double *, double *) ;
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<double>, care::local_ptr<int>, int, double *, double *) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
CARE_HOST_DEVICE int ArrayMinMax(care::local_ptr<globalID>, care::local_ptr<int>, int, double *, double *) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int ArrayCount<bool, RAJADeviceExec>(care::host_device_ptr<const bool>, int, bool) ;
CARE_EXTERN template CARE_DLL_API
int ArrayCount<int, RAJADeviceExec>(care::host_device_ptr<const int>, int, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayCount<float, RAJADeviceExec>(care::host_device_ptr<const float>, int, float) ;
CARE_EXTERN template CARE_DLL_API
int ArrayCount<double, RAJADeviceExec>(care::host_device_ptr<const double>, int, double) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int ArrayCount<globalID, RAJADeviceExec>(care::host_device_ptr<const globalID>, int, globalID) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int ArrayCount<bool, RAJA::seq_exec>(care::host_device_ptr<const bool>, int, bool) ;
CARE_EXTERN template CARE_DLL_API
int ArrayCount<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int, int) ;
CARE_EXTERN template CARE_DLL_API
int ArrayCount<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int, float) ;
CARE_EXTERN template CARE_DLL_API
int ArrayCount<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int, double) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int ArrayCount<globalID, RAJA::seq_exec>(care::host_device_ptr<const globalID>, int, globalID) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int ArraySum<int, int, RAJADeviceExec>(care::host_device_ptr<const int>, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArraySum<float, float, RAJADeviceExec>(care::host_device_ptr<const float>, int, float) ;
CARE_EXTERN template CARE_DLL_API
double ArraySum<double, double, RAJADeviceExec>(care::host_device_ptr<const double>, int, double) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int ArraySum<int, int, RAJA::seq_exec>(care::host_device_ptr<const int>, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArraySum<float, float, RAJA::seq_exec>(care::host_device_ptr<const float>, int, float) ;
CARE_EXTERN template CARE_DLL_API
double ArraySum<double, double, RAJA::seq_exec>(care::host_device_ptr<const double>, int, double) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int ArraySumSubset<int, int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArraySumSubset<float, float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, float) ;
CARE_EXTERN template CARE_DLL_API
double ArraySumSubset<double, double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, double) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int ArraySumSubset<int, int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArraySumSubset<float, float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, float) ;
CARE_EXTERN template CARE_DLL_API
double ArraySumSubset<double, double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, double) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int ArrayMaskedSumSubset<int, int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMaskedSumSubset<float, float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, float) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMaskedSumSubset<double, double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, double) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int ArrayMaskedSumSubset<int, int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMaskedSumSubset<float, float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, float) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMaskedSumSubset<double, double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, double) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int ArrayMaskedSum<int, int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMaskedSum<float, float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, float) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMaskedSum<double, double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, double) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int ArrayMaskedSum<int, int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, int) ;
CARE_EXTERN template CARE_DLL_API
float ArrayMaskedSum<float, float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, float) ;
CARE_EXTERN template CARE_DLL_API
double ArrayMaskedSum<double, double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, double) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int FindIndexGT<int, RAJADeviceExec>(care::host_device_ptr<const int>, int, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexGT<float, RAJADeviceExec>(care::host_device_ptr<const float>, int, float) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexGT<double, RAJADeviceExec>(care::host_device_ptr<const double>, int, double) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int FindIndexGT<globalID, RAJADeviceExec>(care::host_device_ptr<const globalID>, int, globalID) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int FindIndexGT<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexGT<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int, float) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexGT<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int, double) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
int FindIndexGT<globalID, RAJA::seq_exec>(care::host_device_ptr<const globalID>, int, globalID) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int FindIndexMax<int, RAJADeviceExec>(care::host_device_ptr<const int>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMax<float, RAJADeviceExec>(care::host_device_ptr<const float>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMax<double, RAJADeviceExec>(care::host_device_ptr<const double>, int) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int FindIndexMax<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMax<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMax<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJADeviceExec, care::host_device_ptr<bool>, care::host_device_ptr<const bool>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJADeviceExec, care::host_device_ptr<int>, care::host_device_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJADeviceExec, care::host_device_ptr<float>, care::host_device_ptr<const float>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJADeviceExec, care::host_device_ptr<double>, care::host_device_ptr<const double>, int, int, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJADeviceExec, care::host_device_ptr<globalID>, care::host_device_ptr<const globalID>, int, int, int) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJA::seq_exec, care::host_device_ptr<bool>, care::host_device_ptr<const bool>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJA::seq_exec, care::host_device_ptr<int>, care::host_device_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJA::seq_exec, care::host_device_ptr<float>, care::host_device_ptr<const float>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJA::seq_exec, care::host_device_ptr<double>, care::host_device_ptr<const double>, int, int, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(RAJA::seq_exec, care::host_device_ptr<globalID>, care::host_device_ptr<const globalID>, int, int, int) ;
#endif

CARE_EXTERN template CARE_DLL_API
void ArrayCopy(care::host_device_ptr<bool>, care::host_device_ptr<const bool>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(care::host_device_ptr<int>, care::host_device_ptr<const int>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(care::host_device_ptr<float>, care::host_device_ptr<const float>, int, int, int) ;
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(care::host_device_ptr<double>, care::host_device_ptr<const double>, int, int, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
void ArrayCopy(care::host_device_ptr<globalID>, care::host_device_ptr<const globalID>, int, int, int) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<bool> ArrayDup<bool, RAJADeviceExec>(care::host_device_ptr<const bool>, int) ;
CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<int> ArrayDup<int, RAJADeviceExec>(care::host_device_ptr<const int>, int) ;
CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<float> ArrayDup<float, RAJADeviceExec>(care::host_device_ptr<const float>, int) ;
CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<double> ArrayDup<double, RAJADeviceExec>(care::host_device_ptr<const double>, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<globalID> ArrayDup<globalID, RAJADeviceExec>(care::host_device_ptr<const globalID>, int) ;
#endif

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<bool> ArrayDup<bool, RAJA::seq_exec>(care::host_device_ptr<const bool>, int) ;
CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<int> ArrayDup<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int) ;
CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<float> ArrayDup<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int) ;
CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<double> ArrayDup<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int) ;
#if CARE_HAVE_LLNL_GLOBALID
CARE_EXTERN template CARE_DLL_API
care::host_device_ptr<globalID> ArrayDup<globalID, RAJA::seq_exec>(care::host_device_ptr<const globalID>, int) ;
#endif

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int SumArrayOrArraySubset<int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
float SumArrayOrArraySubset<float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
double SumArrayOrArraySubset<double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int SumArrayOrArraySubset<int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
float SumArrayOrArraySubset<float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
double SumArrayOrArraySubset<double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int PickAndPerformSum<int, int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
float PickAndPerformSum<float, float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
double PickAndPerformSum<double, double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int PickAndPerformSum<int, int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
float PickAndPerformSum<float, float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
double PickAndPerformSum<double, double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int FindIndexMinAboveThresholds<int, RAJADeviceExec>(care::host_device_ptr<const int>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinAboveThresholds<float, RAJADeviceExec>(care::host_device_ptr<const float>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinAboveThresholds<double, RAJADeviceExec>(care::host_device_ptr<const double>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int FindIndexMinAboveThresholds<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinAboveThresholds<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinAboveThresholds<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubset<int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubset<float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubset<double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubset<int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubset<float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubset<double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubsetAboveThresholds<int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubsetAboveThresholds<float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubsetAboveThresholds<double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubsetAboveThresholds<int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubsetAboveThresholds<float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMinSubsetAboveThresholds<double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMinIndex<int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMinIndex<float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMinIndex<double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMinIndex<int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMinIndex<float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMinIndex<double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int FindIndexMaxAboveThresholds<int, RAJADeviceExec>(care::host_device_ptr<const int>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxAboveThresholds<float, RAJADeviceExec>(care::host_device_ptr<const float>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxAboveThresholds<double, RAJADeviceExec>(care::host_device_ptr<const double>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int FindIndexMaxAboveThresholds<int, RAJA::seq_exec>(care::host_device_ptr<const int>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxAboveThresholds<float, RAJA::seq_exec>(care::host_device_ptr<const float>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxAboveThresholds<double, RAJA::seq_exec>(care::host_device_ptr<const double>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubset<int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubset<float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubset<double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubset<int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubset<float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubset<double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubsetAboveThresholds<int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubsetAboveThresholds<float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubsetAboveThresholds<double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubsetAboveThresholds<int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubsetAboveThresholds<float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int FindIndexMaxSubsetAboveThresholds<double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

#ifdef CARE_GPUCC

CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMaxIndex<int, RAJADeviceExec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMaxIndex<float, RAJADeviceExec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMaxIndex<double, RAJADeviceExec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

#endif // defined(CARE_GPUCC)

CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMaxIndex<int, RAJA::seq_exec>(care::host_device_ptr<const int>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMaxIndex<float, RAJA::seq_exec>(care::host_device_ptr<const float>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
CARE_EXTERN template CARE_DLL_API
int PickAndPerformFindMaxIndex<double, RAJA::seq_exec>(care::host_device_ptr<const double>, care::host_device_ptr<int const>, care::host_device_ptr<int const>, int, care::host_device_ptr<double const>, double, int *) ;
// TODO GID not implemented

///////////////////////////////////////////////////////////////////////////////

} // namespace care

#else // CARE_ENABLE_EXTERN_INSTANTIATE

// Just include the header if we are not using external instantiations
#include "care/KeyValueSorter.h"

#endif // else CARE_ENABLE_EXTERN_INSTANTIATE

#endif // __CARE_INST_H
