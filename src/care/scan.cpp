//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#define GPU_ACTIVE

#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/scan.h"

#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

namespace care {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Template helper functions for implementations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// exclusive scan functionality

template <typename T, typename Exec, typename Fn>
void exclusive_scan(chai::ManagedArray<T> data, chai::ManagedArray<T> outData,
                    int size, Fn binop, T val, bool inPlace) {
   if (size > 1 && data != nullptr) {
      CHAIDataGetter<T, Exec> D {};
      T * rawData = D.getRawArrayData(data);
      T * rawDataEnd = rawData+size;
      if (inPlace) {
         RAJA::exclusive_scan_inplace<Exec, T *, T, Fn>(Exec {}, rawData, rawDataEnd, binop, val);
      }
      else {
         T * rawOutData = D.getRawArrayData(outData);
         RAJA::exclusive_scan<Exec, T*, T*, T, Fn>(Exec {},
                                                   rawData,
                                                   rawDataEnd,
                                                   rawOutData,
                                                   binop, val);
      }
   }
   else {
      if ( size == 1) {
         if (! inPlace) {
            if (outData != nullptr) {
               outData.set(0,val);
            }
         } else {
            if (data != nullptr) {
               data.set(0,val);
            }
         }
      }
      else {
         printf("care::scan - unhandled combination of size, data, and outData\n - no-op will occur.\n");
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// inclusive scan functionality
template <typename T, typename Exec, typename Fn>
void inclusive_scan(chai::ManagedArray<T> data, chai::ManagedArray<T> outData,
                    int size, Fn binop, bool inPlace) {
   CHAIDataGetter<T, Exec> D {};
   T * rawData = D.getRawArrayData(data);
   T * rawDataEnd = rawData+size;
   if (inPlace) {
      RAJA::inclusive_scan_inplace<Exec, T *, Fn>(Exec {}, rawData, rawDataEnd, binop);
   }
   else {
      T * rawOutData = D.getRawArrayData(outData);
      RAJA::inclusive_scan<Exec, T*, T*, Fn>(Exec {},
                                             rawData,
                                             rawDataEnd,
                                             rawOutData,
                                             binop);
   }
}

// typesafe wrapper for out of place scan
template <typename T, typename Exec, typename Fn>
void inclusive_scan(chai::ManagedArray<const T> inData, chai::ManagedArray<T> outData,
                    int size, Fn binop)
{ 
    const bool inPlace = false;
    inclusive_scan<T, Exec, Fn>(*reinterpret_cast<chai::ManagedArray<T> *>(&inData), outData, size, binop, inPlace);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function implementations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// exclusive scan functionality

void exclusive_scan(RAJA::seq_xec, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, int val, bool inPlace)
{
   exclusive_scan<int, RAJA::seq_exec, RAJA::operators::plus<int>>(data, outData, size,
                                                                   RAJA::operators::plus<int>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size, int val, bool inPlace)
{ 
    const bool inPlace = false;
    exclusive_scan(RAJA::seq_exec{}, *reinterpret_cast<chai::ManagedArray<int> *>(&inData), outData, size, val, inPlace);
}

#if CARE_HAVE_LLNL_GLOBALID

void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<globalID> data, chai::ManagedArray<globalID> outData,
                    int size, globalID val, bool inPlace)
{
   exclusive_scan<globalID, RAJA::seq_exec, RAJA::operators::plus<GIDTYPE>>(data, outData, size,
                                                                            RAJA::operators::plus<GIDTYPE>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<const globalID> inData, chai::ManagedArray<globalID> outData,
                    int size, globalID val, bool inPlace)
{ 
    const bool inPlace = false;
    exclusive_scan(RAJA::seq_exec{}, *reinterpret_cast<chai::ManagedArray<globalID> *>(&inData), outData, size, val, inPlace);
}

#endif // CARE_HAVE_LLNL_GLOBALID

#ifdef RAJA_PARALLEL_ACTIVE

void exclusive_scan(RAJAExec, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, int val, bool inPlace)
{
   exclusive_scan<int, RAJAExec, RAJA::operators::plus<int>>(data, outData, size,
                                                             RAJA::operators::plus<int>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(RAJAExec, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size, int val, bool inPlace)
{ 
    const bool inPlace = false;
    exclusive_scan(RAJAExec{}, *reinterpret_cast<chai::ManagedArray<int> *>(&inData), outData, size, val, inPlace);
}

#if CARE_HAVE_LLNL_GLOBALID

void exclusive_scan(RAJAExec, chai::ManagedArray<globalID> data, chai::ManagedArray<globalID> outData,
                    int size, globalID val, bool inPlace)
{
   exclusive_scan<globalID, RAJAExec, RAJA::operators::plus<GIDTYPE>>(data, outData, size,
                                                                      RAJA::operators::plus<GIDTYPE>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(RAJAExec, chai::ManagedArray<const globalID> inData, chai::ManagedArray<globalID> outData,
                    int size, globalID val, bool inPlace)
{ 
    const bool inPlace = false;
    exclusive_scan(RAJAExec{}, *reinterpret_cast<chai::ManagedArray<globalID> *>(&inData), outData, size, val, inPlace);
}

#endif // CARE_HAVE_LLNL_GLOBALID

#endif // defined(RAJA_PARALLEL_ACTIVE)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// inclusive scan functionality

void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, bool inPlace)
{
   inclusive_scan<int, RAJA::seq_exec, RAJA::operators::plus<int>>(data, outData, size,
                                                                   RAJA::operators::plus<int>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(RAJA::seq_exec{}, *reinterpret_cast<chai::ManagedArray<int> *>(&inData), outData, size, inPlace);
}

#if CARE_HAVE_LLNL_GLOBALID

void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<globalID> data, chai::ManagedArray<globalID> outData,
                    int size, bool inPlace)
{
   inclusive_scan<globalID, RAJA::seq_exec, RAJA::operators::plus<GIDTYPE>>(data, outData, size,
                                                                            RAJA::operators::plus<GIDTYPE>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<const globalID> inData, chai::ManagedArray<globalID> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(RAJA::seq_exec{}, *reinterpret_cast<chai::ManagedArray<globalID> *>(&inData), outData, size, inPlace);
}

#ifdef RAJA_PARALLEL_ACTIVE

void inclusive_scan(RAJAExec, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, bool inPlace)
{
   inclusive_scan<int, RAJAExec, RAJA::operators::plus<int>>(data, outData, size,
                                                             RAJA::operators::plus<int>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(RAJAExec, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(RAJAExec{}, *reinterpret_cast<chai::ManagedArray<int> *>(&inData), outData, size, inPlace);
}

#if CARE_HAVE_LLNL_GLOBALID

void inclusive_scan(RAJAExec, chai::ManagedArray<globalID> data, chai::ManagedArray<globalID> outData,
                    int size, bool inPlace)
{
   inclusive_scan<globalID, RAJAExec, RAJA::operators::plus<GIDTYPE>>(data, outData, size,
                                                                      RAJA::operators::plus<GIDTYPE>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(RAJAExec, chai::ManagedArray<const globalID> inData, chai::ManagedArray<globalID> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(RAJAExec{}, *reinterpret_cast<chai::ManagedArray<globalID> *>(&inData), outData, size, inPlace);
}

#endif // defined(RAJA_PARALLEL_ACTIVE)

#endif // CARE_HAVE_LLNL_GLOBALID

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// scan count accessors

void getFinalScanCountFromPinned(chai::ManagedArray<int> scanvar_length, int& scanCount)
{
   CARE_CHECKED_HOST_KERNEL_WITH_REF_START(scan_loop_check, scanCount) {
      scanCount = scanvar_length[0];
   } CARE_CHECKED_HOST_KERNEL_WITH_REF_END(scan_loop_check)
}

void getFinalScanCount(chai::ManagedArray<int> scanvar, int length, int& scanCount)
{
   scanCount = scanvar.pick(length);
}

#if CARE_HAVE_LLNL_GLOBALID

void getFinalScanCountFromPinned(chai::ManagedArray<globalID> scanvar_length, globalID& scanCount)
{
   CARE_CHECKED_HOST_KERNEL_WITH_REF_START(scan_loop_check, scanCount) {
      scanCount = scanvar_length[0];
   } CARE_CHECKED_HOST_KERNEL_WITH_REF_END(scan_loop_check)
}

void getFinalScanCount(chai::ManagedArray<globalID> scanvar, int length, globalID& scanCount)
{
   scanCount = scanvar.pick(length);
}

#endif // CARE_HAVE_LLNL_GLOBALID

} // namespace care

