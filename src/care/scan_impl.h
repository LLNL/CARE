//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

// This header should be included once for each execution policy, defined by
// CARE_SCAN_EXEC.  This header should only be needed within CARE proper and
// should not be included by the host code.

#ifndef CARE_SCAN_EXEC
#error "CARE_SCAN_EXEC must be defined"
#endif

#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/scan.h"

#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

namespace care {

#ifndef _CARE_SCAN_INST_H_
#define _CARE_SCAN_INST_H_

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Template helper functions for implementations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// exclusive scan functionality

template <typename T, typename Exec, typename Fn, typename ValueType=T>
void exclusive_scan(chai::ManagedArray<T> data, chai::ManagedArray<T> outData,
                    int size, Fn binop, ValueType val, bool inPlace) {
   if (size > 1 && data != nullptr) {
      CHAIDataGetter<T, Exec> D {};
      ValueType * rawData = D.getRawArrayData(data);
      ValueType * rawDataEnd = rawData+size;
      if (inPlace) {
         RAJA::exclusive_scan_inplace(Exec {}, rawData, rawDataEnd, binop, val);
      }
      else {
         ValueType * rawOutData = D.getRawArrayData(outData);
         RAJA::exclusive_scan(Exec {}, rawData, rawDataEnd, rawOutData, binop, val);
      }
   }
   else {
      if ( size == 1) {
         if (! inPlace) {
            if (outData != nullptr) {
               outData.set(0, (T)val);
            }
         } else {
            if (data != nullptr) {
               data.set(0, (T)val);
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
template <typename T, typename Exec, typename Fn, typename ValueType=T>
void inclusive_scan(chai::ManagedArray<T> data, chai::ManagedArray<T> outData,
                    int size, Fn binop, bool inPlace) {
   CHAIDataGetter<T, Exec> D {};
   ValueType * rawData = D.getRawArrayData(data);
   ValueType * rawDataEnd = rawData+size;
   if (inPlace) {
      RAJA::inclusive_scan_inplace(Exec {}, rawData, rawDataEnd, binop);
   }
   else {
      ValueType * rawOutData = D.getRawArrayData(outData);
      RAJA::inclusive_scan(Exec {}, rawData, rawDataEnd, rawOutData, binop);
   }
}

#endif // defined(_CARE_SCAN_INST_H_)

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function implementations
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// exclusive scan functionality

void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, int val, bool inPlace)
{
   exclusive_scan<int, CARE_SCAN_EXEC, RAJA::operators::plus<int>>(data, outData, size,
                                                                   RAJA::operators::plus<int>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size, int val)
{ 
    const bool inPlace = false;
    exclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<int> *>(&inData), outData, size, val, inPlace);
}

void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<float> data, chai::ManagedArray<float> outData,
                    int size, float val, bool inPlace)
{
   exclusive_scan<float, CARE_SCAN_EXEC, RAJA::operators::plus<float>>(data, outData, size,
                                                                       RAJA::operators::plus<float>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const float> inData, chai::ManagedArray<float> outData,
                    int size, float val)
{ 
    const bool inPlace = false;
    exclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<float> *>(&inData), outData, size, val, inPlace);
}

void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<double> data, chai::ManagedArray<double> outData,
                    int size, double val, bool inPlace)
{
   exclusive_scan<double, CARE_SCAN_EXEC, RAJA::operators::plus<double>>(data, outData, size,
                                                                       RAJA::operators::plus<double>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const double> inData, chai::ManagedArray<double> outData,
                    int size, double val)
{ 
    const bool inPlace = false;
    exclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<double> *>(&inData), outData, size, val, inPlace);
}

#if CARE_HAVE_LLNL_GLOBALID

void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<globalID> data, chai::ManagedArray<globalID> outData,
                    int size, globalID val, bool inPlace)
{
   exclusive_scan<globalID, CARE_SCAN_EXEC, RAJA::operators::plus<GIDTYPE>, GIDTYPE>(data, outData, size,
                                                                                     RAJA::operators::plus<GIDTYPE>{}, val.Value(), inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const globalID> inData, chai::ManagedArray<globalID> outData,
                    int size, globalID val)
{ 
    const bool inPlace = false;
    exclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<globalID> *>(&inData), outData, size, val, inPlace);
}

#endif // CARE_HAVE_LLNL_GLOBALID

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// inclusive scan functionality

void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, bool inPlace)
{
   inclusive_scan<int, CARE_SCAN_EXEC, RAJA::operators::plus<int>>(data, outData, size,
                                                                   RAJA::operators::plus<int>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<int> *>(&inData), outData, size, inPlace);
}

void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<float> data, chai::ManagedArray<float> outData,
                    int size, bool inPlace)
{
   inclusive_scan<float, CARE_SCAN_EXEC, RAJA::operators::plus<float>>(data, outData, size,
                                                                       RAJA::operators::plus<float>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const float> inData, chai::ManagedArray<float> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<float> *>(&inData), outData, size, inPlace);
}

void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<double> data, chai::ManagedArray<double> outData,
                    int size, bool inPlace)
{
   inclusive_scan<double, CARE_SCAN_EXEC, RAJA::operators::plus<double>>(data, outData, size,
                                                                       RAJA::operators::plus<double>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const double> inData, chai::ManagedArray<double> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<double> *>(&inData), outData, size, inPlace);
}

#if CARE_HAVE_LLNL_GLOBALID

void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<globalID> data, chai::ManagedArray<globalID> outData,
                    int size, bool inPlace)
{
   inclusive_scan<globalID, CARE_SCAN_EXEC, RAJA::operators::plus<GIDTYPE>, GIDTYPE>(data, outData, size,
                                                                                     RAJA::operators::plus<GIDTYPE>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const globalID> inData, chai::ManagedArray<globalID> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<globalID> *>(&inData), outData, size, inPlace);
}

#endif // CARE_HAVE_LLNL_GLOBALID

} // namespace care

#undef CARE_SCAN_EXEC

