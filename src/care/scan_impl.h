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

#include "care/CHAICallback.h"
#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/ExecutionSpace.h"
#include "care/scan.h"

#include "umpire/util/backtrace.hpp"

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
void exclusive_scan(chai::ManagedArray<T> data, //!< [in/out] Input data (output also if in place)
                    chai::ManagedArray<T> outData, //!< [out] Output data if not in place
                    int size, //!< [in] Number of elements in input/output data
                    Fn binop, //!< [in] The operation to perform (such as addition)
                    ValueType val, //!< [in] The starting value
                    bool inPlace) { //!< [in] Whether or not to do the operations in place
   if (size > 0) {
      if (inPlace) {
         if (!data) {
            printf("[CARE] Warning: Invalid arguments to care::exclusive_scan. If inPlace is true, data cannot be nullptr.\n");
            return;
         }
      }
      else {
         if (!outData) {
            printf("[CARE] Warning: Invalid arguments to care::exclusive_scan. If inPlace is false, outData cannot be nullptr.\n");
            return;
         }
         else if (size > 1 && !data) {
            printf("[CARE] Warning: Invalid arguments to care::exclusive_scan. If inPlace is false and size > 1, data cannot be nullptr.\n");
            return;
         }
      }

#if !defined(CHAI_DISABLE_RM)
      if (size > 1) {
         // Bounds checking is only available with the resource manager because the ManagedArray::size() is not
         // reliable if it is cast to a different template type.
         const chai::PointerRecord* dataRecord = chai::ArrayManager::getInstance()->getPointerRecord((void *)data.data(care::ExecutionSpace::CPU, false));
         int dataSize = dataRecord->m_size/sizeof(T);
         bool warned = false;

         if (dataSize < size) {
            const char* dataName = CHAICallback::getName(dataRecord);
            if (dataName == nullptr) {
               dataName = "array" ;
            }
            printf("[CARE] Warning: Invalid argument to care::exclusive_scan. Size %d < %d in input %s.\n", dataSize, size, dataName);
            warned = true;
         }

         if (!inPlace) {
            const chai::PointerRecord* outDataRecord = chai::ArrayManager::getInstance()->getPointerRecord((void *)outData.data(care::ExecutionSpace::CPU, false)());
            int outDataSize = dataRecord->m_size/sizeof(T);

            if (outDataSize < size) {
               const char* outDataName = CHAICallback::getName(outDataRecord);
               if (outDataName == nullptr) {
                  outDataName = "array" ;
               }
               printf("[CARE] Warning: Invalid argument to care::exclusive_scan. Size %d < %d in output %s.\n", outDataSize, size, outDataName);
               warned = true;
            }
         }

         if (warned) {
            umpire::util::backtrace bt;
            umpire::util::backtracer<umpire::util::trace_always>::get_backtrace(bt);
            std::string stack = umpire::util::backtracer<umpire::util::trace_always>::print(bt);
            printf("%s", stack.c_str());
         }
      }
#endif

      if (size == 1) {
         if (inPlace) {
            data.set(0, (T) val);
         }
         else {
            outData.set(0, (T) val);
         }
      }
      else {
         CHAIDataGetter<T, Exec> D {};
         ValueType * rawData = D.getRawArrayData(data);

         if (inPlace) {
            RAJA::exclusive_scan_inplace<Exec>(RAJA::make_span(rawData, size),
                                               binop, val);
         }
         else {
            ValueType * rawOutData = D.getRawArrayData(outData);
            RAJA::exclusive_scan<Exec>(RAJA::make_span(rawData, size),
                                       RAJA::make_span(rawOutData, size),
                                       binop, val);
         }
      }
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// inclusive scan functionality
template <typename T, typename Exec, typename Fn, typename ValueType=T>
void inclusive_scan(chai::ManagedArray<T> data, chai::ManagedArray<T> outData,
                    int size, Fn binop, bool inPlace) {
   if (size > 0) {
      if (inPlace) {
         if (!data) {
            printf("[CARE] Warning: Invalid arguments to care::inclusive_scan. If inPlace is true, data cannot be nullptr.\n");
            return;
         }
      }
      else {
         if (!outData) {
            printf("[CARE] Warning: Invalid arguments to care::inclusive_scan. If inPlace is false, outData cannot be nullptr.\n");
            return;
         }
         else if (size > 1 && !data) {
            printf("[CARE] Warning: Invalid arguments to care::inclusive_scan. If inPlace is false and size > 1, data cannot be nullptr.\n");
            return;
         }
      }

#if !defined(CHAI_DISABLE_RM)
      // Bounds checking is only available with the resource manager because the ManagedArray::size() is not
      // reliable if it is cast to a different template type.
      const chai::PointerRecord* dataRecord = chai::ArrayManager::getInstance()->getPointerRecord((void *)data.data(care::ExecutionSpace::CPU, false));
      int dataSize = dataRecord->m_size/sizeof(T);
      bool warned = false;

      if (dataSize < size) {
         const char* dataName = CHAICallback::getName(dataRecord);
         if (dataName == nullptr) {
            dataName = "array" ;
         }
         printf("[CARE] Warning: Invalid argument to care::inclusive_scan. Size %d < %d in input %s.\n", dataSize, size, dataName);
         warned = true;
      }

      if (!inPlace) {
         const chai::PointerRecord* outDataRecord = chai::ArrayManager::getInstance()->getPointerRecord((void *)outData.data(care::ExecutionSpace::CPU, false));
         int outDataSize = dataRecord->m_size/sizeof(T);

         if (outDataSize < size) {
            const char* outDataName = CHAICallback::getName(outDataRecord);
            if (outDataName == nullptr) {
               outDataName = "array" ;
            }
            printf("[CARE] Warning: Invalid argument to care::inclusive_scan. Size %d < %d in output %s.\n", outDataSize, size, outDataName);
            warned = true;
         }
      }

      if (warned) {
         umpire::util::backtrace bt;
         umpire::util::backtracer<umpire::util::trace_always>::get_backtrace(bt);
         std::string stack = umpire::util::backtracer<umpire::util::trace_always>::print(bt);
         printf("%s", stack.c_str());
      }
#endif
   }

   CHAIDataGetter<T, Exec> D {};
   ValueType * rawData = D.getRawArrayData(data);

   if (inPlace) {
      RAJA::inclusive_scan_inplace<Exec>(RAJA::make_span(rawData, size),
                                         binop);
   }
   else {
      ValueType * rawOutData = D.getRawArrayData(outData);
      RAJA::inclusive_scan<Exec>(RAJA::make_span(rawData, size),
                                 RAJA::make_span(rawOutData, size),
                                 binop);
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

#if CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<GIDTYPE> data, chai::ManagedArray<GIDTYPE> outData,
                    int size, GIDTYPE val, bool inPlace)
{
   exclusive_scan<GIDTYPE, CARE_SCAN_EXEC, RAJA::operators::plus<GIDTYPE>, GIDTYPE>(data, outData, size,
                                                                                    RAJA::operators::plus<GIDTYPE>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const GIDTYPE> inData, chai::ManagedArray<GIDTYPE> outData,
                    int size, GIDTYPE val)
{ 
    const bool inPlace = false;
    exclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<GIDTYPE> *>(&inData), outData, size, val, inPlace);
}

#endif // CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

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

#if CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<GIDTYPE> data, chai::ManagedArray<GIDTYPE> outData,
                    int size, bool inPlace)
{
   inclusive_scan<GIDTYPE, CARE_SCAN_EXEC, RAJA::operators::plus<GIDTYPE>, GIDTYPE>(data, outData, size,
                                                                                    RAJA::operators::plus<GIDTYPE>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const GIDTYPE> inData, chai::ManagedArray<GIDTYPE> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<GIDTYPE> *>(&inData), outData, size, inPlace);
}

#endif // CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

} // namespace care

#undef CARE_SCAN_EXEC

