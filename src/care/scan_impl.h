//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// This header should be included once for each execution policy, defined by
// CARE_SCAN_EXEC.  This header should only be needed within CARE proper and
// should not be included by the host code.

#ifndef CARE_SCAN_EXEC
#error "CARE_SCAN_EXEC must be defined"
#endif

#include "care/CHAICallback.h"
#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/scan.h"

#include "umpire/util/backtrace.hpp"

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

      if (size > 1) {
         bool warned = false;

         if (data.size() < size) {
            printf("[CARE] Warning: Invalid arguments to care::exclusive_scan. Size of input array (%zu) is less than given size (%d).\n", data.size(), size);
            warned = true;
         }

         if (!inPlace && outData.size() < size) {
            printf("[CARE] Warning: Invalid arguments to care::exclusive_scan. Size of output array (%zu) is less than given size (%d).\n", outData.size(), size);
            warned = true;
         }

         if (warned) {
            umpire::util::backtrace bt;
            umpire::util::backtracer<umpire::util::trace_always>::get_backtrace(bt);
            std::string stack = umpire::util::backtracer<umpire::util::trace_always>::print(bt);
            printf("%s", stack.c_str());
         }
      }

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

         if (inPlace) {
            ValueType * rawData = D.getRawArrayData(data);
            RAJA::exclusive_scan_inplace<Exec>(RAJA::make_span(rawData, size),
                                               binop, val);
         }
         else {
            const ValueType * rawData = D.getConstRawArrayData(data);
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

      bool warned = false;

      if (data.size() < size) {
         printf("[CARE] Warning: Invalid arguments to care::inclusive_scan. Size of input array (%zu) is less than given size (%d).\n", data.size(), size);
         warned = true;
      }

      if (!inPlace && outData.size() < size) {
         printf("[CARE] Warning: Invalid arguments to care::inclusive_scan. Size of output array (%zu) is less than given size (%d).\n", outData.size(), size);
         warned = true;
      }

      if (warned) {
         umpire::util::backtrace bt;
         umpire::util::backtracer<umpire::util::trace_always>::get_backtrace(bt);
         std::string stack = umpire::util::backtracer<umpire::util::trace_always>::print(bt);
         printf("%s", stack.c_str());
      }
   }

   CHAIDataGetter<T, Exec> D {};

   if (inPlace) {
      ValueType * rawData = D.getRawArrayData(data);
      RAJA::inclusive_scan_inplace<Exec>(RAJA::make_span(rawData, size),
                                         binop);
   }
   else {
      const ValueType * rawData = D.getConstRawArrayData(data);
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

void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<unsigned int> data, chai::ManagedArray<unsigned int> outData,
                    int size, unsigned int val, bool inPlace)
{
   exclusive_scan<unsigned int, CARE_SCAN_EXEC, RAJA::operators::plus<unsigned int>>(data, outData, size,
                                                                   RAJA::operators::plus<unsigned int>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const unsigned int> inData, chai::ManagedArray<unsigned int> outData,
                    int size, unsigned int val)
{
    const bool inPlace = false;
    exclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<unsigned int> *>(&inData), outData, size, val, inPlace);
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


void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<size_t> data, chai::ManagedArray<size_t> outData,
                    int size, size_t val, bool inPlace)
{
   exclusive_scan<size_t, CARE_SCAN_EXEC, RAJA::operators::plus<size_t>, size_t>(data, outData, size,
                                                                                    RAJA::operators::plus<size_t>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const size_t> inData, chai::ManagedArray<size_t> outData,
                    int size, size_t val)
{
    const bool inPlace = false;
    exclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<size_t> *>(&inData), outData, size, val, inPlace);
}

void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<int64_t> data, chai::ManagedArray<int64_t> outData,
                    int size, int64_t val, bool inPlace)
{
   exclusive_scan<int64_t, CARE_SCAN_EXEC, RAJA::operators::plus<int64_t>, int64_t>(data, outData, size,
                                                                                    RAJA::operators::plus<int64_t>{}, val, inPlace) ;
}

// typesafe wrapper for out of place scan
void exclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const int64_t> inData, chai::ManagedArray<int64_t> outData,
                    int size, int64_t val)
{
    const bool inPlace = false;
    exclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<int64_t> *>(&inData), outData, size, val, inPlace);
}


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

void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<unsigned int> data, chai::ManagedArray<unsigned int> outData,
                    int size, bool inPlace)
{
   inclusive_scan<unsigned int, CARE_SCAN_EXEC, RAJA::operators::plus<unsigned int>>(data, outData, size,
                                                                   RAJA::operators::plus<unsigned int>{}, inPlace) ;
}

// typesafe wrapper for out of place scan
void inclusive_scan(CARE_SCAN_EXEC, chai::ManagedArray<const unsigned int> inData, chai::ManagedArray<unsigned int> outData,
                    int size)
{
    const bool inPlace = false;
    inclusive_scan(CARE_SCAN_EXEC{}, *reinterpret_cast<chai::ManagedArray<unsigned int> *>(&inData), outData, size, inPlace);
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

