//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_NUMERIC_H_
#define _CARE_NUMERIC_H_

// CARE headers
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"

namespace care {
   /////////////////////////////////////////////////////////////////////////////////
   ///
   /// @author Alan Dayton
   ///
   /// @brief Implements std::iota for care::host_device_ptrs
   ///
   /// @arg[in] policy The execution policy
   /// @arg[out] array The array to operate on
   /// @arg[in] n The size of the array to operate on
   /// @arg[in] value The starting value
   ///
   /////////////////////////////////////////////////////////////////////////////////
   template <typename ExecutionPolicy, typename T>
   inline void iota(ExecutionPolicy policy,
                    care::host_device_ptr<T>& array,
                    size_t n,
                    T value) {
      CARE_LOOP(policy, i, 0, n) {
         array[i] = value + T(i);
      } CARE_LOOP_END
   }
}

#endif // !defined(_CARE_NUMERIC_H_)

