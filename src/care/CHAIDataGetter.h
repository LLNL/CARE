//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_CHAI_DATA_GETTER_H_
#define _CARE_CHAI_DATA_GETTER_H_

// CARE config header
#include "care/config.h"

#include "care/policies.h"

// Other library headers
#include "chai/ManagedArray.hpp"

#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

/* class for getting a raw pointer from CHAI based on exec policy */
template <typename T, typename Exec>
class CHAIDataGetter {
   public:
      typedef T raw_type;
      T * getRawArrayData(chai::ManagedArray<T> data) {
         data.move(chai::CPU);
         data.registerTouch(chai::CPU);
         return (T*)data.data(chai::CPU);
      }

      const T * getConstRawArrayData(chai::ManagedArray<T> data) {
         data.move(chai::CPU);
         return (const T*)data.data(chai::CPU);
      }

      static const auto ChaiPolicy = chai::CPU;
};

#if defined(CARE_GPUCC)

// Partial specialization of CHAIDataGetter for cuda_exec.
template <typename T>
class CHAIDataGetter<T, RAJADeviceExec> {
   public:
      typedef T raw_type;
      T * getRawArrayData(chai::ManagedArray<T> data) {
         data.move(chai::GPU);
         data.registerTouch(chai::GPU);
         return (T*)data.data(chai::GPU);
      }

      const T * getConstRawArrayData(chai::ManagedArray<T> data) {
         data.move(chai::GPU);
         return (const T*)data.data(chai::GPU);
      }

      static const auto ChaiPolicy = chai::GPU;
};

#if CARE_HAVE_LLNL_GLOBALID

/* specialization for globalID */
template <>
class CHAIDataGetter<globalID, RAJADeviceExec> {
   public:
      typedef GIDTYPE raw_type;
      GIDTYPE * getRawArrayData(chai::ManagedArray<globalID> data) {
         data.move(chai::GPU);
         data.registerTouch(chai::GPU);
         return (GIDTYPE*)data.data(chai::GPU);
      }

      const GIDTYPE * getConstRawArrayData(chai::ManagedArray<globalID> data) {
         data.move(chai::GPU);
         return (GIDTYPE*)data.data(chai::GPU);
      }

      static const auto ChaiPolicy = chai::GPU;
};

#endif // CARE_HAVE_LLNL_GLOBALID

#endif // defined(CARE_GPUCC)

#if CARE_HAVE_LLNL_GLOBALID

/* specialization for globalID */
template <>
class CHAIDataGetter<globalID, RAJA::seq_exec> {
   public:
      typedef GIDTYPE raw_type;
      GIDTYPE * getRawArrayData(chai::ManagedArray<globalID> data) {
         data.move(chai::CPU);
         data.registerTouch(chai::CPU);
         return (GIDTYPE*)data.data(chai::CPU);
      }

      const GIDTYPE * getConstRawArrayData(chai::ManagedArray<globalID> data) {
         data.move(chai::CPU);
         return (GIDTYPE*)data.data(chai::CPU);
      }

      static const auto ChaiPolicy = chai::CPU;
};

#endif // CARE_HAVE_LLNL_GLOBALID

#endif // !defined(_CARE_CHAI_DATA_GETTER_H_)
