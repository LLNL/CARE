//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_CHAI_DATA_GETTER_H_
#define _CARE_CHAI_DATA_GETTER_H_

// CARE config header
#include "care/config.h"

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
         return (T*)data.getPointer(chai::CPU);
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
         return (T*)data.getPointer(chai::GPU);
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
         return (GIDTYPE*)data.getPointer(chai::GPU);
      }

      static const auto ChaiPolicy = chai::GPU;
};

#endif // CARE_HAVE_LLNL_GLOBALID

#endif // defined(CARE_GPUCC)

#endif // !defined(_CARE_CHAI_DATA_GETTER_H_)
