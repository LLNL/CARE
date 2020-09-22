//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

/**************************************************************************
  Module:  CARE_CHAI_INTERFACE
  Purpose: CARE interface to CHAI.
 ***************************************************************************/

#ifndef _CARE_CHAI_INTERFACE_H_
#define _CARE_CHAI_INTERFACE_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/device_ptr.h"
#include "care/host_device_ptr.h"
#include "care/host_ptr.h"
#include "care/local_host_device_ptr.h"
#include "care/local_ptr.h"

// CHAI headers
#include "chai/config.hpp"

#if defined(CARE_ENABLE_MANAGED_PTR)
#include "chai/managed_ptr.hpp"
#endif // defined(CARE_ENABLE_MANAGED_PTR)

#define CHAI_DUMMY_TYPE unsigned char
#define CHAI_DUMMY_TYPE_CONST unsigned char const
#define CHAI_DUMMY_PTR_TYPE unsigned char *

namespace care{
   using CARECopyable = chai::CHAICopyable;

#if defined(CARE_ENABLE_MANAGED_PTR)

   template <typename T>
   using managed_ptr = chai::managed_ptr<T>;

   template <typename T,
             typename... Args>
   inline managed_ptr<T> make_managed(Args&&... args) {
      return chai::make_managed<T>(std::forward<Args>(args)...);
   }

   template <typename T,
             typename F,
             typename... Args>
   inline managed_ptr<T> make_managed_from_factory(F&& f, Args&&... args) {
      return chai::make_managed_from_factory<T>(std::forward<F>(f), std::forward<Args>(args)...);
   }

#endif // defined(CARE_ENABLE_MANAGED_PTR)
}

#endif // !defined(_CARE_CHAI_INTERFACE_H_)

