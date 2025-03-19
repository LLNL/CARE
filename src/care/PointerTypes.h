//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

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

#if defined(CARE_ENABLE_MANAGED_PTR)
#include "care/managed_ptr.h"
#endif // defined(CARE_ENABLE_MANAGED_PTR)

#define CHAI_DUMMY_TYPE unsigned char
#define CHAI_DUMMY_TYPE_CONST unsigned char const
#define CHAI_DUMMY_PTR_TYPE unsigned char *

namespace care{
   using CARECopyable = chai::CHAICopyable;
}

#endif // !defined(_CARE_CHAI_INTERFACE_H_)

