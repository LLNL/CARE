//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_ATOMIC_H
#define CARE_ATOMIC_H

#include "care/config.h"
#include "RAJA/RAJA.hpp"

using RAJAAtomic = RAJA::auto_atomic;

#define ATOMIC_ADD(ref, inc)          RAJA::atomicAdd<RAJAAtomic>(&(ref), inc)
#define ATOMIC_SUB(ref, inc)          RAJA::atomicSub<RAJAAtomic>(&(ref), inc)
#define ATOMIC_MIN(ref, val)          RAJA::atomicMin<RAJAAtomic>(&(ref), val)
#define ATOMIC_MAX(ref, val)          RAJA::atomicMax<RAJAAtomic>(&(ref), val)
#define ATOMIC_OR(ref, val)           RAJA::atomicOr<RAJAAtomic>(&(ref), val)
#define ATOMIC_AND(ref, val)          RAJA::atomicAnd<RAJAAtomic>(&(ref), val)
#define ATOMIC_XOR(ref, val)          RAJA::atomicXor<RAJAAtomic>(&(ref), val)
#define ATOMIC_LOAD(ref)              RAJA::atomicLoad<RAJAAtomic>(&(ref))
#define ATOMIC_STORE(ref, val)        RAJA::atomicStore<RAJAAtomic>(&(ref), val)
#define ATOMIC_EXCHANGE(ref, val)     RAJA::atomicExchange<RAJAAtomic>(&(ref), val)
#define ATOMIC_CAS(ref, compare, val) RAJA::atomicCAS<RAJAAtomic>(&(ref), compare, val)

#endif // CARE_ATOMIC_H
