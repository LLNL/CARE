//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-24, Lawrence Livermore National Security, LLC and CARE
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

///
/// Macros that use atomics for a ThreadSanitizer build to avoid false
/// positives, but otherwise do a non-atomic operation (for cases where
/// the order of execution does not matter, such as multiple threads
/// setting the same variable to the same value).
///
/// WARNING: The returned previous value for the TSAN_ONLY_ATOMIC_* macros
///          should generally not be used in a parallel context, since
///          another thread may have modified the value at the given memory
///          location in between the current thread's read and write. If the
///          return value is needed, use the ATOMIC_* macros instead.
///
/// TODO: Evaluate whether the compiler actually does the right thing without
///       atomics and whether using atomics detracts from performance.
///
#if defined(CARE_ENABLE_TSAN_ONLY_ATOMICS)

#define TSAN_ONLY_ATOMIC_ADD(ref, inc)          ATOMIC_ADD(ref, inc)
#define TSAN_ONLY_ATOMIC_SUB(ref, inc)          ATOMIC_SUB(ref, inc)
#define TSAN_ONLY_ATOMIC_MIN(ref, val)          ATOMIC_MIN(ref, val)
#define TSAN_ONLY_ATOMIC_MAX(ref, val)          ATOMIC_MAX(ref, val)
#define TSAN_ONLY_ATOMIC_OR(ref, val)           ATOMIC_OR(ref, val)
#define TSAN_ONLY_ATOMIC_AND(ref, val)          ATOMIC_AND(ref, val)
#define TSAN_ONLY_ATOMIC_XOR(ref, val)          ATOMIC_XOR(ref, val)
#define TSAN_ONLY_ATOMIC_LOAD(ref)              ATOMIC_LOAD(ref)
#define TSAN_ONLY_ATOMIC_STORE(ref, val)        ATOMIC_STORE(ref, val)
#define TSAN_ONLY_ATOMIC_EXCHANGE(ref, val)     ATOMIC_EXCHANGE(ref, val)
#define TSAN_ONLY_ATOMIC_CAS(ref, compare, val) ATOMIC_CAS(ref, compare, val)

#else

using TSANOnlyAtomic = RAJA::seq_atomic;

#define TSAN_ONLY_ATOMIC_ADD(ref, inc)          RAJA::atomicAdd<TSANOnlyAtomic>(&(ref), inc)
#define TSAN_ONLY_ATOMIC_SUB(ref, inc)          RAJA::atomicSub<TSANOnlyAtomic>(&(ref), inc)
#define TSAN_ONLY_ATOMIC_MIN(ref, val)          RAJA::atomicMin<TSANOnlyAtomic>(&(ref), val)
#define TSAN_ONLY_ATOMIC_MAX(ref, val)          RAJA::atomicMax<TSANOnlyAtomic>(&(ref), val)
#define TSAN_ONLY_ATOMIC_OR(ref, val)           RAJA::atomicOr<TSANOnlyAtomic>(&(ref), val)
#define TSAN_ONLY_ATOMIC_AND(ref, val)          RAJA::atomicAnd<TSANOnlyAtomic>(&(ref), val)
#define TSAN_ONLY_ATOMIC_XOR(ref, val)          RAJA::atomicXor<TSANOnlyAtomic>(&(ref), val)
#define TSAN_ONLY_ATOMIC_LOAD(ref)              RAJA::atomicLoad<TSANOnlyAtomic>(&(ref))
#define TSAN_ONLY_ATOMIC_STORE(ref, val)        RAJA::atomicStore<TSANOnlyAtomic>(&(ref), val)
#define TSAN_ONLY_ATOMIC_EXCHANGE(ref, val)     RAJA::atomicExchange<TSANOnlyAtomic>(&(ref), val)
#define TSAN_ONLY_ATOMIC_CAS(ref, compare, val) RAJA::atomicCAS<TSANOnlyAtomic>(&(ref), compare, val)

#endif

#endif // CARE_ATOMIC_H
