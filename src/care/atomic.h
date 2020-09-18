#ifndef CARE_ATOMIC_H
#define CARE_ATOMIC_H

#include "RAJA/RAJA.hpp"

using RAJAAtomic = RAJA::auto_atomic;

#define ATOMIC_ADD(ref, inc) RAJA::atomicAdd<RAJAAtomic>(&(ref), inc)
#define ATOMIC_MIN(ref, val) RAJA::atomicMin<RAJAAtomic>(&(ref), val)
#define ATOMIC_MAX(ref, val) RAJA::atomicMax<RAJAAtomic>(&(ref), val)
#define ATOMIC_OR(ref, val)  RAJA::atomicOr<RAJAAtomic>(&(ref), val)
#define ATOMIC_AND(ref, val) RAJA::atomicAnd<RAJAAtomic>(&(ref), val)
#define ATOMIC_XOR(ref, val) RAJA::atomicXor<RAJAAtomic>(&(ref), val)

#endif // CARE_ATOMIC_H
