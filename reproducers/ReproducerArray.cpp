//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

//
// care::array<care::host_device_ptr<T>, N> crashes at run time with the
// following error:
//
// malloc_consolidate(): unaligned fastbin chunk detected
// flux-job: task(s) exited with exit code 134
//
// If WANT_EXIT_CODE_139 is defined, then the program crashes with this error:
//
// flux-job: task(s) exited with exit code 139
//

// CARE library headers
#include "chai/ManagedArray.hpp"
#include "RAJA/RAJA.hpp"

template <class T, int N>
struct StackArray {
   __host__ __device__ constexpr T& operator[](int i) noexcept {
      return elements[i];
   }

   __host__ __device__ constexpr const T& operator[](int i) const noexcept {
      return elements[i];
   }

   T elements[N];
};

int main(int, char**) {
   // Array containing host_device_ptr
#if WANT_EXIT_CODE_139
   StackArray<chai::ManagedArray<int>, 1> a{chai::ManagedArray<int>{}};
#else
   StackArray<chai::ManagedArray<int>, 1> a;
#endif

   for (int i = 0; i < 1; ++i) {
      a[i] = chai::ManagedArray<int>(10);
   }

   RAJA::forall<RAJA::hip_exec<256, true>>(
      RAJA::RangeSegment(0, 10),
      [=] __device__ (int i) { a[0][i] = i; });

   a[0].free();

   // Kernel afterwards
   StackArray<int, 2> b = {3, 7};

   RAJA::forall<RAJA::hip_exec<256, true>>(
      RAJA::RangeSegment(0, 2),
      [=] __device__ (int i) { static_cast<void>(b[i]); });

   return 0;
}

