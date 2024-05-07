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
#include "care/config.h"
#include "care/care.h"
#include "care/host_device_ptr.h"
#include "care/PointerTypes.h"

template <class T, int N>
struct StackArray {
   CARE_HOST_DEVICE constexpr T& operator[](int i) noexcept {
      return elements[i];
   }

   CARE_HOST_DEVICE constexpr const T& operator[](int i) const noexcept {
      return elements[i];
   }

   T elements[N];
};

int main(int, char**) {
   // Array containing host_device_ptr
#if WANT_EXIT_CODE_139
   StackArray<care::host_device_ptr<int>, 1> a{nullptr};
#else
   StackArray<care::host_device_ptr<int>, 1> a;
#endif

   for (int i = 0; i < 1; ++i) {
      a[i] = care::host_device_ptr<int>(10);
   }

   CARE_STREAM_LOOP(i, 0, 10) {
      a[0][i] = i;
   } CARE_STREAM_LOOP_END

   a[0].free();

   // Kernel afterwards
   StackArray<int, 2> b = {3, 7};
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      if (b[1] != 7) {
         passed.min(false);
      }
   } CARE_REDUCE_LOOP_END

   if ((bool) passed) {
      std::cout << "Kernel after\n";
   }

   return 0;
}

