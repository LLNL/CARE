//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

//
// care::array<care::host_device_ptr<T>, N> crashes at run time.
//

// CARE library headers
#include "care/config.h"
#include "care/care.h"
#include "care/host_device_ptr.h"
#include "care/PointerTypes.h"
#include "care/array.h"

int main(int, char**) {
   care::array<int, 2> a{{3, 7}};
   RAJAReduceMin<bool> passed{true};

   CARE_REDUCE_LOOP(i, 0, 1) {
      if (a.front() != 3) {
         passed.min(false);
         return;
      }
   } CARE_REDUCE_LOOP_END

   if ((bool) passed) {
      std::cout << "Kernel before\n";
   }

   // Array containing host_device_ptr
   care::array<care::host_device_ptr<int>, 1> b;

   for (int i = 0; i < 1; ++i) {
      b[i] = care::host_device_ptr<int>(10);
   }

   CARE_STREAM_LOOP(i, 0, 10) {
      b[0][i] = i;
   } CARE_STREAM_LOOP_END

   b[0].free();

   // Kernel afterwards
   passed.reset(true);

   CARE_REDUCE_LOOP(i, 0, 1) {
      if (a.back() != 7) {
         passed.min(false);
      }
   } CARE_REDUCE_LOOP_END

   if ((bool) passed) {
      std::cout << "Kernel after\n";
   }

   return 0;
}

