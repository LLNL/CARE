//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-24, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE library header
#include "care/care.h"
#include "care/algorithm.h"

// Std library headers
#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

int main(int, char**) {
   int size = 10;
   care::host_device_ptr<int> data(1);

   CARE_OPENMP_LOOP(i, 0, size) {
      data[0] += i;
   } CARE_OPENMP_LOOP_END

   data.free();
   return 0;
}

