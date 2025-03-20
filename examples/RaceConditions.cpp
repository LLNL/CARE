//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
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
   const int size = 10;
   care::host_device_ptr<int> data(size);

   CARE_OPENMP_LOOP(i, 0, size) {
      data[0] += i; // clang tsan catches this
   } CARE_OPENMP_LOOP_END

   CARE_OPENMP_LOOP(i, 0, size - 1) {
      data[i] = data[i + 1]; // clang tsan catches this
   } CARE_OPENMP_LOOP_END

   CARE_OPENMP_LOOP(i, 0, size) {
      data[0] = 1; // clang tsan does not report
   } CARE_OPENMP_LOOP_END

   data.free();
   return 0;
}

