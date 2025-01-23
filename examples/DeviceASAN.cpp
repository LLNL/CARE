//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2024, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE library header
#include "care/care.h"

int main(int, char**) {
   const int size = 1000000;
   care::host_device_ptr<int> data(size);

   CARE_STREAM_LOOP(i, 0, size) {
      data[i] = i;
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(i, 0, size + 1) {
      data[i]++;
   } CARE_STREAM_LOOP_END

   data.free();
   return 0;
}
