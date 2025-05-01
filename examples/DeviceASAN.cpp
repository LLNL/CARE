//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// CARE library header
#include "care/care.h"
#include "care/algorithm.h"

int main(int, char**) {
   int size = 1000000;
   care::host_device_ptr<int> data(size);

   CARE_STREAM_LOOP(i, 0, size) {
      data[i] = i;
   } CARE_STREAM_LOOP_END

   care::sort_uniq<int>(RAJADeviceExec{}, &data, &size);

   CARE_STREAM_LOOP(i, 0, size + 1) {
      data[i]++;
   } CARE_STREAM_LOOP_END

   data.free();
   return 0;
}
