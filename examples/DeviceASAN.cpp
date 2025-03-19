//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
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

   // This will write past the end of the array.
   // ASAN should report something like the following:
   //
   // ==4084715==ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device 0 at pc 0x153d3cdd3db8
   // READ of size 4 in workgroup id (3906,0,0)
   //   #0 0x153d3cdd3db8 in main::'lambda0'(int)::operator()(int) const at /g/g17/dayton8/ale3d/github_care_4/examples/DeviceASAN.cpp:26:14
   CARE_STREAM_LOOP(i, 0, size + 1) {
      data[i]++;
   } CARE_STREAM_LOOP_END

   data.free();
   return 0;
}
