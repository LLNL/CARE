//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/DefaultMacros.h"
#include "care/scan.h"

#define CARE_SCAN_EXEC RAJA::seq_exec
#include "care/scan_impl.h"

#ifdef CARE_PARALLEL_DEVICE
#define CARE_SCAN_EXEC RAJADeviceExec
#include "care/scan_impl.h"
#endif // defined(CARE_PARALEL_DEVICE)

namespace care {

//////////////////////////////////////////////////////////////////////////////
// scan count accessors

void getFinalScanCountFromPinned(chai::ManagedArray<int> scanvar_length,
                                 int& scanCount)
{
   CARE_CHECKED_HOST_KERNEL_WITH_REF_START(scan_loop_check, scanCount) {
      scanCount = scanvar_length[0];
   } CARE_CHECKED_HOST_KERNEL_WITH_REF_END(scan_loop_check)
}

void getFinalScanCount(chai::ManagedArray<int> scanvar,
                       int length, int& scanCount)
{
   scanCount = scanvar.pick(length);
}


void getFinalScanCountFromPinned(chai::ManagedArray<size_t> scanvar_length,
                                 size_t& scanCount)
{
   CARE_CHECKED_HOST_KERNEL_WITH_REF_START(scan_loop_check, scanCount) {
      scanCount = scanvar_length[0];
   } CARE_CHECKED_HOST_KERNEL_WITH_REF_END(scan_loop_check)
}

void getFinalScanCount(chai::ManagedArray<size_t> scanvar,
                       int length, size_t& scanCount)
{
   scanCount = scanvar.pick(length);
}

void getFinalScanCountFromPinned(chai::ManagedArray<int64_t> scanvar_length,
                                 int64_t& scanCount)
{
   CARE_CHECKED_HOST_KERNEL_WITH_REF_START(scan_loop_check, scanCount) {
      scanCount = scanvar_length[0];
   } CARE_CHECKED_HOST_KERNEL_WITH_REF_END(scan_loop_check)
}

void getFinalScanCount(chai::ManagedArray<int64_t> scanvar,
                       int length, int64_t& scanCount)
{
   scanCount = scanvar.pick(length);
}


} // namespace care

