//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-24, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/DefaultMacros.h"
#include "care/scan.h"

#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

#define CARE_SCAN_EXEC RAJA::seq_exec
#include "care/scan_impl.h"

#ifdef CARE_PARALLEL_DEVICE
#define CARE_SCAN_EXEC RAJADeviceExec
#include "care/scan_impl.h"
#endif // defined(CARE_PARALEL_DEVICE)

namespace care {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// scan count accessors

void getFinalScanCountFromPinned(chai::ManagedArray<int> scanvar_length, int& scanCount)
{
   CARE_CHECKED_HOST_KERNEL_WITH_REF_START(scan_loop_check, scanCount) {
      scanCount = scanvar_length[0];
   } CARE_CHECKED_HOST_KERNEL_WITH_REF_END(scan_loop_check)
}

void getFinalScanCount(chai::ManagedArray<int> scanvar, int length, int& scanCount)
{
   scanCount = scanvar.pick(length);
}

#if CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

void getFinalScanCountFromPinned(chai::ManagedArray<GIDTYPE> scanvar_length, GIDTYPE& scanCount)
{
   CARE_CHECKED_HOST_KERNEL_WITH_REF_START(scan_loop_check, scanCount) {
      scanCount = scanvar_length[0];
   } CARE_CHECKED_HOST_KERNEL_WITH_REF_END(scan_loop_check)
}

void getFinalScanCount(chai::ManagedArray<GIDTYPE> scanvar, int length, GIDTYPE& scanCount)
{
   scanCount = scanvar.pick(length);
}

#endif // CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

} // namespace care

