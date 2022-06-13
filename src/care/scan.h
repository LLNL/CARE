//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_SCAN_H_
#define _CARE_SCAN_H_

// CARE config header
#include "care/config.h"
#include "care/DefaultMacros.h"

// Other library headers
#include "chai/ManagedArray.hpp"
#include "RAJA/RAJA.hpp"

#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

#if CARE_ENABLE_PINNED_MEMORY_FOR_SCANS
#define CARE_SCANVARLENGTHNAME_SPACE chai::PINNED
#else
#define CARE_SCANVARLENGTHNAME_SPACE chai::GPU
#endif // CARE_ENABLE_PINNED_MEMORY_FOR_SCANS

namespace care {

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// exclusive scan functionality

CARE_DLL_API
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, int val, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size, int val);

#ifdef CARE_GPUCC

CARE_DLL_API
void exclusive_scan(RAJADeviceExec, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, int val, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void exclusive_scan(RAJADeviceExec, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size, int val);

#endif // defined(CARE_GPUCC)

CARE_DLL_API
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<float> data, chai::ManagedArray<float> outData,
                    int size, float val, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<const float> inData, chai::ManagedArray<float> outData,
                    int size, float val);

#ifdef CARE_GPUCC

CARE_DLL_API
void exclusive_scan(RAJADeviceExec, chai::ManagedArray<float> data, chai::ManagedArray<float> outData,
                    int size, float val, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void exclusive_scan(RAJADeviceExec, chai::ManagedArray<const float> inData, chai::ManagedArray<float> outData,
                    int size, float val);

#endif // defined(CARE_GPUCC)

CARE_DLL_API
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<double> data, chai::ManagedArray<double> outData,
                    int size, double val, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<const double> inData, chai::ManagedArray<double> outData,
                    int size, double val);

#ifdef CARE_GPUCC

CARE_DLL_API
void exclusive_scan(RAJADeviceExec, chai::ManagedArray<double> data, chai::ManagedArray<double> outData,
                    int size, double val, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void exclusive_scan(RAJADeviceExec, chai::ManagedArray<const double> inData, chai::ManagedArray<double> outData,
                    int size, double val);

#endif // defined(CARE_GPUCC)

#if CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

CARE_DLL_API
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<GIDTYPE> data, chai::ManagedArray<GIDTYPE> outData,
                    int size, GIDTYPE val, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void exclusive_scan(RAJA::seq_exec, chai::ManagedArray<const GIDTYPE> inData, chai::ManagedArray<GIDTYPE> outData,
                    int size, GIDTYPE val);

#ifdef CARE_GPUCC

CARE_DLL_API
void exclusive_scan(RAJADeviceExec, chai::ManagedArray<GIDTYPE> data, chai::ManagedArray<GIDTYPE> outData,
                    int size, GIDTYPE val, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void exclusive_scan(RAJADeviceExec, chai::ManagedArray<const GIDTYPE> inData, chai::ManagedArray<GIDTYPE> outData,
                    int size, GIDTYPE val);

#endif // defined(CARE_GPUCC)

#endif // CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// inclusive scan functionality

CARE_DLL_API
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size) ;

#ifdef CARE_GPUCC

CARE_DLL_API
void inclusive_scan(RAJADeviceExec, chai::ManagedArray<int> data, chai::ManagedArray<int> outData,
                    int size, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void inclusive_scan(RAJADeviceExec, chai::ManagedArray<const int> inData, chai::ManagedArray<int> outData,
                    int size) ;

#endif // defined(CARE_GPUCC)

CARE_DLL_API
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<float> data, chai::ManagedArray<float> outData,
                    int size, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<const float> inData, chai::ManagedArray<float> outData,
                    int size) ;

#ifdef CARE_GPUCC

CARE_DLL_API
void inclusive_scan(RAJADeviceExec, chai::ManagedArray<float> data, chai::ManagedArray<float> outData,
                    int size, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void inclusive_scan(RAJADeviceExec, chai::ManagedArray<const float> inData, chai::ManagedArray<float> outData,
                    int size) ;

#endif // defined(CARE_GPUCC)

CARE_DLL_API
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<double> data, chai::ManagedArray<double> outData,
                    int size, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<const double> inData, chai::ManagedArray<double> outData,
                    int size) ;

#ifdef CARE_GPUCC

CARE_DLL_API
void inclusive_scan(RAJADeviceExec, chai::ManagedArray<double> data, chai::ManagedArray<double> outData,
                    int size, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void inclusive_scan(RAJADeviceExec, chai::ManagedArray<const double> inData, chai::ManagedArray<double> outData,
                    int size) ;

#endif // defined(CARE_GPUCC)

#if CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

CARE_DLL_API
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<GIDTYPE> data, chai::ManagedArray<GIDTYPE> outData,
                    int size, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void inclusive_scan(RAJA::seq_exec, chai::ManagedArray<const GIDTYPE> inData, chai::ManagedArray<GIDTYPE> outData,
                    int size) ;

#ifdef CARE_GPUCC

CARE_DLL_API
void inclusive_scan(RAJADeviceExec, chai::ManagedArray<GIDTYPE> data, chai::ManagedArray<GIDTYPE> outData,
                    int size, bool inPlace);
// typesafe wrapper for out of place scan
CARE_DLL_API
void inclusive_scan(RAJADeviceExec, chai::ManagedArray<const GIDTYPE> inData, chai::ManagedArray<GIDTYPE> outData,
                    int size) ;

#endif // defined(CARE_GPUCC)

#endif // CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// scan count accessors

CARE_DLL_API
void getFinalScanCountFromPinned(chai::ManagedArray<int> scanvar_length, int& scanCount) ;
CARE_DLL_API
void getFinalScanCount(chai::ManagedArray<int> scanvar, int length, int& scanCount) ;

#if CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

CARE_DLL_API
void getFinalScanCountFromPinned(chai::ManagedArray<GIDTYPE> scanvar_length, GIDTYPE& scanCount) ;
CARE_DLL_API
void getFinalScanCount(chai::ManagedArray<GIDTYPE> scanvar, int length, GIDTYPE& scanCount) ;

#endif // CARE_HAVE_LLNL_GLOBALID && GLOBALID_IS_64BIT

} // namespace care

// CPU version of scan idiom. Designed to look like we're doing a scan, but
// does the CPU efficient all in one pass idiom
#define SCAN_LOOP_P(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      int SCANVARNAME(SCANINDX) = SCANINDX_OFFSET; \
      CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_START(INDX, START, END, scan_loop_check, SCANVARNAME(SCANINDX)) { \
         if (EXPR) { \
            const int SCANINDX = SCANVARNAME(SCANINDX)++;


#define SCAN_LOOP_P_END(END, SCANINDX, SCANLENGTH) } \
   } CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_END(scan_loop_check) \
   SCANLENGTH = SCANVARNAME(SCANINDX); \
   }

#define SCANVARNAME(INDXNAME) INDXNAME ## _scanvar
#define SCANVARLENGTHNAME(INDXNAME) INDXNAME ## _scanvar_length
#define SCANVARENDNAME(SCANVAR) SCANVAR ## _end
#define SCANVARSTARTNAME(SCANVAR) SCANVAR ## _start

#if !defined(CARE_NEVER_USE_RAJA_PARALLEL_SCAN)
// scan var is an managed array of ints

using ScanVar = chai::ManagedArray<int>;

#if CARE_HAVE_LLNL_GLOBALID

using ScanVarGID = chai::ManagedArray<GIDTYPE>;

#endif // CARE_HAVE_LLNL_GLOBALID

// initialize field to be scanned with the expression given, then perform the scan in place
#define SCAN_LOOP_INIT(INDX, START, END, SCANVAR, SCANVARLENGTH, SCANVAR_OFFSET, EXPR) \
   if (END - START > 0) { \
      int const SCANVARENDNAME(SCANVAR) = END; \
      CARE_CHECKED_PARALLEL_LOOP_START(INDX, START, END+1, scan_loop_init_check) { \
         SCANVAR[INDX-START] = (INDX != SCANVARENDNAME(SCANVAR)) && (EXPR) ; \
      } CARE_CHECKED_PARALLEL_LOOP_END(scan_loop_init_check) \
      care::exclusive_scan(RAJAExec{}, SCANVAR, nullptr, END-START+1, SCANVAR_OFFSET, true); \
   } else { \
      CARE_CHECKED_SEQUENTIAL_LOOP_START(INDX, 0, 1, scan_loop_init_check) { \
         SCANVAR[INDX] = SCANVAR_OFFSET; \
         SCANVARLENGTH[0] = SCANVAR_OFFSET; \
      } CARE_CHECKED_SEQUENTIAL_LOOP_END(scan_loop_init_check) \
   }

#if CARE_HAVE_LLNL_GLOBALID

#define SCAN_LOOP_GID_INIT(INDX, START, END, SCANVAR, SCANVARLENGTH, SCANVAR_OFFSET, EXPR) \
   if (END - START > 0) { \
      int const SCANVARENDNAME(SCANVAR) = END; \
      CARE_CHECKED_PARALLEL_LOOP_START(INDX, START, END+1, scan_loop_gid_init_check) { \
         SCANVAR[INDX-START] = (INDX != SCANVARENDNAME(SCANVAR)) && (EXPR) ; \
      } CARE_CHECKED_PARALLEL_LOOP_END(scan_loop_gid_init_check) \
      care::exclusive_scan(RAJAExec{}, SCANVAR, nullptr, END-START+1, SCANVAR_OFFSET.Value(), true); \
   } else { \
      CARE_CHECKED_SEQUENTIAL_LOOP_START(INDX, 0, 1, scan_loop_gid_init_check) { \
         SCANVAR[INDX] = SCANVAR_OFFSET.Value(); \
         SCANVARLENGTH[0] = SCANVAR_OFFSET.Value(); \
      } CARE_CHECKED_SEQUENTIAL_LOOP_END(scan_loop_gid_init_check) \
   }

#endif // CARE_HAVE_LLNL_GLOBALID

#define MANAGED_PTR_SCAN_LOOP_INIT(INDX, START, END, SCANVAR, SCANVARLENGTH, SCANVAR_OFFSET, EXPR) \
   if (END - START > 0) { \
      int const SCANVARENDNAME(SCANVAR) = END; \
      CARE_CHECKED_MANAGED_PTR_LOOP_START(INDX, START, END+1, managed_ptr_scan_loop_init_check) { \
         SCANVAR[INDX-START] = (INDX != SCANVARENDNAME(SCANVAR)) && (EXPR) ; \
      } CARE_CHECKED_MANAGED_PTR_LOOP_END(managed_ptr_scan_loop_init_check) \
      care::exclusive_scan(RAJAExec{}, SCANVAR, nullptr, END-START+1, SCANVAR_OFFSET, true); \
   } else { \
      CARE_CHECKED_SEQUENTIAL_LOOP_START(INDX, 0, 1, managed_ptr_scan_loop_init_check) { \
         SCANVAR[INDX] = SCANVAR_OFFSET; \
         SCANVARLENGTH[0] = SCANVAR_OFFSET; \
      } CARE_CHECKED_SEQUENTIAL_LOOP_END(managed_ptr_scan_loop_init_check) \
   }

// grab the number of elements that met the scan criteria, place it in SCANLENGTH
#define SCAN_LOOP_FINAL(END, SCANVARLENGTH, SCANCOUNT) care::getFinalScanCountFromPinned(SCANVARLENGTH, SCANCOUNT);

#define SCAN_LOOP(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      int const SCANVARSTARTNAME(SCANINDX) = START; \
      ScanVar SCANVARNAME(SCANINDX)(END-START+1); \
      ScanVar SCANVARLENGTHNAME(SCANINDX)(1, CARE_SCANVARLENGTHNAME_SPACE); \
      SCAN_LOOP_INIT(INDX, SCANVARSTARTNAME(SCANINDX), END, SCANVARNAME(SCANINDX), SCANVARLENGTHNAME(SCANINDX), SCANINDX_OFFSET, EXPR); \
      int const SCANVARENDNAME(SCANINDX) = END; \
      CARE_CHECKED_PARALLEL_LOOP_START(INDX, START, END, scan_loop_check) { \
         if (INDX == SCANVARENDNAME(SCANINDX) -1) { \
            SCANVARLENGTHNAME(SCANINDX)[0] = SCANVARNAME(SCANINDX)[SCANVARENDNAME(SCANINDX)-START]; \
         } \
         const int SCANINDX = SCANVARNAME(SCANINDX)[INDX-SCANVARSTARTNAME(SCANINDX)]; \
         if (SCANINDX != SCANVARNAME(SCANINDX)[INDX-SCANVARSTARTNAME(SCANINDX)+1]) {

#define SCAN_LOOP_END(END, SCANINDX, SCANLENGTH) } \
   } CARE_CHECKED_PARALLEL_LOOP_END(scan_loop_check) \
   SCAN_LOOP_FINAL(END, SCANVARLENGTHNAME(SCANINDX), SCANLENGTH) \
   SCANVARNAME(SCANINDX).free(); \
   SCANVARLENGTHNAME(SCANINDX).free(); \
   }

#if CARE_HAVE_LLNL_GLOBALID

#define SCAN_LOOP_GID(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      int const SCANVARSTARTNAME(SCANINDX) = START; \
      ScanVarGID SCANVARNAME(SCANINDX)(END-START+1); \
      ScanVarGID SCANVARLENGTHNAME(SCANINDX)(1, CARE_SCANVARLENGTHNAME_SPACE); \
      SCAN_LOOP_GID_INIT(INDX, SCANVARSTARTNAME(SCANINDX), END, SCANVARNAME(SCANINDX), SCANVARLENGTHNAME(SCANINDX), SCANINDX_OFFSET, EXPR); \
      int const SCANVARENDNAME(SCANINDX) = END; \
      CARE_CHECKED_PARALLEL_LOOP_START(INDX, START, END, scan_loop_gid_check) { \
         if (INDX == SCANVARENDNAME(SCANINDX)-1) { \
            SCANVARLENGTHNAME(SCANINDX)[0] = SCANVARNAME(SCANINDX)[SCANVARENDNAME(SCANINDX)-START]; \
         } \
         const globalID SCANINDX = globalID(SCANVARNAME(SCANINDX)[INDX-SCANVARSTARTNAME(SCANINDX)]); \
         if (SCANINDX != globalID(SCANVARNAME(SCANINDX)[INDX-SCANVARSTARTNAME(SCANINDX)+1])) {

#define SCAN_LOOP_GID_END(END, SCANINDX, SCANLENGTH) } \
   } CARE_CHECKED_PARALLEL_LOOP_END(scan_loop_gid_check) \
   SCAN_LOOP_FINAL(END, SCANVARLENGTHNAME(SCANINDX), SCANLENGTH.Ref()) \
   SCANVARNAME(SCANINDX).free(); \
   SCANVARLENGTHNAME(SCANINDX).free(); \
   }

#endif // CARE_HAVE_LLNL_GLOBALID

#define SCAN_EVERYWHERE_LOOP(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      ScanVar SCANVARNAME(SCANINDX)(END-START+1); \
      ScanVar SCANVARLENGTHNAME(SCANINDX)(1, CARE_SCANVARLENGTHNAME_SPACE); \
      SCAN_LOOP_INIT(INDX, START, END, SCANVARNAME(SCANINDX), SCANVARLENGTHNAME(SCANINDX), SCANINDX_OFFSET, EXPR); \
      int const SCANVARENDNAME(SCANINDX) = END; \
      CARE_CHECKED_PARALLEL_LOOP_START(INDX, START, END, scaneverywhere_loop_check) { \
         if (INDX == SCANVARENDNAME(SCANINDX)-1) { \
            SCANVARLENGTHNAME(SCANINDX)[0] = SCANVARNAME(SCANINDX)[SCANVARENDNAME(SCANINDX)-START]; \
         } \
         const int SCANINDX = SCANVARNAME(SCANINDX)[INDX-START];


#define SCAN_EVERYWHERE_LOOP_END(END, SCANINDX, SCANLENGTH) \
   } CARE_CHECKED_PARALLEL_LOOP_END(scaneverywhere_loop_check) \
   SCAN_LOOP_FINAL(END, SCANVARLENGTHNAME(SCANINDX), SCANLENGTH) \
   SCANVARNAME(SCANINDX).free(); \
   SCANVARLENGTHNAME(SCANINDX).free(); \
   }

#define SCAN_EVERYWHERE_REDUCE_LOOP(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      ScanVar SCANVARNAME(SCANINDX)(END-START+1); \
      ScanVar SCANVARLENGTHNAME(SCANINDX)(1, CARE_SCANVARLENGTHNAME_SPACE); \
      SCAN_LOOP_INIT(INDX, START, END, SCANVARNAME(SCANINDX), SCANVARLENGTHNAME(SCANINDX), SCANINDX_OFFSET, EXPR); \
      int const SCANVARENDNAME(SCANINDX) = END; \
      CARE_CHECKED_PARALLEL_LOOP_START(INDX, START, END, scaneverywhere_reduce_loop_check) { \
         if (INDX == SCANVARENDNAME(SCANINDX)-1) { \
            SCANVARLENGTHNAME(SCANINDX)[0] = SCANVARNAME(SCANINDX)[SCANVARENDNAME(SCANINDX)-START]; \
         } \
         const int SCANINDX = SCANVARNAME(SCANINDX)[INDX-START];

#define SCAN_EVERYWHERE_REDUCE_LOOP_END(END, SCANINDX, SCANLENGTH) \
   } CARE_CHECKED_PARALLEL_LOOP_END(scaneverywhere_reduce_loop_check) \
   SCAN_LOOP_FINAL(END, SCANVARLENGTHNAME(SCANINDX), SCANLENGTH) \
   SCANVARNAME(SCANINDX).free(); \
   SCANVARLENGTHNAME(SCANINDX).free(); \
   }

#define SCAN_COUNTS_TO_OFFSETS_LOOP(INDX, START, END, SCANVAR) \
   { \
      CARE_CHECKED_PARALLEL_LOOP_START(INDX, START, END, scan_counts_to_offsets_loop_check) { \

#define SCAN_COUNTS_TO_OFFSETS_LOOP_END(INDX, LENGTH, SCANVAR) \
      } CARE_CHECKED_PARALLEL_LOOP_END(scan_counts_to_offsets_loop_check) \
      care::exclusive_scan(RAJAExec{}, SCANVAR, nullptr, LENGTH, 0, true); \
   }
         
#define MANAGED_PTR_SCAN_LOOP(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      int const SCANVARSTARTNAME(SCANINDX) = START; \
      ScanVar SCANVARNAME(SCANINDX)(END-START+1); \
      ScanVar SCANVARLENGTHNAME(SCANINDX)(1, CARE_SCANVARLENGTHNAME_SPACE); \
      MANAGED_PTR_SCAN_LOOP_INIT(INDX, SCANVARSTARTNAME(SCANINDX), END, SCANVARNAME(SCANINDX), SCANVARLENGTHNAME(SCANINDX), SCANINDX_OFFSET, EXPR); \
      int const SCANVARENDNAME(SCANINDX) = END; \
      CARE_CHECKED_MANAGED_PTR_LOOP_START(INDX, START, END, managed_ptr_scan_loop_check) { \
         if (INDX == SCANVARENDNAME(SCANINDX) -1) { \
            SCANVARLENGTHNAME(SCANINDX)[0] = SCANVARNAME(SCANINDX)[SCANVARENDNAME(SCANINDX)-START]; \
         } \
         const int SCANINDX = SCANVARNAME(SCANINDX)[INDX-SCANVARSTARTNAME(SCANINDX)]; \
         if (SCANINDX != SCANVARNAME(SCANINDX)[INDX-SCANVARSTARTNAME(SCANINDX)+1]) {

#define MANAGED_PTR_SCAN_LOOP_END(END, SCANINDX, SCANLENGTH) } \
   } CARE_CHECKED_MANAGED_PTR_LOOP_END(managed_ptr_scan_loop_check) \
   SCAN_LOOP_FINAL(END, SCANVARLENGTHNAME(SCANINDX), SCANLENGTH) \
   SCANVARNAME(SCANINDX).free(); \
   SCANVARLENGTHNAME(SCANINDX).free(); \
   }

#else // !defined(CARE_NEVER_USE_RAJA_PARALLEL_SCAN)

// CPU version of scan idiom. Designed to look like we're doing a scan, but
// does the CPU efficient all in one pass idiom
#define SCAN_LOOP(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR)  \
   SCAN_LOOP_P(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR)

#define SCAN_LOOP_END(END, SCANINDX, SCANLENGTH) \
   SCAN_LOOP_P_END(END, SCANINDX, SCANLENGTH)

#if CARE_HAVE_LLNL_GLOBALID

#define SCAN_LOOP_GID(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      globalID SCANVARNAME(SCANINDX) = SCANINDX_OFFSET; \
      CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_START(INDX, START, END, scan_loop_gid_check, SCANVARNAME(SCANINDX)) { \
         if (EXPR) { \
            const globalID SCANINDX = SCANVARNAME(SCANINDX)++;

#define SCAN_LOOP_GID_END(END, SCANINDX, SCANLENGTH) } \
   } CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_END(scan_loop_gid_check) \
   SCANLENGTH = SCANVARNAME(SCANINDX); \
   }

#endif // CARE_HAVE_LLNL_GLOBALID

#define SCAN_EVERYWHERE_LOOP(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      int SCANVARNAME(SCANINDX) = SCANINDX_OFFSET; \
      CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_START(INDX, START, END, scaneverywhere_loop_check, SCANVARNAME(SCANINDX)) { \
         const int SCANINDX = (EXPR) ?  SCANVARNAME(SCANINDX)++ : SCANVARNAME(SCANINDX);

#define SCAN_EVERYWHERE_LOOP_END(END, SCANINDX, SCANLENGTH) \
   } CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_END(scaneverywhere_loop_check) \
   SCANLENGTH = SCANVARNAME(SCANINDX); \
   }

#define SCAN_EVERYWHERE_REDUCE_LOOP(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      int SCANVARNAME(SCANINDX) = SCANINDX_OFFSET; \
      CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_START(INDX, START, END, scaneverywhere_reduce_loop_check, SCANVARNAME(SCANINDX)) { \
         const int SCANINDX = (EXPR) ?  SCANVARNAME(SCANINDX)++ : SCANVARNAME(SCANINDX);

#define SCAN_EVERYWHERE_REDUCE_LOOP_END(END, SCANINDX, SCANLENGTH) \
   } CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_END(scaneverywhere_reduce_loop_check) \
   SCANLENGTH = SCANVARNAME(SCANINDX); \
   }

#define SCAN_COUNTS_TO_OFFSETS_LOOP(INDX, START, END, SCANVAR) \
   { \
      CARE_CHECKED_SEQUENTIAL_LOOP_START(INDX, START, END, scan_counts_to_offsets_loop_check) { \

#define SCAN_COUNTS_TO_OFFSETS_LOOP_END(INDX, LENGTH, SCANVAR) \
      } CARE_CHECKED_SEQUENTIAL_LOOP_END(scan_counts_to_offsets_loop_check) \
      care::exclusive_scan(RAJA::seq_exec{}, SCANVAR, nullptr, LENGTH, 0, true); \
   }

#define MANAGED_PTR_SCAN_LOOP(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR)  \
   SCAN_LOOP_P(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR)

#define MANAGED_PTR_SCAN_LOOP_END(END, SCANINDX, SCANLENGTH) \
   SCAN_LOOP_P_END(END, SCANINDX, SCANLENGTH)

#endif // !defined(CARE_NEVER_USE_RAJA_PARALLEL_SCAN)

#endif // !defined(_CARE_SCAN_H_)

