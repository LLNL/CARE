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

// Other Care headers
#include "care/CHAIDataGetter.h"

// Other library headers
#include "chai/ManagedArray.hpp"
#include "RAJA/RAJA.hpp"


// exclusive scan functionality
template <typename T, typename Exec, typename Fn>
void exclusive_scan(chai::ManagedArray<T> data, chai::ManagedArray<T> outData,
                    int size, Fn binop, T val, bool inPlace);

template <typename T, typename Exec, typename Fn>
void exclusive_scan(chai::ManagedArray<T> data, chai::ManagedArray<T> outData,
                    int size, Fn binop, T val, bool inPlace) {
   if (size > 1 && data != nullptr) {
      CHAIDataGetter<T, Exec> D {};
      T * rawData = D.getRawArrayData(data);
      T * rawDataEnd = rawData+size;
      if (inPlace) {
         RAJA::exclusive_scan_inplace<Exec, T *, T, Fn>(Exec {}, rawData, rawDataEnd, binop, val);
      }
      else {
         T * rawOutData = D.getRawArrayData(outData);
         RAJA::exclusive_scan<Exec, T*, T*, T, Fn>(Exec {},
                                                   rawData,
                                                   rawDataEnd,
                                                   rawOutData,
                                                   binop, val);
      }
   }
   else {
      if ( size == 1) {
         if (! inPlace) {
            if (outData != nullptr) {
               outData.set(0,val);
            }
         } else {
            if (data != nullptr) {
               data.set(0,val);
            }
         }
      }
      else {
         printf("care::scan - unhandled combination of size, data, and outData\n - no-op will occur.\n");
      }
   }
}

//typesafe wrapper for out of place scan
template <typename T, typename Exec, typename Fn>
void exclusive_scan(chai::ManagedArray<const T> inData, chai::ManagedArray<T> outData,
                    int size, Fn binop, T val) { 
    const bool inPlace = false;
    exclusive_scan<T, Exec, Fn>(*reinterpret_cast<chai::ManagedArray<T> *>(&inData), outData, size, binop, val, inPlace);
}

// inclusive scan functionality
template <typename T, typename Exec, typename Fn>
void inclusive_scan(chai::ManagedArray<T> data, chai::ManagedArray<T> outData,
                    int size, Fn binop, bool inPlace);

template <typename T, typename Exec, typename Fn>
void inclusive_scan(chai::ManagedArray<T> data, chai::ManagedArray<T> outData,
                    int size, Fn binop, bool inPlace) {
   CHAIDataGetter<T, Exec> D {};
   T * rawData = D.getRawArrayData(data);
   T * rawDataEnd = rawData+size;
   if (inPlace) {
      RAJA::inclusive_scan_inplace<Exec, T *, Fn>(Exec {}, rawData, rawDataEnd, binop);
   }
   else {
      T * rawOutData = D.getRawArrayData(outData);
      RAJA::inclusive_scan<Exec, T*, T*, Fn>(Exec {},
                                             rawData,
                                             rawDataEnd,
                                             rawOutData,
                                             binop);
   }
}

//typesafe wrapper for out of place scan
template <typename T, typename Exec, typename Fn>
void inclusive_scan(chai::ManagedArray<const T> inData, chai::ManagedArray<T> outData,
                    int size, Fn binop, T val) { 
    const bool inPlace = false;
    inclusive_scan<T, Exec, Fn>(*reinterpret_cast<chai::ManagedArray<T> *>(&inData), outData, size, binop, val, inPlace);
}

template<typename T>
inline void getFinalScanCountFromPinned(chai::ManagedArray<T> scanvar_length, T& scanCount) {
   CARE_CHECKED_HOST_KERNEL_WITH_REF_START(scan_loop_check, scanCount) {
      scanCount = scanvar_length[0];
   } CARE_CHECKED_HOST_KERNEL_WITH_REF_END(scan_loop_check)
}

template<typename T>
inline void getFinalScanCount(chai::ManagedArray<T> scanvar, int length, T& scanCount) {
   scanCount = scanvar.pick(length);
}

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

#if defined GPU_ACTIVE || defined CARE_ALWAYS_USE_RAJA_SCAN
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
      exclusive_scan<int, RAJAExec>(SCANVAR, nullptr, END-START+1, RAJA::operators::plus<int>{}, SCANVAR_OFFSET, true); \
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
      exclusive_scan<GIDTYPE, RAJAExec>(SCANVAR, nullptr, END-START+1, RAJA::operators::plus<GIDTYPE>{}, SCANVAR_OFFSET.Value(), true); \
   } else { \
      CARE_CHECKED_SEQUENTIAL_LOOP_START(INDX, 0, 1, scan_loop_gid_init_check) { \
         SCANVAR[INDX] = SCANVAR_OFFSET.Value(); \
         SCANVARLENGTH[0] = SCANVAR_OFFSET.Value(); \
      } CARE_CHECKED_SEQUENTIAL_LOOP_END(scan_loop_gid_init_check) \
   }

#endif // CARE_HAVE_LLNL_GLOBALID

// grab the number of elements that met the scan criteria, place it in SCANLENGTH
#define SCAN_LOOP_FINAL(END, SCANVARLENGTH, SCANCOUNT) getFinalScanCountFromPinned(SCANVARLENGTH, SCANCOUNT);

#define SCAN_LOOP(INDX, START, END, SCANINDX, SCANINDX_OFFSET, EXPR) \
   { \
      int const SCANVARSTARTNAME(SCANINDX) = START; \
      ScanVar SCANVARNAME(SCANINDX)(END-START+1); \
      ScanVar SCANVARLENGTHNAME(SCANINDX)(1, chai::PINNED); \
      SCAN_LOOP_INIT(INDX, SCANVARSTARTNAME(SCANINDX), END, SCANVARNAME(SCANINDX), SCANVARLENGTHNAME(SCANINDX), SCANINDX_OFFSET, EXPR); \
      int const SCANVARENDNAME(SCANINDX) = END; \
      CARE_CHECKED_PARALLEL_LOOP_START(INDX, START, END, scan_loop_check) { \
         if (INDX == SCANVARENDNAME(SCANINDX) -1) { \
            SCANVARLENGTHNAME(SCANINDX)[0] = SCANVARNAME(SCANINDX)[SCANVARENDNAME(SCANINDX)-START]; \
         } \
         if (EXPR) { \
            const int SCANINDX = SCANVARNAME(SCANINDX)[INDX-SCANVARSTARTNAME(SCANINDX)];

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
      ScanVarGID SCANVARLENGTHNAME(SCANINDX)(1, chai::PINNED); \
      SCAN_LOOP_GID_INIT(INDX, SCANVARSTARTNAME(SCANINDX), END, SCANVARNAME(SCANINDX), SCANVARLENGTHNAME(SCANINDX), SCANINDX_OFFSET, EXPR); \
      int const SCANVARENDNAME(SCANINDX) = END; \
      CARE_CHECKED_PARALLEL_LOOP_START(INDX, START, END, scan_loop_gid_check) { \
         if (INDX == SCANVARENDNAME(SCANINDX)-1) { \
            SCANVARLENGTHNAME(SCANINDX)[0] = SCANVARNAME(SCANINDX)[SCANVARENDNAME(SCANINDX)-START]; \
         } \
         if (EXPR) { \
            const globalID SCANINDX = globalID(SCANVARNAME(SCANINDX)[INDX-SCANVARSTARTNAME(SCANINDX)]);

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
      ScanVar SCANVARLENGTHNAME(SCANINDX)(1, chai::PINNED); \
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
      ScanVar SCANVARLENGTHNAME(SCANINDX)(1, chai::PINNED); \
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
      exclusive_scan<int, RAJAExec>(SCANVAR, nullptr, LENGTH, RAJA::operators::plus<int>{}, 0, true); \
   }
         

#else // GPU_ACTIVE || CARE_ALWAYS_USE_RAJA_SCAN

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
      exclusive_scan<int, RAJA::seq_exec>(SCANVAR, nullptr, LENGTH, RAJA::operators::plus<int>{}, 0, true); \
   }
#endif // GPU_ACTIVE || CARE_ALWAYS_USE_RAJA_SCAN

#endif // !defined(_CARE_SCAN_H_)

