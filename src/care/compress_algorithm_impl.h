//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_COMPRESS_ALGORITHM_IMPL_H
#define CARE_COMPRESS_ALGORITHM_IMPL_H

// This header includes the implementations of the CARE compress algorithms.

#include "care/algorithm_decl.h"
#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/scan.h"

namespace care {

/************************************************************************
* Function  : CompressArray<T>
* Author(s) : Peter Robinson, Benjamin Liu, Extended by AI Assistant
* Purpose   : Compress an array based on list of array indices or flags.
*             Based on listType, the list is either:
*             removed_list: a list of indices to remove
*                or
*             mapping_list: a mapping from compressed indices to original indices
*                or
*             remove_flag_list: a list of 1s and 0s where 1 means remove the element
*                or
*             keep_flag_list: a list of 1s and 0s where 1 means keep the element
*             All entries in list must be > 0 and < arrLen for index lists.
*             Flag lists must be the same length as the array.
*             If the realloc parameter is true, arr will be resized/reallocated to
*             the compressed size.
*             Thread safe version of CompressArray.
**************************************************************************/
#ifdef CARE_PARALLEL_DEVICE
template <typename T>
CARE_INLINE int CompressArray(RAJADeviceExec exec, care::host_device_ptr<T> & arr, const int arrLen,
                              care::host_device_ptr<int const> list, const int listLen,
                              const care::compress_array listType, bool realloc)
{
   //GPU VERSION
   if (listType == care::compress_array::removed_list) {
      care::host_device_ptr<T> tmp(arrLen-listLen, "CompressArray_tmp");
      int numKept = 0;
      SCAN_LOOP(i, 0, arrLen, pos, numKept,
                -1 == BinarySearch<int>(list, 0, listLen, i)) {
         tmp[pos] = arr[i];
      } SCAN_LOOP_END(arrLen, pos, numKept)

#ifdef CARE_DEBUG
      int numRemoved = arrLen - numKept;
      if (listLen != numRemoved) {
         printf("Warning in CompressArray<T>: did not remove expected number of members!\n");
      }
#endif
      if (realloc) {
         arr.free();
         arr = tmp;
      }
      else {
         ArrayCopy(exec, arr, reinterpret_cast<care::host_device_ptr<const T> &>(tmp), numKept);
         tmp.free();
      }
      return numKept;
   }
   else if (listType == care::compress_array::mapping_list) {
      care::host_device_ptr<T> tmp(arrLen, "CompressArray tmp");
      ArrayCopy<T>(tmp, arr, arrLen);
      if (realloc) {
         arr.realloc(listLen) ;
      }
      CARE_STREAM_LOOP(newIndex, 0, listLen) {
         int oldIndex = list[newIndex] ;
         arr[newIndex] = tmp[oldIndex] ;
      } CARE_STREAM_LOOP_END
      tmp.free();
      return listLen;
   }
   else if (listType == care::compress_array::remove_flag_list) {
      // For remove_flag_list, 1 means remove the element, 0 means keep it
      care::host_device_ptr<int> keepIndices(arrLen, "CompressArray keepIndices");
      int numKept = 0;
      
      // First, identify which elements to keep
      SCAN_LOOP(i, 0, arrLen, pos, numKept, list[i] == 0) {
         keepIndices[pos] = i;
      } SCAN_LOOP_END(arrLen, pos, numKept)
      
      // Create a temporary array to hold the kept elements
      care::host_device_ptr<T> tmp(numKept, "CompressArray_tmp");
      
      // Copy the kept elements to the temporary array
      CARE_STREAM_LOOP(i, 0, numKept) {
         tmp[i] = arr[keepIndices[i]];
      } CARE_STREAM_LOOP_END
      
      if (realloc) {
         arr.free();
         arr = tmp;
      }
      else {
         ArrayCopy(exec, arr, reinterpret_cast<care::host_device_ptr<const T> &>(tmp), numKept);
         tmp.free();
      }
      
      keepIndices.free();
      return numKept;
   }
   else if (listType == care::compress_array::keep_flag_list) {
      // For keep_flag_list, 1 means keep the element, 0 means remove it
      care::host_device_ptr<int> keepIndices(arrLen, "CompressArray keepIndices");
      int numKept = 0;
      
      // First, identify which elements to keep
      SCAN_LOOP(i, 0, arrLen, pos, numKept, list[i] == 1) {
         keepIndices[pos] = i;
      } SCAN_LOOP_END(arrLen, pos, numKept)
      
      // Create a temporary array to hold the kept elements
      care::host_device_ptr<T> tmp(numKept, "CompressArray_tmp");
      
      // Copy the kept elements to the temporary array
      CARE_STREAM_LOOP(i, 0, numKept) {
         tmp[i] = arr[keepIndices[i]];
      } CARE_STREAM_LOOP_END
      
      if (realloc) {
         arr.free();
         arr = tmp;
      }
      else {
         ArrayCopy(exec, arr, reinterpret_cast<care::host_device_ptr<const T> &>(tmp), numKept);
         tmp.free();
      }

      keepIndices.free();

      return numKept;
      
   }
   else {
#ifdef CARE_DEBUG
      printf("Warning in CompressArray<T>: unsupported compressArray mode!\n");
#endif
      return -1;
   }
}

#endif // defined(CARE_PARALLEL_DEVICE)

/************************************************************************
* Function  : CompressArray<T>
* Author(s) : Peter Robinson, Benjamin Liu, Extended by AI Assistant
* Purpose   : Compress an array based on list of array indices or flags.
*             Based on listType, the list is either:
*             removed_list: a list of indices to remove
*                or
*             mapping_list: a mapping from compressed indices to original indices
*                or
*             remove_flag_list: a list of 1s and 0s where 1 means remove the element
*                or
*             keep_flag_list: a list of 1s and 0s where 1 means keep the element
*             All entries in list must be > 0 and < arrLen for index lists.
*             Flag lists must be the same length as the array.
*             If the realloc parameter is true, arr will be resized/reallocated to
*             the compressed size.
*             Sequential Version of CompressArray
*             Requires both arr and list to be sorted for removed_list.
**************************************************************************/
template <typename T>
CARE_INLINE int CompressArray(RAJA::seq_exec, care::host_device_ptr<T> & arr, const int arrLen,
                              care::host_device_ptr<int const> list, const int listLen,
                              const care::compress_array listType, bool realloc)
{
   // CPU VERSION
   if (listType == care::compress_array::removed_list) {
      int readLoc;
      int writeLoc = 0, numRemoved = 0;
      care::host_ptr<int const> listHost = list ;
      care::host_ptr<T> arrHost = arr ;
#ifdef CARE_DEBUG
      if (listHost[listLen-1] > arrLen-1) {
         printf("Warning in CompressArray<T> seq_exec: asking to remove entries not in array!\n");
      }
#endif
      for (readLoc = 0; readLoc < arrLen; ++readLoc) {
         if ((numRemoved == listLen) || (readLoc < listHost[numRemoved])) {
            arrHost[writeLoc++] = arrHost[readLoc];
         }
         else if (readLoc == listHost[numRemoved]) {
            ++numRemoved;
         }
#ifdef CARE_DEBUG
         else {
            printf("Warning in CompressArray<int> seq_exec: list of removed members not sorted!\n");
         }
#endif
      }
#ifdef CARE_DEBUG
      if ((listLen != numRemoved) || (writeLoc != arrLen - listLen)) {
         printf("CompressArray<T> seq_exec: did not remove expected number of members!\n");
      }
#endif
      if (realloc) {
         arr.realloc(arrLen - listLen) ;
      }
      return arrLen - listLen;
   }
   else if (listType == care::compress_array::mapping_list) {
      CARE_SEQUENTIAL_LOOP(newIndex, 0, listLen) {
         int oldIndex = list[newIndex] ;
#ifdef CARE_DEBUG
         if (oldIndex > arrLen-1 || oldIndex < 0) {
            printf("Warning in CompressArray<T> seq_exec: asking to remove entries not in array!\n");
         }
#endif
         arr[newIndex] = arr[oldIndex] ;
      } CARE_SEQUENTIAL_LOOP_END
      if (realloc) {
         arr.realloc(listLen) ;
      }
      return listLen;
   }
   else if (listType == care::compress_array::remove_flag_list) {
      // For remove_flag_list, 1 means remove the element, 0 means keep it
      care::host_ptr<int const> listHost = list;
      care::host_ptr<T> arrHost = arr;
      
      int writeLoc = 0;
      for (int readLoc = 0; readLoc < arrLen; ++readLoc) {
         if (listHost[readLoc] == 0) { // Keep this element
            arrHost[writeLoc++] = arrHost[readLoc];
         }
      }
      
      int numKept = writeLoc;
      
      if (realloc) {
         arr.realloc(numKept);
      }
      return numKept;
   }
   else if (listType == care::compress_array::keep_flag_list) {
      // For keep_flag_list, 1 means keep the element, 0 means remove it
      care::host_ptr<int const> listHost = list;
      care::host_ptr<T> arrHost = arr;
      
      int writeLoc = 0;
      for (int readLoc = 0; readLoc < arrLen; ++readLoc) {
         if (listHost[readLoc] == 1) { // Keep this element
            arrHost[writeLoc++] = arrHost[readLoc];
         }
      }
      
      int numKept = writeLoc;
      
      if (realloc) {
         arr.realloc(numKept);
      }
      return numKept;
   }
   else {
#ifdef CARE_DEBUG
      printf("Warning in CompressArray<T>: unsupported compressArray mode!\n");
#endif
      return -1;
   }
}

/************************************************************************
* Function  : CompressArray<T>
* Author(s) : Peter Robinson, Benjamin Liu, Extended by AI Assistant
* Purpose   : Compress an array based on list of array indices or flags.
*             Based on listType, the list is either:
*             removed_list: a list of indices to remove
*                or
*             mapping_list: a mapping from compressed indices to original indices
*                or
*             remove_flag_list: a list of 1s and 0s where 1 means remove the element
*                or
*             keep_flag_list: a list of 1s and 0s where 1 means keep the element
*             All entries in list must be > 0 and < arrLen for index lists.
*             Flag lists must be the same length as the array.
*             If the realloc parameter is true, arr will be resized/reallocated to
*             the compressed size.
*             Both arr and list should be sorted to support the sequential
*             implementation for removed_list.
**************************************************************************/
template <typename T>
CARE_INLINE int CompressArray(care::host_device_ptr<T> & arr, const int arrLen,
                              care::host_device_ptr<int const> list, const int listLen,
                              const care::compress_array listType, bool realloc)
{
#ifdef CARE_DEBUG
   if (listType == care::compress_array::removed_list) {
      checkSorted<T>(arr, arrLen, "CompressArray", "arr") ;
      checkSorted<int>(list, listLen, "CompressArray", "list") ;
   }
#endif
   return CompressArray(RAJAExec(), arr, arrLen, list, listLen, listType, realloc);
}

} // namespace care

#endif // CARE_COMPRESS_ALGORITHM_IMPL_H