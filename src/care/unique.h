//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_UNIQUE_H
#define CARE_UNIQUE_H

#include "care/host_device_ptr.h"

#if defined(__CUDACC__)
#include "care/cuda/unique.h"
#elif defined(__HIPCC__)
#include "care/hip/unique.h"
#else
#include "care/host/unique.h"
#endif

namespace care {

/**
 * @brief Remove adjacent duplicate keys from a sorted array.
 * @param keys Sorted keys to compact in place. The compacted keys occupy the
 * first returned-number entries; the allocation and size are unchanged.
 * @param binaryPredicate Returns true when two adjacent keys are equivalent.
 * When compiling for CUDA or HIP, it must be callable on the device.
 * @return The number of unique keys.
 */
template <typename KeyT, typename BinaryPredicate>
CARE_INLINE size_t unique(care::host_device_ptr<KeyT>& keys,
                          BinaryPredicate binaryPredicate)
{
#if defined(__CUDACC__)
   return care::cuda::unique(keys, binaryPredicate);
#elif defined(__HIPCC__)
   return care::hip::unique(keys, binaryPredicate);
#else
   return care::host::unique(keys, binaryPredicate);
#endif
}

/**
 * @brief Remove adjacent duplicate keys from a sorted array using equality
 * comparison.
 * @param keys Sorted keys to compact in place. The compacted keys occupy the
 * first returned-number entries; the allocation and size are unchanged.
 * @return The number of unique keys.
 */
template <typename KeyT>
CARE_INLINE size_t unique(care::host_device_ptr<KeyT>& keys)
{
   return care::unique(keys,
      [] CARE_HOST_DEVICE (KeyT const& left, KeyT const& right) {
         return left == right;
      });
}

/**
 * @brief Remove duplicate keys independently within each sorted segment.
 * @param keys Sorted keys to compact in place. On return, the compacted keys
 * occupy the first offsets[N] entries; the allocation and size are unchanged.
 * Equal keys in different segments remain distinct.
 * @param offsets Segment boundaries to update in place. For N segments,
 * offsets must contain N + 1 entries: offsets[i] begins segment i, and
 * offsets[N] marks the end of the final segment. Segment i is therefore
 * [offsets[i], offsets[i + 1]). The entries must form a nondecreasing sequence
 * from 0 to keys.size(); that is, offsets[0] must be 0 and offsets[N] must
 * equal keys.size(). Repeated entries denote empty segments. On return,
 * contains the segment boundaries into the compacted @p keys.
 * @param binaryPredicate Returns true when two adjacent keys are equivalent.
 * When compiling for CUDA or HIP, it must be callable on the device.
 */
template <typename KeyT, typename OffsetT, typename BinaryPredicate>
CARE_INLINE void segmented_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets,
   BinaryPredicate binaryPredicate)
{
#if defined(__CUDACC__)
   care::cuda::segmented_unique(keys, offsets, binaryPredicate);
#elif defined(__HIPCC__)
   care::hip::segmented_unique(keys, offsets, binaryPredicate);
#else
   care::host::segmented_unique(keys, offsets, binaryPredicate);
#endif
}

/**
 * @brief Remove duplicate keys independently within each sorted segment using
 * equality comparison.
 * @param keys Sorted keys to compact in place. On return, the compacted keys
 * occupy the first offsets[N] entries; the allocation and size are unchanged.
 * Equal keys in different segments remain distinct.
 * @param offsets Segment boundaries to update in place. For N segments,
 * offsets must contain N + 1 entries: offsets[i] begins segment i, and
 * offsets[N] marks the end of the final segment. Segment i is therefore
 * [offsets[i], offsets[i + 1]). The entries must form a nondecreasing sequence
 * from 0 to keys.size(); that is, offsets[0] must be 0 and offsets[N] must
 * equal keys.size(). Repeated entries denote empty segments. On return,
 * contains the segment boundaries into the compacted @p keys.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets)
{
   segmented_unique(keys, offsets,
      [] CARE_HOST_DEVICE (KeyT const& left, KeyT const& right) {
         return left == right;
      });
}

} // namespace care

#endif // CARE_UNIQUE_H
