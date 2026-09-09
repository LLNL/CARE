//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_SORT_H
#define CARE_SORT_H

#include "care/host_device_ptr.h"

#if defined(__CUDACC__)
#include "care/cuda/sort.h"
#elif defined(__HIPCC__)
#include "care/hip/sort.h"
#else
#include "care/host/sort.h"
#endif

namespace care {

/**
 * @brief Sort keys in ascending order independently within each segment.
 * @param keys Keys to sort in place.
 * @param offsets Segment boundaries. For N segments, offsets must contain
 * N + 1 entries: offsets[i] begins segment i, and offsets[N] marks the end of
 * the final segment. Segment i is therefore [offsets[i], offsets[i + 1]).
 * The entries must form a nondecreasing sequence from 0 to keys.size(); that
 * is, offsets[0] must be 0 and offsets[N] must equal keys.size(). Repeated
 * entries denote empty segments.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_sort(care::host_device_ptr<KeyT>& keys,
                                care::host_device_ptr<OffsetT> const& offsets)
{
#if defined(__CUDACC__)
   care::cuda::segmented_sort(keys, offsets);
#elif defined(__HIPCC__)
   care::hip::segmented_sort(keys, offsets);
#else
   care::host::segmented_sort(keys, offsets);
#endif
}

/**
 * @brief Sort each segment and copy its unique keys into a compact array.
 *
 * Equal keys in different segments remain distinct. Empty segments are
 * preserved in @p uniqueOffsets.
 *
 * @param keys Keys containing all input segments.
 * @param offsets Input segment boundaries. For N segments, offsets must
 * contain N + 1 entries: offsets[i] begins segment i, and offsets[N] marks
 * the end of the final segment. The entries must form a nondecreasing
 * sequence from 0 to keys.size(). Repeated entries denote empty segments.
 * @param uniqueKeys Compact output containing the sorted unique keys from
 * every segment. This array must not alias @p keys or @p offsets.
 * @param uniqueOffsets Output segment boundaries into @p uniqueKeys. It has
 * the same number of entries as @p offsets and must not alias an input.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_sort_and_unique(
   care::host_device_ptr<KeyT> const& keys,
   care::host_device_ptr<OffsetT> const& offsets,
   care::host_device_ptr<KeyT>& uniqueKeys,
   care::host_device_ptr<OffsetT>& uniqueOffsets)
{
#if defined(__CUDACC__)
   care::cuda::segmented_sort_and_unique(
      keys, offsets, uniqueKeys, uniqueOffsets);
#elif defined(__HIPCC__)
   care::hip::segmented_sort_and_unique(
      keys, offsets, uniqueKeys, uniqueOffsets);
#else
   care::host::segmented_sort_and_unique(
      keys, offsets, uniqueKeys, uniqueOffsets);
#endif
}

/**
 * @brief Replace segmented keys and offsets with sorted unique results.
 *
 * This overload replaces both allocations, so slices are detached from their
 * original backing allocations. Use the four-argument overload when the input
 * handles must remain unchanged.
 *
 * @param keys Keys containing all input segments. Replaced by the compact
 * sorted unique keys.
 * @param offsets Segment boundaries into @p keys. Replaced by boundaries into
 * the compact result.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_sort_and_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets)
{
#if defined(__CUDACC__)
   care::cuda::segmented_sort_and_unique(keys, offsets);
#elif defined(__HIPCC__)
   care::hip::segmented_sort_and_unique(keys, offsets);
#else
   care::host::segmented_sort_and_unique(keys, offsets);
#endif
}

} // namespace care

#endif // CARE_SORT_H
