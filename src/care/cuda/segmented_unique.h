//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_CUDA_SEGMENTED_UNIQUE_H
#define CARE_CUDA_SEGMENTED_UNIQUE_H

#include "care/device/segmented_unique.h"

namespace care::cuda {

/**
 * @brief Remove duplicate keys independently within each sorted segment.
 * @param keys Sorted keys to compact in place. On return, contains the unique
 * keys from every segment and is resized to the compacted length. Equal keys
 * in different segments remain distinct.
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
   care::device::segmented_unique(keys, offsets);
}

} // namespace care::cuda

#endif // CARE_CUDA_SEGMENTED_UNIQUE_H
