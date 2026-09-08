//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_SEGMENTED_UNIQUE_H
#define CARE_SEGMENTED_UNIQUE_H

#include "care/host_device_ptr.h"

#if defined(__CUDACC__)
#include "care/cuda/segmented_unique.h"
#elif defined(__HIPCC__)
#include "care/hip/segmented_unique.h"
#else
#include "care/host/segmented_unique.h"
#endif

namespace care {

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
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets)
{
#if defined(__CUDACC__)
   care::cuda::segmented_unique(keys, offsets);
#elif defined(__HIPCC__)
   care::hip::segmented_unique(keys, offsets);
#else
   care::host::segmented_unique(keys, offsets);
#endif
}

} // namespace care

#endif // CARE_SEGMENTED_UNIQUE_H
