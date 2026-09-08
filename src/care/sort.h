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

} // namespace care

#endif // CARE_SORT_H
