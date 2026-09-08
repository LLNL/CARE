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
 * @brief Compact each sorted segment to its unique values in place.
 *
 * Equal values in different segments remain distinct. Empty segments are
 * preserved in @p offsets.
 *
 * @param keys Sorted keys containing all input segments. Replaced by the
 * compact unique keys.
 * @param offsets Segment boundaries. For N segments, offsets must
 * contain N + 1 entries: offsets[i] begins segment i, and offsets[N] marks
 * the end of the final segment. The entries must form a nondecreasing
 * sequence from 0 to keys.size(). Repeated entries denote empty segments.
 * Replaced by boundaries into the compacted @p keys.
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
