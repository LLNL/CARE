//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HIP_SORT_H
#define CARE_HIP_SORT_H

#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"

#include <cstddef>
#include <utility>

#include "rocprim/rocprim.hpp"

namespace care::hip {

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
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;
   const size_t numItems = keys.size();

   if (numSegments == 0 || numItems == 0) {
      return;
   }

   CHAIDataGetter<KeyT, RAJADeviceExec> keyGetter {};
   auto* rawKeys = keyGetter.getRawArrayData(keys);

   CHAIDataGetter<OffsetT, RAJADeviceExec> offsetGetter {};
   const auto* rawOffsets = offsetGetter.getConstRawArrayData(offsets);

   care::host_device_ptr<KeyT> result(numItems);
   auto* rawResult = keyGetter.getRawArrayData(result);

   size_t tempStorageBytes = 0;
   rocprim::segmented_radix_sort_keys(nullptr, tempStorageBytes,
                                      rawKeys, rawResult, numItems, numSegments,
                                      rawOffsets, rawOffsets + 1);

   CHAIDataGetter<char, RAJADeviceExec> charGetter {};
   care::host_device_ptr<char> tempStorage(tempStorageBytes);
   auto* rawTempStorage = charGetter.getRawArrayData(tempStorage);
   rocprim::segmented_radix_sort_keys(rawTempStorage, tempStorageBytes,
                                      rawKeys, rawResult, numItems, numSegments,
                                      rawOffsets, rawOffsets + 1);

   tempStorage.free();

   if (keys.isSlice()) {
      care::host_device_ptr<const KeyT> source = result;

      CARE_STREAM_LOOP(i, 0, numItems) {
         keys[i] = source[i];
      } CARE_STREAM_LOOP_END

      result.free();
   } else {
      keys.free();
      keys = std::move(result);
   }
}

} // namespace care::hip

#endif // CARE_HIP_SORT_H
