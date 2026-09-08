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
#include "care/scan.h"

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

/**
 * @brief Sort each segment and copy its unique keys into a compact array.
 *
 * Equal keys in different segments remain distinct. Empty segments are
 * preserved in @p uniqueOffsets.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_sort_and_unique(
   care::host_device_ptr<KeyT> const& keys,
   care::host_device_ptr<OffsetT> const& offsets,
   care::host_device_ptr<KeyT>& uniqueKeys,
   care::host_device_ptr<OffsetT>& uniqueOffsets)
{
   const size_t numItems = keys.size();
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;

   care::host_device_ptr<KeyT> sortedKeys(numItems);
   CARE_STREAM_LOOP(i, 0, numItems) {
      sortedKeys[i] = keys[i];
   } CARE_STREAM_LOOP_END

   segmented_sort(sortedKeys, offsets);

   // The final scan entry holds the total number of unique keys.
   care::host_device_ptr<size_t> positions(numItems + 1);
   CARE_STREAM_LOOP(i, 0, numItems + 1) {
      positions[i] = 0;
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(i, 0, numItems) {
      positions[i] = static_cast<size_t>(
         i == 0 || sortedKeys[i - 1] < sortedKeys[i] ||
         sortedKeys[i] < sortedKeys[i - 1]);
   } CARE_STREAM_LOOP_END

   // Segment starts are always unique, including when adjacent segments end
   // and begin with equal keys.
   CARE_STREAM_LOOP(segment, 0, numSegments) {
      const OffsetT begin = offsets[segment];
      if (begin < offsets[segment + 1]) {
         positions[begin] = 1;
      }
   } CARE_STREAM_LOOP_END

   care::exclusive_scan(RAJADeviceExec {}, positions, nullptr,
                        static_cast<int>(numItems + 1), size_t {}, true);

   const size_t numUnique = positions.pick(numItems);
   care::host_device_ptr<KeyT> result(numUnique);
   care::host_device_ptr<OffsetT> resultOffsets(offsets.size());

   CARE_STREAM_LOOP(i, 0, numItems) {
      if (positions[i] != positions[i + 1]) {
         result[positions[i]] = sortedKeys[i];
      }
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(segment, 0, offsets.size()) {
      resultOffsets[segment] =
         static_cast<OffsetT>(positions[offsets[segment]]);
   } CARE_STREAM_LOOP_END

   positions.free();
   sortedKeys.free();

   uniqueKeys.free();
   uniqueOffsets.free();
   uniqueKeys = std::move(result);
   uniqueOffsets = std::move(resultOffsets);
}

/**
 * @brief Replace segmented keys and offsets with sorted unique results.
 */
template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_sort_and_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets)
{
   care::host_device_ptr<KeyT> uniqueKeys;
   care::host_device_ptr<OffsetT> uniqueOffsets;

   segmented_sort_and_unique(
      static_cast<care::host_device_ptr<KeyT> const&>(keys),
      static_cast<care::host_device_ptr<OffsetT> const&>(offsets),
      uniqueKeys,
      uniqueOffsets);

   keys.free();
   offsets.free();
   keys = std::move(uniqueKeys);
   offsets = std::move(uniqueOffsets);
}

} // namespace care::hip

#endif // CARE_HIP_SORT_H
