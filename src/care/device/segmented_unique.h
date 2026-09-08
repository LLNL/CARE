//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_DEVICE_SEGMENTED_UNIQUE_H
#define CARE_DEVICE_SEGMENTED_UNIQUE_H

#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"
#include "care/scan.h"

#include <cstddef>
namespace care::device {

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
   const size_t numItems = keys.size();
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;

   // One extra entry lets the exclusive scan's final value hold the total
   // number of unique keys.
   care::host_device_ptr<int> positions(numItems + 1);

   CARE_STREAM_LOOP(i, 0, numItems + 1) {
      positions[i] = 0;
   } CARE_STREAM_LOOP_END

   // Mark unique values by comparing adjacent keys. Segment starts are fixed
   // in a separate kernel so equal values on opposite sides of a boundary are
   // both retained.
   CARE_STREAM_LOOP(i, 0, numItems) {
      positions[i] = static_cast<int>(
         i == 0 || keys[i - 1] < keys[i] || keys[i] < keys[i - 1]);
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(segment, 0, numSegments) {
      const OffsetT begin = offsets[segment];
      if (begin < offsets[segment + 1]) {
         positions[begin] = 1;
      }
   } CARE_STREAM_LOOP_END

   care::exclusive_scan(RAJADeviceExec {}, positions, nullptr,
                        static_cast<int>(numItems + 1), 0, true);

   const int numUnique = positions.pick(numItems);
   care::host_device_ptr<KeyT> result(static_cast<size_t>(numUnique));

   CARE_STREAM_LOOP(i, 0, numItems) {
      if (positions[i] != positions[i + 1]) {
         result[positions[i]] = keys[i];
      }
   } CARE_STREAM_LOOP_END

   CARE_STREAM_LOOP(segment, 0, offsets.size()) {
      offsets[segment] = static_cast<OffsetT>(positions[offsets[segment]]);
   } CARE_STREAM_LOOP_END

   positions.free();

   care::host_device_ptr<const KeyT> source = result;

   CARE_STREAM_LOOP(i, 0, numUnique) {
      keys[i] = source[i];
   } CARE_STREAM_LOOP_END

   result.free();
}

} // namespace care::device

#endif // CARE_DEVICE_SEGMENTED_UNIQUE_H
