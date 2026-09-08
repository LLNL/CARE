//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HOST_SEGMENTED_UNIQUE_H
#define CARE_HOST_SEGMENTED_UNIQUE_H

#include "care/host_device_ptr.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace care::host {

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
 */
template <typename KeyT, typename OffsetT, typename BinaryPredicate>
void segmented_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets,
   BinaryPredicate binaryPredicate)
{
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;
   KeyT* rawKeys = keys.data();
   OffsetT* rawOffsets = offsets.data();

   size_t output = 0;
   for (size_t segment = 0; segment < numSegments; ++segment) {
      const size_t begin = static_cast<size_t>(rawOffsets[segment]);
      const size_t end = static_cast<size_t>(rawOffsets[segment + 1]);
      rawOffsets[segment] = static_cast<OffsetT>(output);

      if (begin < end) {
         KeyT* uniqueEnd = std::unique(
            rawKeys + begin,
            rawKeys + end,
            binaryPredicate);
         const size_t numUnique = static_cast<size_t>(
            uniqueEnd - (rawKeys + begin));

         if (output != begin) {
            // The source and destination ranges may overlap.
            for (size_t i = 0; i < numUnique; ++i) {
               rawKeys[output + i] = std::move(rawKeys[begin + i]);
            }
         }
         output += numUnique;
      }
   }

   if (offsets.size() > 0) {
      rawOffsets[numSegments] = static_cast<OffsetT>(output);
   }
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
void segmented_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets)
{
   segmented_unique(keys, offsets,
      [] (KeyT const& left, KeyT const& right) {
         return left == right;
      });
}

} // namespace care::host

#endif // CARE_HOST_SEGMENTED_UNIQUE_H
