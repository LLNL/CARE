//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HOST_SORT_H
#define CARE_HOST_SORT_H

#include "care/host_device_ptr.h"

#include <algorithm>
#include <cstddef>
#include <utility>

namespace care::host {

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
void segmented_sort(care::host_device_ptr<KeyT>& keys,
                    care::host_device_ptr<OffsetT> const& offsets)
{
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;

   KeyT* rawKeys = keys.data();
   const OffsetT* rawOffsets = offsets.cdata();

   for (size_t segment = 0; segment < numSegments; ++segment) {
      const size_t begin = static_cast<size_t>(rawOffsets[segment]);
      const size_t end = static_cast<size_t>(rawOffsets[segment + 1]);
      std::sort(rawKeys + begin, rawKeys + end);
   }
}

/**
 * @brief Sort each segment and copy its unique keys into a compact array.
 *
 * Equal keys in different segments remain distinct. Empty segments are
 * preserved in @p uniqueOffsets.
 */
template <typename KeyT, typename OffsetT>
void segmented_sort_and_unique(
   care::host_device_ptr<KeyT> const& keys,
   care::host_device_ptr<OffsetT> const& offsets,
   care::host_device_ptr<KeyT>& uniqueKeys,
   care::host_device_ptr<OffsetT>& uniqueOffsets)
{
   const size_t numItems = keys.size();
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;
   const KeyT* rawKeys = keys.cdata();
   const OffsetT* rawOffsets = offsets.cdata();

   care::host_device_ptr<KeyT> result(numItems);
   KeyT* rawResult = result.data();
   if (numItems > 0) {
      std::copy(rawKeys, rawKeys + numItems, rawResult);
   }

   care::host_device_ptr<OffsetT> resultOffsets(offsets.size());
   OffsetT* rawResultOffsets = resultOffsets.data();

   size_t output = 0;
   for (size_t segment = 0; segment < numSegments; ++segment) {
      rawResultOffsets[segment] = static_cast<OffsetT>(output);

      const size_t begin = static_cast<size_t>(rawOffsets[segment]);
      const size_t end = static_cast<size_t>(rawOffsets[segment + 1]);

      if (begin < end) {
         std::sort(rawResult + begin, rawResult + end);

         KeyT previous = rawResult[begin];
         rawResult[output++] = previous;

         for (size_t i = begin + 1; i < end; ++i) {
            KeyT current = rawResult[i];
            if (previous < current || current < previous) {
               rawResult[output++] = current;
            }
            previous = current;
         }
      }
   }

   if (offsets.size() > 0) {
      rawResultOffsets[numSegments] = static_cast<OffsetT>(output);
   }

   result.realloc(output);

   uniqueKeys.free();
   uniqueOffsets.free();
   uniqueKeys = std::move(result);
   uniqueOffsets = std::move(resultOffsets);
}

/**
 * @brief Replace segmented keys and offsets with sorted unique results.
 */
template <typename KeyT, typename OffsetT>
void segmented_sort_and_unique(
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

} // namespace care::host

#endif // CARE_HOST_SORT_H
