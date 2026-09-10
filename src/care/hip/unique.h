//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HIP_UNIQUE_H
#define CARE_HIP_UNIQUE_H

#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/device/unique.h"

#include <cstddef>

#include "rocprim/rocprim.hpp"

namespace care::hip {

/**
 * @brief Remove adjacent duplicate keys from a sorted array.
 * @param keys Sorted keys to compact in place. The compacted keys occupy the
 * first returned-number entries; the allocation and size are unchanged.
 * @param binaryPredicate Returns true when two adjacent keys are equivalent.
 * It must be callable on the device.
 * @return The number of unique keys.
 */
template <typename KeyT, typename BinaryPredicate>
CARE_INLINE size_t unique(care::host_device_ptr<KeyT>& keys,
                          BinaryPredicate binaryPredicate)
{
   const size_t numItems = keys.size();
   if (numItems == 0) {
      return 0;
   }

   CHAIDataGetter<KeyT, RAJADeviceExec> keyGetter {};
   auto* rawKeys = keyGetter.getRawArrayData(keys);
   care::host_device_ptr<KeyT> result(numItems);
   auto* rawResult = keyGetter.getRawArrayData(result);

   care::host_device_ptr<size_t> numUnique(1);
   CHAIDataGetter<size_t, RAJADeviceExec> countGetter {};
   auto* rawNumUnique = countGetter.getRawArrayData(numUnique);

   size_t tempStorageBytes = 0;
   rocprim::unique(nullptr, tempStorageBytes, rawKeys, rawResult,
                   rawNumUnique, numItems, binaryPredicate);

   CHAIDataGetter<char, RAJADeviceExec> charGetter {};
   care::host_device_ptr<char> tempStorage(tempStorageBytes);
   auto* rawTempStorage = charGetter.getRawArrayData(tempStorage);
   rocprim::unique(rawTempStorage, tempStorageBytes, rawKeys, rawResult,
                   rawNumUnique, numItems, binaryPredicate);

   size_t numUniqueKeys = 0;
   numUnique.pick(0, numUniqueKeys);

   tempStorage.free();
   numUnique.free();

   care::host_device_ptr<const KeyT> source = result;
   CARE_STREAM_LOOP(i, 0, numUniqueKeys) {
      keys[i] = source[i];
   } CARE_STREAM_LOOP_END

   result.free();
   return numUniqueKeys;
}

/**
 * @brief Remove adjacent duplicate keys from a sorted array using equality
 * comparison.
 * @param keys Sorted keys to compact in place. The compacted keys occupy the
 * first returned-number entries; the allocation and size are unchanged.
 * @return The number of unique keys.
 */
template <typename KeyT>
CARE_INLINE size_t unique(care::host_device_ptr<KeyT>& keys)
{
   return care::hip::unique(keys,
      [] CARE_HOST_DEVICE (KeyT const& left, KeyT const& right) {
         return left == right;
      });
}

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
 * It must be callable on the device.
 */
template <typename KeyT, typename OffsetT, typename BinaryPredicate>
CARE_INLINE void segmented_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets,
   BinaryPredicate binaryPredicate)
{
   care::device::segmented_unique(keys, offsets, binaryPredicate);
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
CARE_INLINE void segmented_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets)
{
   segmented_unique(keys, offsets,
      [] CARE_HOST_DEVICE (KeyT const& left, KeyT const& right) {
         return left == right;
      });
}

} // namespace care::hip

#endif // CARE_HIP_UNIQUE_H
