//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_HIP_SEGMENTED_SCAN_H
#define CARE_HIP_SEGMENTED_SCAN_H

#include "care/CHAIDataGetter.h"
#include "care/DefaultMacros.h"
#include "care/host_device_ptr.h"

#include <cstddef>
#include <utility>

#include "rocprim/rocprim.hpp"

namespace care::hip {

/**
 * @brief Perform an in-place exclusive scan independently within each segment.
 * @param values Values to scan and replace with the exclusive scan results.
 * @param offsets Segment boundaries. For N segments, offsets must contain
 * N + 1 entries: offsets[i] begins segment i, and offsets[N] marks the end of
 * the final segment. Segment i is therefore [offsets[i], offsets[i + 1]).
 * The entries must form a nondecreasing sequence from 0 to values.size(); that
 * is, offsets[0] must be 0 and offsets[N] must equal values.size(). Repeated
 * entries denote empty segments.
 * @param initialValue Initial value assigned to the first item of each segment.
 * @param binaryOp Associative binary operation used to perform the scan.
 */
template <typename ValueT, typename OffsetT, typename BinaryOp>
CARE_INLINE void segmented_exclusive_scan(
   care::host_device_ptr<ValueT>& values,
   care::host_device_ptr<OffsetT> const& offsets,
   ValueT initialValue,
   BinaryOp binaryOp)
{
   const size_t numSegments = offsets.size() > 0 ? offsets.size() - 1 : 0;
   const size_t numItems = values.size();

   if (numSegments == 0 || numItems == 0) {
      return;
   }

   CHAIDataGetter<ValueT, RAJADeviceExec> valueGetter {};
   auto* rawValues = valueGetter.getRawArrayData(values);

   CHAIDataGetter<OffsetT, RAJADeviceExec> offsetGetter {};
   const auto* rawOffsets = offsetGetter.getConstRawArrayData(offsets);

   care::host_device_ptr<ValueT> result(numItems);
   auto* rawResult = valueGetter.getRawArrayData(result);

   size_t tempStorageBytes = 0;
   rocprim::segmented_exclusive_scan(nullptr, tempStorageBytes,
                                     rawValues, rawResult,
                                     static_cast<unsigned int>(numSegments),
                                     rawOffsets, rawOffsets + 1,
                                     initialValue, binaryOp);

   CHAIDataGetter<char, RAJADeviceExec> charGetter {};
   care::host_device_ptr<char> tempStorage(tempStorageBytes);
   auto* rawTempStorage = charGetter.getRawArrayData(tempStorage);
   rocprim::segmented_exclusive_scan(rawTempStorage, tempStorageBytes,
                                     rawValues, rawResult,
                                     static_cast<unsigned int>(numSegments),
                                     rawOffsets, rawOffsets + 1,
                                     initialValue, binaryOp);

   tempStorage.free();

   if (values.isSlice()) {
      care::host_device_ptr<const ValueT> source = result;

      CARE_STREAM_LOOP(i, 0, numItems) {
         values[i] = source[i];
      } CARE_STREAM_LOOP_END

      result.free();
   } else {
      values.free();
      values = std::move(result);
   }
}

/**
 * @brief Perform an in-place exclusive sum independently within each segment.
 * @param values Values to scan and replace with the exclusive sums.
 * @param offsets Segment boundaries. For N segments, offsets must contain
 * N + 1 entries: offsets[i] begins segment i, and offsets[N] marks the end of
 * the final segment. Segment i is therefore [offsets[i], offsets[i + 1]).
 * The entries must form a nondecreasing sequence from 0 to values.size(); that
 * is, offsets[0] must be 0 and offsets[N] must equal values.size(). Repeated
 * entries denote empty segments.
 * @param initialValue Initial value assigned to the first item of each segment.
 */
template <typename ValueT, typename OffsetT>
CARE_INLINE void segmented_exclusive_scan(
   care::host_device_ptr<ValueT>& values,
   care::host_device_ptr<OffsetT> const& offsets,
   ValueT initialValue)
{
   segmented_exclusive_scan(values, offsets, initialValue,
                            rocprim::plus<ValueT> {});
}

/**
 * @brief Perform an in-place exclusive sum with an initial value of zero
 * independently within each segment.
 * @param values Values to scan and replace with the exclusive sums.
 * @param offsets Segment boundaries. For N segments, offsets must contain
 * N + 1 entries: offsets[i] begins segment i, and offsets[N] marks the end of
 * the final segment. Segment i is therefore [offsets[i], offsets[i + 1]).
 * The entries must form a nondecreasing sequence from 0 to values.size(); that
 * is, offsets[0] must be 0 and offsets[N] must equal values.size(). Repeated
 * entries denote empty segments.
 */
template <typename ValueT, typename OffsetT>
CARE_INLINE void segmented_exclusive_scan(
   care::host_device_ptr<ValueT>& values,
   care::host_device_ptr<OffsetT> const& offsets)
{
   segmented_exclusive_scan(values, offsets, ValueT {});
}

} // namespace care::hip

#endif // CARE_HIP_SEGMENTED_SCAN_H
