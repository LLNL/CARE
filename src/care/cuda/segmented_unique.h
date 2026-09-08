//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_CUDA_SEGMENTED_UNIQUE_H
#define CARE_CUDA_SEGMENTED_UNIQUE_H

#include "care/device/segmented_unique.h"

namespace care::cuda {

template <typename KeyT, typename OffsetT>
CARE_INLINE void segmented_unique(
   care::host_device_ptr<KeyT>& keys,
   care::host_device_ptr<OffsetT>& offsets)
{
   care::device::segmented_unique(keys, offsets);
}

} // namespace care::cuda

#endif // CARE_CUDA_SEGMENTED_UNIQUE_H
