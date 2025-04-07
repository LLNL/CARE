//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

// This header should be included once for each type, defined by CARE_TEMPLATE_ARRAY_TYPE,
// in the instantiation header that is included by the host code.
// The instantiation header should be included in one compilation unit with
// CARE_INSTANTIATE defined. All other compilation units should include it
// without CARE_INSTANTIATE set.

#ifndef CARE_TEMPLATE_ARRAY_TYPE
#error "CARE_TEMPLATE_ARRAY_TYPE must be defined"
#endif
#ifndef CARE_TEMPLATE_KEY_TYPE
#error "CARE_TEMPLATE_KEY_TYPE must be defined"
#endif

#include "care/KeyValueSorter_decl.h"

#ifdef CARE_EXTERN
#undef CARE_EXTERN
#endif

#ifdef CARE_INSTANTIATE
#define CARE_EXTERN
#else
#define CARE_EXTERN extern
#endif

namespace care {

   CARE_EXTERN template CARE_DLL_API void setKeyValueArraysFromArray(host_device_ptr<_kv<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE> > &, const size_t, const CARE_TEMPLATE_ARRAY_TYPE*);
   CARE_EXTERN template CARE_DLL_API void setKeyValueArraysFromManagedArray(host_device_ptr<_kv<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE> > &, const size_t, const host_device_ptr<const CARE_TEMPLATE_ARRAY_TYPE>&);
   CARE_EXTERN template CARE_DLL_API size_t eliminateKeyValueDuplicates(host_device_ptr<_kv<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE> > &, const size_t);
   CARE_EXTERN template CARE_DLL_API void initializeKeyArray(host_device_ptr<CARE_TEMPLATE_KEY_TYPE>&, const host_device_ptr<const _kv<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE> >&, const size_t);
   CARE_EXTERN template CARE_DLL_API void initializeValueArray(host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE>&, const host_device_ptr<const _kv<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE> >&, const size_t);

#if !CARE_ENABLE_GPU_SIMULATION_MODE
   CARE_EXTERN template class CARE_DLL_API KeyValueSorter<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE, RAJA::seq_exec>;

   CARE_EXTERN template CARE_DLL_API void IntersectKeyValueSorters(RAJA::seq_exec, KeyValueSorter<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE, RAJA::seq_exec>, int, KeyValueSorter<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE, RAJA::seq_exec>, int, host_device_ptr<CARE_TEMPLATE_KEY_TYPE> &, host_device_ptr<CARE_TEMPLATE_KEY_TYPE> &, int &);
#endif // !CARE_ENABLE_GPU_SIMULATION_MODE

#if defined(CARE_PARALLEL_DEVICE) || CARE_ENABLE_GPU_SIMULATION_MODE

   CARE_EXTERN template CARE_DLL_API void setKeyValueArraysFromArray(host_device_ptr<CARE_TEMPLATE_KEY_TYPE> &, host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE> &, const size_t, const CARE_TEMPLATE_ARRAY_TYPE*);
   CARE_EXTERN template CARE_DLL_API void setKeyValueArraysFromManagedArray(host_device_ptr<CARE_TEMPLATE_KEY_TYPE> &, host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE> &, const size_t, const host_device_ptr<const CARE_TEMPLATE_ARRAY_TYPE>&);
   CARE_EXTERN template CARE_DLL_API size_t eliminateKeyValueDuplicates(host_device_ptr<CARE_TEMPLATE_KEY_TYPE>&, host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE>&, const host_device_ptr<const CARE_TEMPLATE_KEY_TYPE>&, const host_device_ptr<const CARE_TEMPLATE_ARRAY_TYPE>&, const size_t);
   CARE_EXTERN template CARE_DLL_API void sortKeyValueArrays<RAJADeviceExec, CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE>(host_device_ptr<CARE_TEMPLATE_KEY_TYPE> &, host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE> &, const size_t, const size_t, const bool);

   CARE_EXTERN template class CARE_DLL_API KeyValueSorter<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE, RAJADeviceExec>;

   CARE_EXTERN template CARE_DLL_API void IntersectKeyValueSorters(RAJADeviceExec, KeyValueSorter<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE, RAJADeviceExec>, int, KeyValueSorter<CARE_TEMPLATE_KEY_TYPE, CARE_TEMPLATE_ARRAY_TYPE, RAJADeviceExec>, int, host_device_ptr<CARE_TEMPLATE_KEY_TYPE> &, host_device_ptr<CARE_TEMPLATE_KEY_TYPE> &, int &);

#endif // defined(CARE_PARALLEL_DEVICE) || CARE_ENABLE_GPU_SIMULATION_MODE

} // namespace care

#undef CARE_TEMPLATE_ARRAY_TYPE
#undef CARE_TEMPLATE_KEY_TYPE

