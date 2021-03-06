// This header should be included once for each type, defined by CARE_TEMPLATE_ARRAY_TYPE,
// in the instantiation header that is included by the host code.
// The instantiation header should be included in one compilation unit with
// CARE_INSTANTIATE defined. All other compilation units should include it
// without CARE_INSTANTIATE set.

#ifndef CARE_TEMPLATE_ARRAY_TYPE
#error "CARE_TEMPLATE_ARRAY_TYPE must be defined"
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

   CARE_EXTERN template CARE_DLL_API void setKeyValueArraysFromArray(host_device_ptr<_kv<CARE_TEMPLATE_ARRAY_TYPE> > &, const size_t, const CARE_TEMPLATE_ARRAY_TYPE*);
   CARE_EXTERN template CARE_DLL_API void setKeyValueArraysFromManagedArray(host_device_ptr<_kv<CARE_TEMPLATE_ARRAY_TYPE> > &, const size_t, const host_device_ptr<const CARE_TEMPLATE_ARRAY_TYPE>&);
   CARE_EXTERN template CARE_DLL_API size_t eliminateKeyValueDuplicates(host_device_ptr<_kv<CARE_TEMPLATE_ARRAY_TYPE> > &, const size_t);
   CARE_EXTERN template CARE_DLL_API void initializeKeyArray(host_device_ptr<size_t>&, const host_device_ptr<const _kv<CARE_TEMPLATE_ARRAY_TYPE> >&, const size_t);
   CARE_EXTERN template CARE_DLL_API void initializeValueArray(host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE>&, const host_device_ptr<const _kv<CARE_TEMPLATE_ARRAY_TYPE> >&, const size_t);

   CARE_EXTERN template CARE_DLL_API class KeyValueSorter<CARE_TEMPLATE_ARRAY_TYPE, RAJA::seq_exec>;

   CARE_EXTERN template CARE_DLL_API void IntersectKeyValueSorters(RAJA::seq_exec, KeyValueSorter<CARE_TEMPLATE_ARRAY_TYPE, RAJA::seq_exec>, int, KeyValueSorter<CARE_TEMPLATE_ARRAY_TYPE, RAJA::seq_exec>, int, host_device_ptr<int> &, host_device_ptr<int>&, int &);

#ifdef CARE_GPUCC

   CARE_EXTERN template CARE_DLL_API void setKeyValueArraysFromArray(host_device_ptr<size_t> &, host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE> &, const size_t, const CARE_TEMPLATE_ARRAY_TYPE*);
   CARE_EXTERN template CARE_DLL_API void setKeyValueArraysFromManagedArray(host_device_ptr<size_t> &, host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE> &, const size_t, const host_device_ptr<const CARE_TEMPLATE_ARRAY_TYPE>&);
   CARE_EXTERN template CARE_DLL_API size_t eliminateKeyValueDuplicates(host_device_ptr<size_t>&, host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE>&, const host_device_ptr<const size_t>&, const host_device_ptr<const CARE_TEMPLATE_ARRAY_TYPE>&, const size_t);
   CARE_EXTERN template CARE_DLL_API void sortKeyValueArrays<size_t, CARE_TEMPLATE_ARRAY_TYPE, RAJADeviceExec>(host_device_ptr<size_t> &, host_device_ptr<CARE_TEMPLATE_ARRAY_TYPE> &, const size_t, const size_t, const bool);

   CARE_EXTERN template CARE_DLL_API class KeyValueSorter<CARE_TEMPLATE_ARRAY_TYPE, RAJADeviceExec>;

   CARE_EXTERN template CARE_DLL_API void IntersectKeyValueSorters(RAJADeviceExec, KeyValueSorter<CARE_TEMPLATE_ARRAY_TYPE, RAJADeviceExec>, int, KeyValueSorter<CARE_TEMPLATE_ARRAY_TYPE, RAJADeviceExec>, int, host_device_ptr<int> &, host_device_ptr<int>&, int &);

#endif // defined(CARE_GPUCC)

} // namespace care

#undef CARE_TEMPLATE_ARRAY_TYPE

