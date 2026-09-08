//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_KEY_VALUE_SORTER_DECL_H_
#define _CARE_KEY_VALUE_SORTER_DECL_H_

// CARE config header
#include "care/config.h"

#include "care/algorithm_decl.h"
#include "care/CHAIDataGetter.h"

#include "care/scan.h"

// Other library headers
#ifdef CARE_GPUCC
#if defined(__CUDACC__)
#include "cub/cub.cuh"
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
#endif

#if defined(__HIPCC__)
#include "hipcub/hipcub.hpp"
#endif
#endif

#include <utility> // For std::move

namespace care {

///////////////////////////////////////////////////////////////////////////
/// @class KeyValueSorter
/// @author Peter Robinson, Alan Dayton
/// @brief Sorts and unsorts arrays.
/// KeyValue Sorter is an Object Oriented take on the legacy
///    _intSorter / _floatSorter / _gidResorter stuff.
/// Currently we have a CUDA and a sequential partial specialization of
///    this template class. Templating rather than inheritance is used
///    to make this GPU friendly.
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType, typename Exec=RAJAExec>
class CARE_KEY_VALUE_SORTER_DLL_API KeyValueSorter;

/// LocalKeyValueSorter should be used as the type for HOSTDEV functions
/// to indicate that the function should only be called in a RAJA context.
/// This will prevent the clang-check from checking for functions that
/// should not be called outside a lambda context.
/// Note that this does not actually enforce that the HOSTDEV function
/// is only called from RAJA loops.
template <typename KeyType, typename ValueType, typename Exec>
using LocalKeyValueSorter = KeyValueSorter<KeyType, ValueType, Exec> ;

template <typename KeyValueType>
inline bool cmpKeys(KeyValueType const & left, KeyValueType const & right);

template <typename KeyValueType>
inline bool cmpKeysThenValues(KeyValueType const & left, KeyValueType const & right);

namespace detail {
   template <typename KeyT, typename ValueT>
   CARE_INLINE void stableSortKeyValuePairs(host_device_ptr<KeyT> & keys,
                                            host_device_ptr<ValueT> & values,
                                            const size_t len,
                                            const size_t start = 0) {

      host_device_ptr<_kv<KeyT,ValueT>> keyValues(len);

      CARE_SEQUENTIAL_LOOP(i, 0, (int) len) {
         keyValues[i].key = keys[i+start];
         keyValues[i].value = values[i+start];
      } CARE_SEQUENTIAL_LOOP_END

      CHAIDataGetter<_kv<KeyT, ValueT>, RAJA::seq_exec> getter {};
      _kv<KeyT, ValueT> * rawData = getter.getRawArrayData(keyValues);
      std::stable_sort(rawData, rawData + len, cmpKeys<_kv<KeyT,ValueT>>);

      CARE_SEQUENTIAL_LOOP(i, 0, (int) len) {
         keys[i+start] = keyValues[i].key;
         values[i+start] = keyValues[i].value;
      } CARE_SEQUENTIAL_LOOP_END

      keyValues.free();
   }
} // namespace detail



///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson, Alan Dayton
/// @brief ManagedArray API for sorting paired key and value arrays
/// @param[in, out] keys   - The array to sort
/// @param[in, out] values - The array that is sorted simultaneously
/// @param[in]      start  - The index to start sorting at
/// @param[in]      len    - The number of elements to sort
/// @param[in]      noCopy - Whether or not to copy the result into the
///                             original arrays or simply replace the
///                             original arrays. Should be false if only
///                             sorting part of the arrays or you will
///                             have bugs!
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename Exec, typename KeyT, typename ValueT>
std::enable_if_t<std::is_arithmetic<typename CHAIDataGetter<KeyT, Exec>::raw_type>::value, void>
inline sortKeyValueArrays(host_device_ptr<KeyT> & keys,
                          host_device_ptr<ValueT> & values,
                          const size_t start, const size_t len,
                          const bool noCopy=false)
{
   bool _noCopy ;
   if (noCopy && start > 0) {
      printf("[CARE] Warning: sortKeyValueArrays. noCopy should not be set if start > 0 (%d)\n", (int)start);
      _noCopy = false;
   }
   else {
      _noCopy = noCopy;
   }

   if constexpr (std::is_same_v<Exec, RAJA::seq_exec>) {
      detail::stableSortKeyValuePairs(keys, values, len, start);
   }
   else {
      // TODO openMP parallel implementation
#if defined(__HIPCC__) || (defined(__CUDACC__) && defined(CUB_MAJOR_VERSION) && defined(CUB_MINOR_VERSION) && (CUB_MAJOR_VERSION >= 2 || (CUB_MAJOR_VERSION == 1 && CUB_MINOR_VERSION >= 14)))

      // Allocate space for the result
      host_device_ptr<KeyT> keyResult{len};
      host_device_ptr<ValueT> valueResult{len};

      // Get the raw data to pass to cub
      CHAIDataGetter<KeyT, Exec> keyGetter {};
      CHAIDataGetter<ValueT, Exec> valueGetter {};

      auto * rawKeyData = keyGetter.getRawArrayData(keys) + start;
      auto * rawValueData = valueGetter.getRawArrayData(values) + start;

      auto * rawKeyResult = keyGetter.getRawArrayData(keyResult);
      auto * rawValueResult = valueGetter.getRawArrayData(valueResult);

      // Get the temp storage length
      char * d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;

      // When called with a nullptr for temp storage, this returns how much
      // temp storage should be allocated.
      if (len > 0) {
#if defined(__CUDACC__)
         cub::DeviceRadixSort::SortPairs((void *)d_temp_storage, temp_storage_bytes,
                                         rawKeyData, rawKeyResult,
                                         rawValueData, rawValueResult,
                                         len);
#elif defined(__HIPCC__)
         hipcub::DeviceRadixSort::SortPairs((void *)d_temp_storage, temp_storage_bytes,
                                            rawKeyData, rawKeyResult,
                                            rawValueData, rawValueResult,
                                            len);
#endif
      }

      // Allocate the temp storage and get raw data to pass to cub
      host_device_ptr<char> tmpManaged {temp_storage_bytes};

      CHAIDataGetter<char, Exec> charGetter {};
      d_temp_storage = charGetter.getRawArrayData(tmpManaged);

      // Now sort
      if (len > 0) {
#if defined(CHAI_THIN_GPU_ALLOCATE)
         chai::ArrayManager::getInstance()->setExecutionSpace(chai::GPU);
#endif

#if defined(__CUDACC__)
         cub::DeviceRadixSort::SortPairs((void *)d_temp_storage, temp_storage_bytes,
                                         rawKeyData, rawKeyResult,
                                         rawValueData, rawValueResult,
                                         len);
#elif defined(__HIPCC__)
         hipcub::DeviceRadixSort::SortPairs((void *)d_temp_storage, temp_storage_bytes,
                                            rawKeyData, rawKeyResult,
                                            rawValueData, rawValueResult,
                                            len);
#endif

#if defined(CHAI_THIN_GPU_ALLOCATE)
         chai::ArrayManager::getInstance()->setExecutionSpace(chai::NONE);
#endif

         tmpManaged.free();
      }

      // Get the result
      if (_noCopy) {
         if (len > 0) {
            keys.free();
            values.free();
         }

         keys = keyResult;
         values = valueResult;
      }
      else {
         CARE_STREAM_LOOP(i, 0, len) {
            keys[i+start] = keyResult[i];
            values[i+start] = valueResult[i];
         } CARE_STREAM_LOOP_END

         if (len > 0) {
            keyResult.free();
            valueResult.free();
         }
      }

#else // defined(CARE_GPUCC)
      detail::stableSortKeyValuePairs(keys, values, len, start);
#endif // defined(CARE_GPUCC)
   }

}

template <typename Exec, typename KeyT, typename ValueT>
std::enable_if_t<!std::is_arithmetic<typename CHAIDataGetter<KeyT, Exec>::raw_type>::value, void>
inline sortKeyValueArrays(host_device_ptr<KeyT> & keys,
                          host_device_ptr<ValueT> & values,
                          const size_t start, const size_t len,
                          const bool noCopy=false)
{
   bool _noCopy ;
   if (noCopy && start > 0) {
      printf("[CARE] Warning: sortKeyValueArrays. noCopy should not be set if start > 0 (%d)\n", (int)start);
      _noCopy = false;
   }
   else {
      _noCopy = noCopy;
   }

   if constexpr (std::is_same_v<Exec, RAJA::seq_exec>) {
      detail::stableSortKeyValuePairs(keys, values, len, start);
   }
   else {
      // TODO openMP parallel implementation
#if defined(__HIPCC__) || (defined(__CUDACC__) && defined(CUB_MAJOR_VERSION) && defined(CUB_MINOR_VERSION) && (CUB_MAJOR_VERSION >= 2 || (CUB_MAJOR_VERSION == 1 && CUB_MINOR_VERSION >= 14)))

      // Allocate space for the result
      host_device_ptr<KeyT> keyResult{len};
      host_device_ptr<ValueT> valueResult{len};

      // Get the raw data to pass to cub
      CHAIDataGetter<KeyT, Exec> keyGetter {};
      CHAIDataGetter<ValueT, Exec> valueGetter {};

      auto * rawKeyData = keyGetter.getRawArrayData(keys) + start;
      auto * rawValueData = valueGetter.getRawArrayData(values) + start;

      auto * rawKeyResult = keyGetter.getRawArrayData(keyResult);
      auto * rawValueResult = valueGetter.getRawArrayData(valueResult);

      using RawKeyType = std::remove_reference_t<decltype(*rawKeyData)>;

      auto custom_comparator = [] CARE_HOST_DEVICE (const RawKeyType& lhs,
                                                    const RawKeyType& rhs) {
         return lhs < rhs;
      };

      // Get the temp storage length
      char * d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;

      // When called with a nullptr for temp storage, this returns how much
      // temp storage should be allocated.
      if (len > 0) {
#if defined(__CUDACC__)
         cub::DeviceMergeSort::StableSortPairs((void *)d_temp_storage, temp_storage_bytes,
                                               rawKeyData,
                                               rawValueData,
                                               len, custom_comparator);
#elif defined(__HIPCC__)
         hipcub::DeviceMergeSort::StableSortPairs((void *)d_temp_storage, temp_storage_bytes,
                                                  rawKeyData,
                                                  rawValueData,
                                                  len, custom_comparator);
#endif
      }

      // Allocate the temp storage and get raw data to pass to cub
      host_device_ptr<char> tmpManaged {temp_storage_bytes};

      CHAIDataGetter<char, Exec> charGetter {};
      d_temp_storage = charGetter.getRawArrayData(tmpManaged);

      // Now sort
      if (len > 0) {
#if defined(CHAI_THIN_GPU_ALLOCATE)
         chai::ArrayManager::getInstance()->setExecutionSpace(chai::GPU);
#endif

#if defined(__CUDACC__)
         cub::DeviceMergeSort::StableSortPairs((void *)d_temp_storage, temp_storage_bytes,
                                               rawKeyData,
                                               rawValueData,
                                               len,
                                               custom_comparator);
#elif defined(__HIPCC__)
         hipcub::DeviceMergeSort::StableSortPairs((void *)d_temp_storage, temp_storage_bytes,
                                                  rawKeyData,
                                                  rawValueData,
                                                  len,
                                                  custom_comparator);
#endif

#if defined(CHAI_THIN_GPU_ALLOCATE)
         chai::ArrayManager::getInstance()->setExecutionSpace(chai::NONE);
#endif

         tmpManaged.free();
      }

      // merge sort did an inplace sort, so the answer is already in keys and Values
      if (len > 0) {
         keyResult.free();
         valueResult.free();
      }

#else // defined(CARE_GPUCC)
      detail::stableSortKeyValuePairs(keys, values, len, start);
#endif // defined(CARE_GPUCC)
   }

}

#if defined(CARE_PARALLEL_DEVICE) || CARE_ENABLE_GPU_SIMULATION_MODE
///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes keys and values by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to copy
/// @param[in] arr - input array
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
void setKeyValueArraysFromArray(host_device_ptr<KeyType> & keys, host_device_ptr<ValueType> & values,
                                const size_t len, const ValueType* arr) ;

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes the KeyValueSorter by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
void setKeyValueArraysFromManagedArray(host_device_ptr<KeyType> & keys, host_device_ptr<ValueType> & values,
                                       const size_t len, const host_device_ptr<const ValueType>& arr) ;

///////////////////////////////////////////////////////////////////////////
/// @author Jeff Keasler, Alan Dayton
/// @brief Eliminates duplicate values
/// Remove duplicate values from old key/value arrays.
/// Old key/value arrays should already be sorted by value.
/// New key/value arrays should be allocated to the old size.
/// @param[out] newKeys New key array with duplicates removed
/// @param[out] newValues New value array with duplicates removed
/// @param[in] oldKeys Old key array (key-value pairs sorted by value)
/// @param[in] oldValues Old value array (sorted)
/// @param[in] oldLen Length of old key/value array and initial length for new
/// @return Length of new key/value arrays
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
size_t eliminateKeyValueDuplicates(host_device_ptr<KeyType>& newKeys,
                                   host_device_ptr<ValueType>& newValues,
                                   const host_device_ptr<const KeyType>& oldKeys,
                                   const host_device_ptr<const ValueType>& oldValues,
                                   const size_t oldLen);

///////////////////////////////////////////////////////////////////////////
/// GPU partial specialization of KeyValueSorter
/// The GPU version of KeyValueSorter stores keys and values as separate
///    arrays to be compatible with sortKeyValueArrays.
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
class CARE_KEY_VALUE_SORTER_DLL_API KeyValueSorter<KeyType, ValueType, RAJADeviceExec> {
   public:

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Default constructor
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE KeyValueSorter() noexcept {}

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson, Alan Dayton
      /// @brief Constructor
      /// Allocates space for the given number of elements
      /// @param[in] len - The number of elements to allocate space for
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      explicit KeyValueSorter(const size_t len)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(len, "m_keys")
      , m_values(len, "m_values")
      {
      };

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Constructor
      /// Allocates space and initializes the KeyValueSorter by copying
      ///    elements and ordering from the given raw array
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - The raw array to copy elements from
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter(const size_t len, const ValueType* arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(len, "m_keys")
      , m_values(len, "m_values")
      {
         setKeyValueArraysFromArray(m_keys, m_values, len, arr);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Constructor
      /// Allocates space and initializes the KeyValueSorter by copying
      ///    elements and ordering from the given managed array
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - The managed array to copy elements from
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter(const size_t len, const host_device_ptr<const ValueType> & arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(len, "m_keys")
      , m_values(len, "m_values")
      {
         setKeyValueArraysFromManagedArray(m_keys, m_values, len, arr);
      }
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Constructor
      /// Takes ownership of the provided keys and values arrays
      /// @param[in] len - The number of elements in the arrays
      /// @param[in] keys - The keys array to take ownership of
      /// @param[in] values - The values array to take ownership of
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter(const size_t len, host_device_ptr<KeyType> && keys, host_device_ptr<ValueType> && values)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(std::move(keys))
      , m_values(std::move(values))
      {
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief (Shallow) Copy constructor
      /// Does a shallow copy and indicates that the copy should NOT free
      ///    the underlying memory. This must be a shallow copy because it is
      ///    called upon lambda capture, and upon exiting the scope of a lambda
      ///    capture, the copy must NOT free the underlying memory.
      /// @param[in] other - The other KeyValueSorter to copy from
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE KeyValueSorter(const KeyValueSorter& other)
      : m_len(other.m_len)
      , m_ownsPointers(false)
      , m_keys(other.m_keys)
      , m_values(other.m_values)
      {
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Destructor
      /// Frees the underlying memory if this is the owner.
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE ~KeyValueSorter()
      {
#ifndef CARE_DEVICE_COMPILE
         /// Only attempt to free if we are on the CPU
         free();
#endif
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief (Shallow) Copy assignment operator
      /// Does a shallow copy and indicates that the copy should NOT free
      ///    the underlying memory.
      /// @param[in] other - The other KeyValueSorter to copy from
      /// @return *this
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter& operator=(KeyValueSorter& other)
      {
         if (this != &other) {
            free();

            m_len = other.m_len;
            m_ownsPointers = false;
            m_keys = other.m_keys;
            m_values = other.m_values;
         }

         return *this;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Move assignment operator
      /// Does a move, and therefore this may or may not own the underlying
      ///    memory.
      /// @param[in] other - The other KeyValueSorter to move from
      /// @return *this
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter& operator=(KeyValueSorter&& other)
      {
         if (this != &other) {
            free();

            m_len = other.m_len;
            m_ownsPointers = other.m_ownsPointers;
            m_keys = other.m_keys;
            m_values = other.m_values;

            other.m_len = 0;
            other.m_ownsPointers = false;
            other.m_keys = nullptr;
            other.m_values = nullptr;
         }

         return *this;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the key at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to get the key
      /// @return the key at the given index
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE KeyType key(const size_t index) const {
         local_ptr<const KeyType> keys = m_keys;
         return keys[index];
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sets the key at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to set the key
      /// @param[in] key   - The new key
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE void setKey(const size_t index, const KeyType key) const {
         local_ptr<KeyType> keys = m_keys;
         keys[index] = key;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the value at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to get the value
      /// @return the value at the given index
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE ValueType value(const size_t index) const {
         local_ptr<const ValueType> values = m_values;
         return values[index];
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sets the value at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to set the value
      /// @param[in] value - The new value
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE void setValue(const size_t index, const ValueType value) const {
         local_ptr<ValueType> values = m_values;
         values[index] = value;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the keys contained in the KeyValueSorter
      /// @return the keys contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE host_device_ptr<KeyType> & keys() {
         return m_keys;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets a const copy of the keys contained in the KeyValueSorter
      /// @return a const copy of the keys contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE const host_device_ptr<KeyType> & keys() const {
         return m_keys;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the values contained in the KeyValueSorter
      /// @return the values contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE host_device_ptr<ValueType> & values() {
         return m_values;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets a const copy of the values contained in the KeyValueSorter
      /// @return a const copy of the values contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE const host_device_ptr<ValueType> & values() const {
         return m_values;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the number of elements the KeyValueSorter is managing
      /// @return the number of elements the KeyValueSorter is managing
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE size_t len() const {
         return m_len;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts "len" elements starting at "start" by value
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to sort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void sort(const size_t start, const size_t len) {
         sortKeyValueArrays<RAJADeviceExec>(m_values, m_keys, start, len, false);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts the first "len" elements by value
      /// @param[in] len - The number of elements to sort
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sort(const size_t len) {
         sort(0, len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts all the elements
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sort() {
         sortKeyValueArrays<RAJADeviceExec>(m_values, m_keys, 0, m_len, true);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts "len" elements starting at "start" by key
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to sort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void sortByKey(const size_t start, const size_t len) {
         sortKeyValueArrays<RAJADeviceExec>(m_keys, m_values, start, len, false);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts the first "len" elements by key
      /// @param[in] len - The number of elements to sort
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sortByKey(const size_t len) {
         sortByKey(0, len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts all the elements by key
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sortByKey() {
         sortKeyValueArrays<RAJADeviceExec>(m_keys, m_values, 0, m_len, true);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts "len" elements starting at "start" by key, then by value
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to sort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void sortByKeyThenValue(const size_t start, const size_t len) {
         if (len <= 1) return;

         // First sort by key
         sortKeyValueArrays<RAJADeviceExec>(m_keys, m_values, start, len, false);

         // Phase 1: Identify ranges of identical keys
         host_device_ptr<int> rangeStarts(len+1);
         host_device_ptr<int> rangeEnds(len+1);

         int count = 0;

         auto keys = m_keys;
         
         // Use SCAN_LOOP to identify where ranges start
         SCAN_LOOP(i, start, start+len, idx, count,
                  (i == start) || (keys[i] != keys[i-1])) {
            rangeStarts[idx] = i;
         } SCAN_LOOP_END(start+len, idx, count)

         // Set the last range end
         rangeStarts.set(count , start+len);

         auto values = m_values;

         // Phase 2: Sort each range by value using insertion sort in parallel
         CARE_STREAM_LOOP(i, 0, count) {
            int rangeStart = rangeStarts[i];
            int rangeEnd = rangeStarts[i+1];
            int rangeLen = rangeEnd - rangeStart;

            // Only sort if range has more than one element
            if (rangeLen > 1) {
               // remmber that keys are identical over this range, so no need to modify m_keys
               InsertionSort<ValueType>(values.slice(rangeStart), rangeEnd-rangeStart);
            }
         } CARE_STREAM_LOOP_END

         // Free temporary arrays
         rangeStarts.free();
         rangeEnds.free();
      }


      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts the first "len" elements by key, then by value
      /// @param[in] len - The number of elements to sort
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sortByKeyThenValue(const size_t len) {
         sortByKeyThenValue(0, len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts all the elements by key, then by value
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sortByKeyThenValue() {
         sortByKeyThenValue(0, m_len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Does a stable sort on "len" elements starting at "start" by value
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to sort
      /// @return void
      /// TODO: investigate whether radix device sort is a stable sort
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void stableSort(const size_t start, const size_t len) {
         sortKeyValueArrays<RAJADeviceExec>(m_values, m_keys, start, len, false);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Does a stable sort on the first "len" elements by value
      /// @param[in] len - The number of elements to sort
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void stableSort(const size_t len) {
         stableSort(0, len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Does a stable sort on all the elements by value
      /// @return void
      /// TODO: investigate whether radix device sort is a stable sort
      ///////////////////////////////////////////////////////////////////////////
      void stableSort() {
         sortKeyValueArrays<RAJADeviceExec>(m_values, m_keys, 0, m_len, true);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Jeff Keasler, Alan Dayton
      /// @brief Eliminates duplicate values
      /// First does a stable sort based on the values, which preserves the
      ///    ordering in case of a tie. Then duplicates are removed. The final
      ///    step is to unsort.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void eliminateDuplicates() {
         if (m_len > 1) {
            // Do a STABLE sort by value.
            // I believe cub::DeviceRadixSort is a
            stableSort();

            // Allocate storage for the key value pairs without duplicates
            host_device_ptr<KeyType> newKeys{m_len, "newKeys"};
            host_device_ptr<ValueType> newValues{m_len, "newValues"};

            int newSize = eliminateKeyValueDuplicates(newKeys, newValues,
                                                      (host_device_ptr<const KeyType>)m_keys,
                                                      (host_device_ptr<const ValueType>)m_values,
                                                      m_len) ;

            // Free the original key value pairs
            free();

            // Set to new key value pairs
            m_keys = newKeys;
            m_values = newValues;
            m_len = newSize;

            // Restore original ordering
            sortByKey();
         }
      }
      
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Eliminates duplicate key-value pairs
      /// First does a sort by key and then by value, which groups identical pairs.
      /// Then duplicates are removed.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void eliminateDuplicatePairs() {
         if (m_len > 1) {
            // First sort by key and then by value to group identical pairs
            sortByKeyThenValue();
            
            // Allocate storage for tracking unique elements
            host_device_ptr<int> isUnique(m_len+1);
            
            // Mark unique elements (first element is always unique)
            auto len = m_len;
            auto keys = m_keys;
            auto values = m_values;
            CARE_STREAM_LOOP(i, 0, len+1) {
               if (i == 0) {
                  isUnique[i] = 1;
               }
               else if (i == len) {
                  isUnique[i] = 0;
               }
               else {
                  // Element is unique if it differs from the previous element
                  // in either key or value
                  isUnique[i] = (keys[i] != keys[i-1] ||
                                 values[i] != values[i-1]) ? 1 : 0;
               }
            } CARE_STREAM_LOOP_END
            
            // Use exclusive scan to compute output positions
            host_device_ptr<int> positions(m_len+1);
            exclusive_scan(RAJADeviceExec{}, isUnique, positions, m_len + 1, 0, false);
            
            // Get the total number of unique elements
            int newSize = positions.pick(m_len);
            
            // Allocate new arrays for the unique elements
            host_device_ptr<KeyType> newKeys(newSize, "newKeys");
            host_device_ptr<ValueType> newValues(newSize, "newValues");

            // Copy unique elements to their new positions
            CARE_STREAM_LOOP(i, 0, m_len) {
               if (isUnique[i]) {
                  int pos = positions[i];
                  newKeys[pos] = keys[i];
                  newValues[pos] = values[i];
               }
            } CARE_STREAM_LOOP_END
            
            // Free temporary arrays
            isUnique.free();
            positions.free();
            
            // Free the original key value pairs
            free();
            
            // Set to new key value pairs
            m_keys = newKeys;
            m_values = newValues;
            m_len = newSize;
         }
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// @brief no-op
      /// GPU version does not require separate allocation for keys array.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void initializeKeys() const {
         return;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// @brief no-op
      /// GPU version does not require separate allocation for values array.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void initializeValues() const {
         return;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// GPU version does not require separate allocation for keys array.
      /// @return true
      ///////////////////////////////////////////////////////////////////////////
      bool keysAllocated() const {
         return true ;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// GPU version does not require separate allocation for keys array.
      /// @return true
      ///////////////////////////////////////////////////////////////////////////
      bool valuesAllocated() const {
         return true ;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// @brief no-op
      /// GPU version does not require separate allocation for keys array.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void freeKeys() const {
         return;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// @brief no-op
      /// GPU version does not require separate allocation for values array.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void freeValues() const {
         return;
      }
   private:
      size_t m_len = 0;
      bool m_ownsPointers = false; /// Prevents memory from being freed by lambda captures
      host_device_ptr<KeyType> m_keys = nullptr;
      host_device_ptr<ValueType> m_values = nullptr;

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson, Alan Dayton
      /// @brief Frees the underlying memory if this is the owner
      /// Used by the destructor and by the assignment operators. Should be private.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      inline void free() {
         if (m_ownsPointers) {
            if (m_keys) {
               m_keys.free();
            }

            if (m_values) {
               m_values.free();
            }
         }
      }
};

#endif // defined(CARE_PARALLEL_DEVICE) || CARE_ENABLE_GPU_SIMULATION_MODE



///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson
/// @brief Less than comparison operator for values
/// Used as a comparator in the STL
/// @param left  - left _kv to compare
/// @param right - right _kv to compare
/// @return true if left's value is less than right's value, false otherwise
///////////////////////////////////////////////////////////////////////////
template <typename KeyValueType>
inline bool operator <(KeyValueType const & left, KeyValueType const & right)
{
   return left.value < right.value;
}

///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson
/// @brief Less than comparison operator for keys
/// Used as a comparator in the STL
/// @param left  - left _kv to compare
/// @param right - right _kv to compare
/// @return true if left's key is less than right's key, false otherwise
///////////////////////////////////////////////////////////////////////////
template <typename KeyValueType>
inline bool cmpKeys(KeyValueType const & left, KeyValueType const & right)
{
   return left.key < right.key;
}

///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson
/// @brief Less than comparison operator for keys, then values
/// Used as a comparator in the STL
/// @param left  - left _kv to compare
/// @param right - right _kv to compare
/// @return true if left's key is less than right's key, or if keys are equal
///         and left's value is less than right's value
///////////////////////////////////////////////////////////////////////////
template <typename KeyValueType>
inline bool cmpKeysThenValues(KeyValueType const & left, KeyValueType const & right)
{
   return (left.key < right.key) || 
          ((left.key == right.key) && (left.value < right.value));
}

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes keys and values by copying elements from the array
/// @param[out] keyValues - The key value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
void setKeyValueArraysFromArray(host_device_ptr<_kv<KeyType, ValueType>> & keyValues,
                                const size_t len, const ValueType* arr) ;

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes the KeyValueSorter by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
void setKeyValueArraysFromManagedArray(host_device_ptr<_kv<KeyType, ValueType>> & keyValues,
                                       const size_t len, const host_device_ptr<const ValueType>& arr) ;


///////////////////////////////////////////////////////////////////////////
/// @author Jeff Keasler, Alan Dayton
/// @brief Eliminates duplicate values
/// Assumes key value array sorted by values.
/// @param[in/out] keyValues - The key value array to eliminate duplicates in
/// @param[in/out] len - original length of key value array
/// @return new length of array
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
size_t eliminateKeyValueDuplicates(host_device_ptr<_kv<KeyType, ValueType>> & keyValues, const size_t len) ;

///////////////////////////////////////////////////////////////////////////
/// @author Alan Dayton
/// @brief Initializes the keys
/// The keys are stored in the managed array of _kv structs. To get the
/// keys separately, they must be copied into their own array.
/// @param[out] keys - The key array
/// @param[in] keyValues - The key value array
/// @param[in/out] len - length of key value array
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
void initializeKeyArray(host_device_ptr<KeyType>& keys, const host_device_ptr<const _kv<KeyType, ValueType>>& keyValues, const size_t len) ;

///////////////////////////////////////////////////////////////////////////
/// @author Alan Dayton
/// @brief Initializes the values
/// The values are stored in the managed array of _kv structs. To get the
///    values separately, they must be copied into their own array.
/// @param[out] values - The values array
/// @param[in] keyValues - The key value array
/// @param[in/out] len - length of key value array
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
void initializeValueArray(host_device_ptr<ValueType>& values, const host_device_ptr<const _kv<KeyType, ValueType> >& keyValues, const size_t len);

#if !CARE_ENABLE_GPU_SIMULATION_MODE
///////////////////////////////////////////////////////////////////////////
/// Sequential partial specialization of KeyValueSorter
/// The CPU implementation relies on routines that use the < operator on
/// a key-value struct.
/// TODO make a version of this that sorts indices as in:
/// https://stackoverflow.com/questions/3909272/sorting-two-corresponding-arrays
/// This has the advantage of having the same underlying data layout for keys
/// and values as the GPU version of the code, which in many instances removes
/// the need for copying the keys and values into separate arrays after the sort.
///////////////////////////////////////////////////////////////////////////
template <typename KeyType, typename ValueType>
class CARE_KEY_VALUE_SORTER_DLL_API KeyValueSorter<KeyType, ValueType, RAJA::seq_exec> {
   public:

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Default constructor
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter() {}

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson, Alan Dayton
      /// @brief Constructor
      /// Allocates space for the given number of elements
      /// @param[in] len - The number of elements to allocate space for
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      explicit KeyValueSorter(size_t len)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(nullptr)
      , m_values(nullptr)
      , m_keyValues(len, "m_keyValues")
      {
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Constructor
      /// Allocates space and initializes the KeyValueSorter by copying
      ///    elements and ordering from the given raw array
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - The raw array to copy elements from
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter(const size_t len, const ValueType* arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(nullptr)
      , m_values(nullptr)
      , m_keyValues(len, "m_keyValues")
      {
         setKeyValueArraysFromArray(m_keyValues, len, arr);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Constructor
      /// Allocates space and initializes the KeyValueSorter by copying
      ///    elements and ordering from the given managed array
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - The managed array to copy elements from
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter(const size_t len, const host_device_ptr<const ValueType> & arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(nullptr)
      , m_values(nullptr)
      , m_keyValues(len, "m_keyValues")
      {
         setKeyValueArraysFromManagedArray(m_keyValues, len, arr);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Constructor
      /// Takes ownership of the provided keys and values arrays
      /// @param[in] len - The number of elements in the arrays
      /// @param[in] keys - The keys array to take ownership of
      /// @param[in] values - The values array to take ownership of
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter(const size_t len, host_device_ptr<KeyType> && keys, host_device_ptr<ValueType> && values)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(std::move(keys))
      , m_values(std::move(values))
      , m_keyValues(len, "m_keyValues")
      {
         auto mkeyValues  = m_keyValues;
         auto mkeys = m_keys;
         auto mvalues = m_values;

         // Initialize m_keyValues from the provided keys and values
         CARE_SEQUENTIAL_LOOP(i, 0, m_len) {
            mkeyValues[i].key = mkeys[i];
            mkeyValues[i].value = mvalues[i];
         } CARE_SEQUENTIAL_LOOP_END
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief (Shallow) Copy constructor
      /// Does a shallow copy and indicates that the copy should NOT free
      ///    the underlying memory. This must be a shallow copy because it is
      ///    called upon lambda capture, and upon exiting the scope of a lambda
      ///    capture, the copy must NOT free the underlying memory.
      /// @param[in] other - The other KeyValueSorter to copy from
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE KeyValueSorter(const KeyValueSorter& other)
      : m_len(other.m_len)
      , m_ownsPointers(false)
      , m_keys(other.m_keys)
      , m_values(other.m_values)
      , m_keyValues(other.m_keyValues)
      {
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Destructor
      /// Frees the underlying memory if this is the owner.
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE ~KeyValueSorter()
      {
#ifndef CARE_DEVICE_COMPILE
         free();
#endif
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief (Shallow) Copy assignment operator
      /// Does a shallow copy and indicates that the copy should NOT free
      ///    the underlying memory.
      /// @param[in] other - The other KeyValueSorter to copy from
      /// @return *this
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter& operator=(KeyValueSorter& other)
      {
         if (this != &other) {
            free();

            m_len = other.m_len;
            m_ownsPointers = false;
            m_keys = other.m_keys;
            m_values = other.m_values;
            m_keyValues = other.m_keyValues;
         }

         return *this;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Move assignment operator
      /// Does a move, and therefore this may or may not own the underlying
      ///    memory.
      /// @param[in] other - The other KeyValueSorter to move from
      /// @return *this
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter& operator=(KeyValueSorter&& other)
      {
         if (this != &other) {
            free();

            m_len = other.m_len;
            m_ownsPointers = other.m_ownsPointers;
            m_keys = other.m_keys;
            m_values = other.m_values;
            m_keyValues = other.m_keyValues;

            other.m_len = 0;
            other.m_ownsPointers = false;
            other.m_keys = nullptr;
            other.m_values = nullptr;
            other.m_keyValues = nullptr;
         }

         return *this;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the key at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to get the key
      /// @return the key at the given index
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE KeyType key(const size_t index) const {
         local_ptr<_kv<KeyType,ValueType> > local_keyValues = m_keyValues;
         return local_keyValues[index].key;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sets the key at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to set the key
      /// @param[in] key   - The new key
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE void setKey(const size_t index, const KeyType key) const {
         local_ptr<_kv<KeyType, ValueType> > local_keyValues = m_keyValues;
         local_keyValues[index].key = key;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the value at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to get the value
      /// @return the value at the given index
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE ValueType value(const size_t index) const {
         local_ptr<_kv<KeyType, ValueType> > local_keyValues = m_keyValues;
         return local_keyValues[index].value;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sets the value at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to set the value
      /// @param[in] value - The new value
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE void setValue(const size_t index, const ValueType value) const {
         local_ptr<_kv<KeyType, ValueType> > local_keyValues = m_keyValues;
         local_keyValues[index].value = value;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the keys contained in the KeyValueSorter
      /// @return the keys contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      host_device_ptr<KeyType> & keys() {
         initializeKeys();
         return m_keys;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets a const copy of the keys contained in the KeyValueSorter
      /// @return a const copy of the keys contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      const host_device_ptr<KeyType> & keys() const {
         initializeKeys();
         return m_keys;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the values contained in the KeyValueSorter
      /// @return the values contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      host_device_ptr<ValueType> & values() {
         initializeValues();
         return m_values;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets a const copy of the values contained in the KeyValueSorter
      /// @return a const copy of the values contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      const host_device_ptr<ValueType> & values() const {
         initializeValues();
         return m_values;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the number of elements the KeyValueSorter is managing
      /// @return the number of elements the KeyValueSorter is managing
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE size_t len() const {
         return m_len;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts "len" elements starting at "start" by value
      /// Calls std::stable_sort, which uses _kv::operator< to do the comparisons
      /// @note stable_sort is used for consistency with GPU implementation
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to sort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void sort(const size_t start, const size_t len) const {
         CHAIDataGetter<_kv<KeyType, ValueType>, RAJA::seq_exec> getter {};
         _kv<KeyType, ValueType> * rawData = getter.getRawArrayData(m_keyValues) + start;
         std::stable_sort(rawData, rawData + len);

         // Free stale arrays
         if (m_keys) {
            m_keys.free();
         }

         if (m_values) {
            m_values.free();
         }
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts the first "len" elements by value
      /// @param[in] len - The number of elements to sort
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sort(const size_t len) const {
         sort(0, len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts all the elements by value
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sort() const {
         sort(m_len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts "len" elements starting at "start" by key
      /// Calls std::stable_sort, which uses _kv::cmpKeys to do the comparisons (this
      ///    is an unsort when the keys stored constitute the original ordering)
      /// @note stable_sort is used for consistency with GPU implementation
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to unsort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void sortByKey(const size_t start, const size_t len) const {
         CHAIDataGetter<_kv<KeyType, ValueType>, RAJA::seq_exec> getter {};
         _kv<KeyType, ValueType> * rawData = getter.getRawArrayData(m_keyValues) + start;
         std::stable_sort(rawData, rawData + len, cmpKeys<_kv<KeyType,ValueType>>);

         // Free stale arrays
         if (m_keys) {
            m_keys.free();
         }

         if (m_values) {
            m_values.free();
         }
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts the first "len" elements by value
      /// @param[in] len - The number of elements to unsort
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sortByKey(const size_t len) const {
         sortByKey(0, len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts all the elements by key
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sortByKey() const {
         sortByKey(m_len);
      }
      
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts "len" elements starting at "start" by key, then by value
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to sort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void sortByKeyThenValue(const size_t start, const size_t len) const {
         CHAIDataGetter<_kv<KeyType, ValueType>, RAJA::seq_exec> getter {};
         _kv<KeyType, ValueType> * rawData = getter.getRawArrayData(m_keyValues) + start;
         std::stable_sort(rawData, rawData + len, cmpKeysThenValues<_kv<KeyType,ValueType>>);

         // Free stale arrays
         if (m_keys) {
            m_keys.free();
         }

         if (m_values) {
            m_values.free();
         }
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts the first "len" elements by key, then by value
      /// @param[in] len - The number of elements to sort
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sortByKeyThenValue(const size_t len) const {
         sortByKeyThenValue(0, len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sorts all the elements by key, then by value
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void sortByKeyThenValue() const {
         sortByKeyThenValue(m_len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Does a stable sort on "len" elements starting at "start" by value
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to sort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void stableSort(const size_t start, const size_t len) {
         sort(start, len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Does a stable sort on the first "len" elements by value
      /// @param[in] len - The number of elements to sort
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void stableSort(const size_t len) {
         stableSort(0, len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Does a stable sort on all the elements by value
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void stableSort() {
         stableSort(m_len);
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Jeff Keasler, Alan Dayton
      /// @brief Eliminates duplicate values
      /// First does a stable sort based on the values, which preserves the
      ///    ordering in case of a tie. Then duplicates are removed. The final
      ///    step is to unsort.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void eliminateDuplicates() {
         if (m_len > 1) {
            m_len = eliminateKeyValueDuplicates(m_keyValues, m_len) ;

            // Free stale arrays
            if (m_keys) {
               m_keys.free();
            }

            if (m_values) {
               m_values.free();
            }
         }
      }
      
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Eliminates duplicate key-value pairs
      /// First does a sort by key and then by value, which groups identical pairs.
      /// Then duplicates are removed.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void eliminateDuplicatePairs() {
         if (m_len > 1) {
            // First sort by key and then by value to group identical pairs
            sortByKeyThenValue();
            // Create a new array to hold the unique pairs
            host_device_ptr<_kv<KeyType, ValueType>> uniquePairs(m_len, "uniquePairs");
            // Copy the first element
            uniquePairs.set(0, m_keyValues.pick(0));
            // Copy only non-duplicate elements
            size_t newSize = 1;
            auto keyValues  = m_keyValues;
            CARE_SEQUENTIAL_REF_LOOP(i, 1, m_len, newSize) {
               if (keyValues[i].key != keyValues[i-1].key ||
                  keyValues[i].value != keyValues[i-1].value) {
                  uniquePairs[newSize] = keyValues[i];
                  ++newSize;
               }
            } CARE_SEQUENTIAL_REF_LOOP_END
            
            // Free the original key value pairs
            m_keyValues.free();
            
            // Set to new key value pairs
            m_keyValues = uniquePairs;
            m_len = newSize;
            
            // Reallocate to the correct size
            m_keyValues.realloc(newSize);
            
            // Free stale arrays
            if (m_keys) {
               m_keys.free();
            }

            if (m_values) {
               m_values.free();
            }
         }
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Initializes the keys
      /// The keys are stored in the managed array of _kv structs. To get the
      /// keys separately, they must be copied into their own array.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void initializeKeys() const {
         if (!m_keys) {
            m_keys.alloc(m_len);
            m_keys.namePointer("m_keys");
            initializeKeyArray(m_keys, (host_device_ptr<const _kv<KeyType, ValueType> >)m_keyValues, m_len);
         }

         return;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Initializes the values
      /// The values are stored in the managed array of _kv structs. To get the
      ///    values separately, they must be copied into their own array.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void initializeValues() const {
         if (!m_values) {
            m_values.alloc(m_len);
            m_values.namePointer("m_values");
            initializeValueArray(m_values, (host_device_ptr<const _kv<KeyType, ValueType> >)m_keyValues, m_len);
         }

         return;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// @brief whether keys allocated
      /// The keys are stored in the managed array of _kv structs. To get the
      /// keys separately, they must be copied into their own array.
      /// This routine returns whether that copy is allocated.
      /// @return whether keys are allocated
      ///////////////////////////////////////////////////////////////////////////
      bool keysAllocated() const {
         return m_keys != nullptr;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// @brief whether values allocated
      /// The values are stored in the managed array of _kv structs. To get the
      /// values separately, they must be copied into their own array.
      /// This routine returns whether that copy is allocated.
      /// @return whether values are allocated
      ///////////////////////////////////////////////////////////////////////////
      bool valuesAllocated() const {
         return m_values != nullptr;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// @brief Free the keys
      /// The keys are stored in the managed array of _kv structs. To get the
      /// keys separately, they must be copied into their own array.
      /// This routine frees that copy.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void freeKeys() const {
         if (m_keys) {
            m_keys.free();
         }

         return;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Benjamin Liu
      /// @brief Free the values
      /// The values are stored in the managed array of _kv structs. To get the
      /// values separately, they must be copied into their own array.
      /// This routine frees that copy.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      void freeValues() const {
         if (m_values) {
            m_values.free();
         }

         return;
      }
   private:
      size_t m_len = 0;
      bool m_ownsPointers = false; /// Prevents memory from being freed by lambda captures
      mutable host_device_ptr<KeyType> m_keys = nullptr;
      mutable host_device_ptr<ValueType> m_values = nullptr;
      host_device_ptr<_kv<KeyType, ValueType> > m_keyValues = nullptr;

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson, Alan Dayton
      /// @brief Frees the underlying memory if this is the owner
      /// Used by the destructor and by the assignment operators. Should be private.
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      inline void free() {
         if (m_ownsPointers) {
            if (m_keyValues) {
               m_keyValues.free();
            }

            if (m_keys) {
               m_keys.free();
            }

            if (m_values) {
               m_values.free();
            }
         }
      }
};

#endif // !CARE_ENABLE_GPU_SIMULATION_MODE


// Return the keys for each KVS where their values are the same
#ifdef CARE_PARALLEL_DEVICE
template <typename KeyType, typename ValueType>
void IntersectKeyValueSorters(RAJADeviceExec exec, KeyValueSorter<KeyType, ValueType, RAJADeviceExec> sorter1, int size1,
                              KeyValueSorter<KeyType, ValueType, RAJADeviceExec> sorter2, int size2,
                              host_device_ptr<KeyType> &matches1, host_device_ptr<KeyType>& matches2,
                              int & numMatches) ;
#endif // defined(CARE_PARALLEL_DEVICE)

// This assumes arrays have been sorted.

template <typename KeyType, typename ValueType>
void IntersectKeyValueSorters(RAJA::seq_exec exec, 
                              KeyValueSorter<KeyType, ValueType, RAJA::seq_exec> sorter1, int size1,
                              KeyValueSorter<KeyType, ValueType, RAJA::seq_exec> sorter2, int size2,
                              host_device_ptr<KeyType> &matches1, host_device_ptr<KeyType>& matches2, int & numMatches) ;

} // namespace care

#endif // !defined(_CARE_KEY_VALUE_SORTER_DECL_H_)
