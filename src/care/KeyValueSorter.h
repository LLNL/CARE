//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_KEY_VALUE_SORTER_H_
#define _CARE_KEY_VALUE_SORTER_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/care.h"
#include "care/LoopFuser.h"
#include "care/array_utils.h"

// Other library headers
#ifdef RAJA_GPU_ACTIVE
#ifdef __CUDACC__
#include "cub/cub.cuh"
#undef CUB_NS_POSTFIX
#undef CUB_NS_PREFIX
#endif

#ifdef __HIPCC__
#include "hipcub/hipcub.hpp"
#endif
#endif

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
template <typename T, typename Exec=RAJAExec>
class KeyValueSorter {};

/// LocalKeyValueSorter should be used as the type for HOSTDEV functions
/// to indicate that the function should only be called in a RAJA context.
/// This will prevent the clang-check from checking for functions that
/// should not be called outside a lambda context.
/// Note that this does not actually enforce that the HOSTDEV function
/// is only called from RAJA loops.
template <typename T, typename Exec>
using LocalKeyValueSorter = KeyValueSorter<T, Exec> ;


#ifdef RAJA_GPU_ACTIVE

///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson, Alan Dayton
/// @brief ManagedArray API to cub::DeviceRadixSort::SortPairs
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
template <typename KeyT, typename ValueT, typename Exec=RAJAExec>
inline void sortKeyValueArrays(host_device_ptr<KeyT> & keys,
                               host_device_ptr<ValueT> & values,
                               const size_t start, const size_t len,
                               const bool noCopy=false) {
   // Allocate space for the result
   host_device_ptr<KeyT> keyResult{len};
   host_device_ptr<ValueT> valueResult{len};

   // Get the raw data to pass to cub
   CHAIDataGetter<ValueT, Exec> valueGetter {};
   CHAIDataGetter<KeyT, Exec> keyGetter {};

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

   // Get the result
   if (noCopy) {
      if (len > 0) {
         keys.free(); 
         values.free();
      }

      keys = keyResult;
      values = valueResult;
   }
   else {
      LOOP_STREAM(i, start, start + len) {
         keys[i] = keyResult[i];
         values[i] = valueResult[i];
      } LOOP_STREAM_END

      if (len > 0) {
         keyResult.free();
         valueResult.free();
      }
   }

   if (len > 0) {
      tmpManaged.free();
   }
}

///////////////////////////////////////////////////////////////////////////
/// CUDA partial specialization of KeyValueSorter
/// The CUDA version of KeyValueSorter stores keys and values as separate
///    arrays to be compatible with sortKeyValueArrays.
///////////////////////////////////////////////////////////////////////////
template <typename T>
class KeyValueSorter<T, RAJADeviceExec> {
   public:
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Default constructor
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter<T, RAJADeviceExec>() = default;

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson, Alan Dayton
      /// @brief Constructor
      /// Allocates space for the given number of elements
      /// @param[in] len - The number of elements to allocate space for
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      explicit KeyValueSorter<T, RAJADeviceExec>(const size_t len)
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
      KeyValueSorter<T, RAJADeviceExec>(const size_t len, const T* arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(len, "m_keys")
      , m_values(len, "m_values")
      {
         setFromArray(len, arr);
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
      KeyValueSorter<T, RAJADeviceExec>(const size_t len, host_device_ptr<T> const & arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(len, "m_keys")
      , m_values(len, "m_values")
      {
         setFromArray(len, arr);
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
      CARE_HOST_DEVICE KeyValueSorter<T, RAJADeviceExec>(const KeyValueSorter<T, RAJADeviceExec> &other)
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
      CARE_HOST_DEVICE ~KeyValueSorter<T, RAJADeviceExec>()
      {
#ifndef __CUDA_ARCH__
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
      KeyValueSorter<T, RAJADeviceExec> & operator=(KeyValueSorter<T, RAJADeviceExec> & other)
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
      KeyValueSorter<T, RAJADeviceExec> & operator=(KeyValueSorter<T, RAJADeviceExec> && other)
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
      /// @author Alan Dayton
      /// @brief Initializes the KeyValueSorter by copying elements from the array
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - An array to copy elements from
      /// @return void
      /// TODO: check if len matches m_len (may need to realloc)
      ///////////////////////////////////////////////////////////////////////////
      void setFromArray(const size_t len, const T* arr) {
         host_device_ptr<size_t> keys = m_keys;
         host_device_ptr<T> values = m_values;

         LOOP_SEQUENTIAL(i, 0, len) {
            keys[i] = i;
            values[i] = arr[i];
         } LOOP_SEQUENTIAL_END
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Initializes the KeyValueSorter by copying elements from the array
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - An array to copy elements from
      /// @return void
      /// TODO: check if len matches m_len (may need to realloc).
      /// This method must be public because device lambda functions cannot be in
      /// private or protected functions. They cannot be in constructors, either.
      ///////////////////////////////////////////////////////////////////////////
      void setFromArray(const size_t len, host_device_ptr<T> const & arr) {
         host_device_ptr<size_t> keys = m_keys;
         host_device_ptr<T> values = m_values;

         FUSIBLE_LOOP_STREAM(i, 0, len) {
            keys[i] = (size_t) i;
            values[i] = arr[i];
         } FUSIBLE_LOOP_STREAM_END
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the key at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to get the key
      /// @return the key at the given index
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE size_t key(const size_t index) const {
         return m_keys[index];
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sets the key at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to set the key
      /// @param[in] key   - The new key
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE void setKey(const size_t index, const size_t key) const {
         m_keys[index] = key;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the value at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to get the value
      /// @return the value at the given index
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE T value(const size_t index) const {
         return m_values[index];
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Sets the value at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to set the value
      /// @param[in] value - The new value
      /// @return void
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE void setValue(const size_t index, const T value) const {
         m_values[index] = value;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the keys contained in the KeyValueSorter
      /// @return the keys contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE host_device_ptr<size_t> & keys() {
         return m_keys;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets a const copy of the keys contained in the KeyValueSorter
      /// @return a const copy of the keys contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE const host_device_ptr<size_t> & keys() const {
         return m_keys;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the values contained in the KeyValueSorter
      /// @return the values contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE host_device_ptr<T> & values() {
         return m_values;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets a const copy of the values contained in the KeyValueSorter
      /// @return a const copy of the values contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE const host_device_ptr<T> & values() const {
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
         sortKeyValueArrays(m_values, m_keys, start, len, false);
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
         sortKeyValueArrays(m_values, m_keys, 0, m_len, true);
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
         sortKeyValueArrays(m_keys, m_values, start, len, false);
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
         sortKeyValueArrays(m_keys, m_values, 0, m_len, true);
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
         sortKeyValueArrays(m_values, m_keys, start, len, false);
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
         sortKeyValueArrays(m_values, m_keys, 0, m_len, true);
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
            host_device_ptr<size_t> newKeys{m_len, "newKeys"};
            host_device_ptr<T> newValues{m_len, "newValues"};


            // Save values that are not duplicates and their corresponding keys
            int newSize = 0;

            const size_t len = m_len;
            host_device_ptr<size_t const> keys = m_keys;
            host_device_ptr<T const> values = m_values;

            SCAN_LOOP(i, 0, len, idx, newSize, (i == 0) || (values[i] != values[i-1])) {
               newKeys[idx] = keys[i];
               newValues[idx] = values[i];
            } SCAN_LOOP_END(len, idx, newSize)

            // Free the original key value pairs
            free();

            // Update space for the key value pairs without duplicates
            newKeys.realloc(newSize);
            newValues.realloc(newSize);

            m_keys = newKeys;
            m_values = newValues;
            m_len = newSize;

            // Restore original ordering
            sortByKey();
         }
      }

   private:
      size_t m_len = 0;
      bool m_ownsPointers = false; /// Prevents memory from being freed by lambda captures
      host_device_ptr<size_t> m_keys = nullptr;
      host_device_ptr<T> m_values = nullptr;

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

#endif // RAJA_GPU_ACTIVE



///////////////////////////////////////////////////////////////////////////
/// @author Peter Robinson
/// @brief Less than comparison operator for values
/// Used as a comparator in the STL
/// @param left  - left _kv to compare
/// @param right - right _kv to compare
/// @return true if left's value is less than right's value, false otherwise
///////////////////////////////////////////////////////////////////////////
template <typename T>
inline bool operator <(_kv<T> const & left, _kv<T> const & right)
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
template <typename T>
inline bool cmpKeys(_kv<T> const & left, _kv<T> const & right)
{
   return left.key < right.key;
}

///////////////////////////////////////////////////////////////////////////
/// @author Alan Dayton
/// @brief Less than comparison operator for values first and keys second
/// Used as a comparator in the STL
/// @param left  - left _kv to compare
/// @param right - right _kv to compare
/// @return true if left's value is less than right's value. If they are
///    equal, returns true if left's key is less than right's key.
///    Otherwise returns false.
///////////////////////////////////////////////////////////////////////////
template <typename T>
inline bool cmpValsStable(_kv<T> const & left, _kv<T> const & right)
{
   if (left.value == right.value) {
      return cmpKeys(left, right);
   }
   else {
      return left < right;
   }
}

/// cmpValsStable<real8> specialization
template <>
inline bool cmpValsStable<real8>(_kv<real8> const & left, _kv<real8> const & right)
{
   if (left < right) {
      return true;
   }
   else if (left.value > right.value) {
      return false;
   }
   else {
      return cmpKeys(left, right);
   }
}

/// cmpValsStable<float> specialization
template <>
inline bool cmpValsStable<float>(_kv<float> const & left, _kv<float> const & right)
{
   if (left < right) {
      return true;
   }
   else if (left.value > right.value) {
      return false;
   }
   else {
      return cmpKeys(left, right);
   }
}

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
template <typename T>
class KeyValueSorter<T, RAJA::seq_exec> {
   public:

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Default constructor
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter<T, RAJA::seq_exec>() = default;

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson, Alan Dayton
      /// @brief Constructor
      /// Allocates space for the given number of elements
      /// @param[in] len - The number of elements to allocate space for
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      explicit KeyValueSorter<T, RAJA::seq_exec>(size_t len)
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
      KeyValueSorter<T, RAJA::seq_exec>(const size_t len, const T* arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(nullptr)
      , m_values(nullptr)
      , m_keyValues(len, "m_keyValues")
      {
         setFromArray(len, arr);
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
      KeyValueSorter<T, RAJA::seq_exec>(const size_t len, host_device_ptr<T> const & arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(nullptr)
      , m_values(nullptr)
      , m_keyValues(len, "m_keyValues")
      {
         setFromArray(len, arr);
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
      CARE_HOST_DEVICE KeyValueSorter<T, RAJA::seq_exec>(const KeyValueSorter<T, RAJA::seq_exec> &other)
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
      CARE_HOST_DEVICE ~KeyValueSorter<T, RAJA::seq_exec>()
      {
#ifndef __CUDA_ARCH__
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
      KeyValueSorter<T, RAJA::seq_exec> & operator=(KeyValueSorter<T, RAJA::seq_exec> & other)
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
      KeyValueSorter<T, RAJA::seq_exec> & operator=(KeyValueSorter<T, RAJA::seq_exec> && other)
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
      /// @author Alan Dayton
      /// @brief Initializes the KeyValueSorter by copying elements from the array
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - An array to copy elements from
      /// @return void
      /// TODO: check if len matches m_len (may need to realloc)
      ///////////////////////////////////////////////////////////////////////////
      void setFromArray(const size_t len, const T* arr) {
         host_device_ptr<_kv<T> > keyValues = m_keyValues;

         LOOP_SEQUENTIAL(i, 0, (int) len) {
            keyValues[i].key = i;
            keyValues[i].value = arr[i];
         } LOOP_SEQUENTIAL_END
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      /// @brief Initializes the KeyValueSorter by copying elements from the array
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - An array to copy elements from
      /// @return void
      /// TODO: check if len matches m_len (may need to realloc).
      /// This method must be public because device lambda functions cannot be in
      /// private or protected functions. They cannot be in constructors, either.
      ///////////////////////////////////////////////////////////////////////////
      void setFromArray(const size_t len, host_device_ptr<T> const & arr) {
         host_device_ptr<_kv<T> > keyValues = m_keyValues;

         FUSIBLE_LOOP_STREAM(i, 0, (int)len) {
            keyValues[i].key = i;
            keyValues[i].value = arr[i];
         } FUSIBLE_LOOP_STREAM_END
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the key at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to get the key
      /// @return the key at the given index
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE size_t key(const size_t index) const {
         local_ptr<_kv<T> > local_keyValues = m_keyValues;
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
      CARE_HOST_DEVICE void setKey(const size_t index, const size_t key) const {
         local_ptr<_kv<T> > local_keyValues = m_keyValues;
         local_keyValues[index].key = key;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the value at the given index
      /// @note This should only be called from within a RAJA context.
      /// @param[in] index - The index at which to get the value
      /// @return the value at the given index
      ///////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE T value(const size_t index) const {
         local_ptr<_kv<T> > local_keyValues = m_keyValues;
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
      CARE_HOST_DEVICE void setValue(const size_t index, const T value) const {
         local_ptr<_kv<T> > local_keyValues = m_keyValues;
         local_keyValues[index].value = value;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the keys contained in the KeyValueSorter
      /// @return the keys contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      host_device_ptr<size_t> & keys() {
         initializeKeys();
         return m_keys;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets a const copy of the keys contained in the KeyValueSorter
      /// @return a const copy of the keys contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      const host_device_ptr<size_t> & keys() const {
         initializeKeys();
         return m_keys;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets the values contained in the KeyValueSorter
      /// @return the values contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      host_device_ptr<T> & values() {
         initializeValues();
         return m_values;
      }

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Gets a const copy of the values contained in the KeyValueSorter
      /// @return a const copy of the values contained in the KeyValueSorter
      ///////////////////////////////////////////////////////////////////////////
      const host_device_ptr<T> & values() const {
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
      /// Calls std::sort, which uses _kv::operator< to do the comparisons
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to sort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void sort(const size_t start, const size_t len) const {
         CHAIDataGetter<_kv<T>, RAJA::seq_exec> getter {};
         _kv<T> * rawData = getter.getRawArrayData(m_keyValues) + start;
         std::sort(rawData, rawData + len);
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
      /// Calls std::sort, which uses _kv::cmpKeys to do the comparisons (this
      ///    is an unsort when the keys stored constitute the original ordering)
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to unsort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void sortByKey(const size_t start, const size_t len) const {
         CHAIDataGetter<_kv<T>, RAJA::seq_exec> getter {};
         _kv<T> * rawData = getter.getRawArrayData(m_keyValues) + start;
         std::sort(rawData, rawData + len, cmpKeys<T>);
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
      /// @author Alan Dayton
      /// @brief Does a stable sort on "len" elements starting at "start" by value
      /// Calls std::sort, which uses _kv::cmpValsStable to do the comparisons
      /// @param[in] start - The index to start at
      /// @param[in] len   - The number of elements to sort
      /// @return void
      /// TODO: add bounds checking
      ///////////////////////////////////////////////////////////////////////////
      void stableSort(const size_t start, const size_t len) {
         CHAIDataGetter<_kv<T>, RAJA::seq_exec> getter {};
         _kv<T> * rawData = getter.getRawArrayData(m_keyValues) + start;
         std::sort(rawData, rawData + len, cmpValsStable<T>);
         // TODO: investigate performance of std::stable_sort
         //std::stable_sort(rawData, rawData + len);
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
            CHAIDataGetter<_kv<T>, RAJA::seq_exec> getter {};
            _kv<T> * rawData = getter.getRawArrayData(m_keyValues);

            // First do a stable sort by value (preserve the original order
            // in the case of a tie)
            std::sort(rawData, rawData + m_len, cmpValsStable<T>);
            // TODO: investigate performance of std::stable_sort
            // std::stable_sort(rawData, rawData + m_len);

            // Then eliminate duplicates
            size_t lsize = m_len - 1;  /* adjust search range */
            size_t put = 0;
            size_t get = 0;

            while (get < lsize) {
               if (put != get) {
                  memcpy(&rawData[put], &rawData[get], sizeof(struct _kv<T>));
               }

               if (rawData[get].value == rawData[get+1].value) {
                  ++get;
                  ++put;

                  while (get < lsize && rawData[get].value == rawData[get+1].value) {
                     ++get;
                  }
                  ++get;
               }
               else {
                  ++get;
                  ++put;
               }
            }

            if (rawData[lsize].value != rawData[lsize-1].value) {
               memmove(&rawData[put++], &rawData[lsize], sizeof(struct _kv<T>));
            }

            lsize = put;

            // Then sort by key to get the original ordering
            std::sort(rawData, rawData + lsize, cmpKeys<T>);

            // Reallocate memory
            if (m_keyValues) {
               m_keyValues.realloc(lsize);
            }

            if (m_keys) {
               m_keys.realloc(lsize);
            }

            if (m_values) {
               m_values.realloc(lsize);
            }

            m_len = lsize;
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

            host_device_ptr<size_t> keys = m_keys;
            host_device_ptr<_kv<T> const> keyValues = m_keyValues;

            LOOP_STREAM(i, 0, m_len) {
               keys[i] = keyValues[i].key;
            } LOOP_STREAM_END
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

            host_device_ptr<T> values = m_values;
            host_device_ptr<_kv<T> const> keyValues = m_keyValues;

            LOOP_STREAM(i, 0, m_len) {
               values[i] = keyValues[i].value;
            } LOOP_STREAM_END
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
      mutable host_device_ptr<size_t> m_keys = nullptr;
      mutable host_device_ptr<T> m_values = nullptr;
      host_device_ptr<_kv<T> > m_keyValues = nullptr;

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


#ifdef RAJA_GPU_ACTIVE
template <typename T>
void IntersectKeyValueSorters(RAJAExec exec, KeyValueSorter<T> sorter1, int size1,
                              KeyValueSorter<T> sorter2, int size2,
                              host_device_ptr<int> &matches1, host_device_ptr<int>& matches2,
                              int & numMatches) {
 
   
   int smaller = (size1 < size2) ? size1 : size2 ;
   int start1 = 0;
   int start2 = 0;

   numMatches = 0 ;
   if (smaller == 0) {
      matches1 = nullptr ;
      matches2 = nullptr ;
      return ;
   }
   else {
      matches1.alloc(smaller);
      matches1.namePointer("matches1");
      matches2.alloc(smaller);
      matches2.namePointer("matches2");
   }

   host_device_ptr<int> smallerMatches, largerMatches;
   host_device_ptr<size_t> smallerKeys, largerKeys;
   int larger, smallStart, largeStart;
   host_device_ptr<const T> smallerArray, largerArray;
   if (smaller == size1) {
      smallerArray = sorter1.values();
      largerArray = sorter2.values();
      smallerKeys = sorter1.keys();
      largerKeys = sorter2.keys();
      larger = size2;
      smallStart = start1;
      largeStart = start2;
      smallerMatches = matches1;
      largerMatches = matches2;
   }
   else {
      smallerArray = sorter2.values();
      largerArray = sorter1.values();
      smallerKeys = sorter2.keys();
      largerKeys = sorter1.keys();
      larger = size1;
      smallStart = start2;
      largeStart = start1;
      smallerMatches = matches2;
      largerMatches = matches1;
   }

   host_device_ptr<int> searches{smaller+1};
   host_device_ptr<int> matched{smaller+1};
   LOOP_STREAM(i, 0, smaller+1) {
      searches[i] = i != smaller ? care_utils::BinarySearch<T>(largerArray, largeStart, larger, smallerArray[i+smallStart]) : -1;
      matched[i] = i != smaller && searches[i] > -1;
   } LOOP_STREAM_END

   exclusive_scan<int, RAJAExec>(matched, nullptr, smaller+1, RAJA::operators::plus<int>{}, 0, true);

   LOOP_STREAM(i, 0, smaller) {
      if (searches[i] > -1) {
         smallerMatches[matched[i]] = smallerKeys[i+smallStart];
         largerMatches[matched[i]] = largerKeys[searches[i]];
      }
   } LOOP_STREAM_END
   numMatches =  matched.pick(smaller);
   searches.free();
   matched.free();
   
   /* change the size of the array */
   if (numMatches == 0) {
      matches1.free();
      matches2.free();
   }
   else {
      matches1.realloc(numMatches);
      matches2.realloc(numMatches);
   }

}
#endif

// This assumes arrays have been sorted and unique. If they are not uniqued the GPU
// and CPU versions may have different behaviors (the index they match to may be different, 
// with the GPU implementation matching whatever binary search happens to land on, and the// CPU version matching the first instance. 

template <typename T>
void IntersectKeyValueSorters(RAJA::seq_exec exec, 
                              KeyValueSorter<T> sorter1, int size1,
                              KeyValueSorter<T> sorter2, int size2,
                              host_device_ptr<int> &matches1, host_device_ptr<int>& matches2, int & numMatches) {

   numMatches = 0 ;
   const int smaller = (size1 < size2) ? size1 : size2 ;

   if (smaller == 0) {
      matches1 = nullptr ;
      matches2 = nullptr ;
      return ;
   }
   else {
      matches1.alloc(smaller);
      matches1.namePointer("matches1");
      matches2.alloc(smaller);
      matches2.namePointer("matches2");
   }

   /* This algorithm assumes that the nodelists are sorted */


   int i = 0 ;
   int j = 0 ;
   host_ptr<int> host_matches1 = matches1 ;
   host_ptr<int> host_matches2 = matches2 ;
   /* keys() and values() will allocate managed arrays for the keys and values,
    * respectively, if they were not previously allocated.
    * Check to see whether they were previously allocated. */
   bool sorter1KeysAllocated = sorter1.keysAllocated() ;
   bool sorter2KeysAllocated = sorter2.keysAllocated() ;
   bool sorter1ValuesAllocated = sorter1.valuesAllocated() ;
   bool sorter2ValuesAllocated = sorter2.valuesAllocated() ;
   host_ptr<size_t const> host_sorter1_key = sorter1.keys() ;
   host_ptr<size_t const> host_sorter2_key = sorter2.keys() ;
   host_ptr<T const> host_sorter1_value = sorter1.values() ;
   host_ptr<T const> host_sorter2_value = sorter2.values() ;

   for (;; ) {
      if ((i >= size1) || (j >= size2)) {
         break ;
      }
      while ((i < size1) && (host_sorter1_value[i] < host_sorter2_value[j])) {
         i++ ;
      }
      if (i >= size1) {
         break ;
      }
      while ((j < size2) && (host_sorter2_value[j] < host_sorter1_value[i])) {
         j++ ;
      }
      if (j >= size2) {
         break ;
      }
      if (host_sorter1_value[i] == host_sorter2_value[j]) {
         host_matches1[numMatches] = host_sorter1_key[i] ;
         host_matches2[numMatches] = host_sorter2_key[j] ;
         numMatches++ ;
         i++ ;
         j++ ;
      }
      else if (host_sorter1_value[i] < host_sorter2_value[j]) {
         i++ ;
      }
      else if (host_sorter2_value[j] < host_sorter1_value[i]) {
         j++ ;
      }
   }

   /* change the size of the array */
   /* (reallocing to a size of zero should be the same as freeing
    * the object, but insight doesn't seem to think so... hence
    * the extra check here with an explicit free */
   if (numMatches == 0) {
      matches1.free();
      matches2.free();
   }
   else {
      matches1.realloc(numMatches);
      matches2.realloc(numMatches);
   }

   /* If the keys/values arrays were not previously allocated, free them. */
   if (!sorter1KeysAllocated) {
      sorter1.freeKeys() ;
   }
   if (!sorter2KeysAllocated) {
      sorter2.freeKeys() ;
   }
   if (!sorter1ValuesAllocated) {
      sorter1.freeValues() ;
   }
   if (!sorter2ValuesAllocated) {
      sorter2.freeValues() ;
   }
}

}

#endif // !defined(_CARE_KEY_VALUE_SORTER_H_)

