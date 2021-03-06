//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_KEY_VALUE_SORTER_DECL_H_
#define _CARE_KEY_VALUE_SORTER_DECL_H_

// CARE config header
#include "care/config.h"

#include "care/algorithm_decl.h"
#include "care/CHAIDataGetter.h"

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


// TODO openMP parallel implementation
#ifdef CARE_GPUCC

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
template <typename KeyT, typename ValueT, typename Exec=RAJADeviceExec>
void sortKeyValueArrays(host_device_ptr<KeyT> & keys,
                        host_device_ptr<ValueT> & values,
                        const size_t start, const size_t len,
                        const bool noCopy=false) ;

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes keys and values by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to copy
/// @param[in] arr - input array
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename T>
void setKeyValueArraysFromArray(host_device_ptr<size_t> & keys, host_device_ptr<T> & values,
                                const size_t len, const T* arr) ;

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes the KeyValueSorter by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename T>
void setKeyValueArraysFromManagedArray(host_device_ptr<size_t> & keys, host_device_ptr<T> & values,
                                       const size_t len, const host_device_ptr<const T>& arr) ;

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
template <typename T>
size_t eliminateKeyValueDuplicates(host_device_ptr<size_t>& newKeys,
                                   host_device_ptr<T>& newValues,
                                   const host_device_ptr<const size_t>& oldKeys,
                                   const host_device_ptr<const T>& oldValues,
                                   const size_t oldLen);

///////////////////////////////////////////////////////////////////////////
/// GPU partial specialization of KeyValueSorter
/// The GPU version of KeyValueSorter stores keys and values as separate
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
      KeyValueSorter() {}

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
      KeyValueSorter<T, RAJADeviceExec>(const size_t len, const host_device_ptr<const T> & arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(len, "m_keys")
      , m_values(len, "m_values")
      {
         setKeyValueArraysFromManagedArray(m_keys, m_values, len, arr);
      }

#if defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      ///
      /// @brief Constructor
      ///
      /// Allocates space and initializes the KeyValueSorter by copying
      ///    elements and ordering from the given managed array
      ///
      /// @note This overload is needed to prevent ambiguity when implicit
      ///       casts are enabled
      ///
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - The managed array to copy elements from
      ///
      /// @return a KeyValueSorter instance
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter<T, RAJADeviceExec>(const size_t len, const host_device_ptr<T> & arr)
      : KeyValueSorter<T, RAJADeviceExec>(len, host_device_ptr<const T>(arr))
      {
      }

#endif // defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

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

            int newSize = eliminateKeyValueDuplicates(newKeys, newValues,
                                                      (host_device_ptr<const size_t>)m_keys,
                                                      (host_device_ptr<const T>)m_values,
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
      /// @author Benjamin Liu
      /// @brief no-op
      /// GPU version does not require separate allocation for kesy array.
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

#endif // defined(CARE_GPUCC)



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

/// cmpValsStable<double> specialization
template <>
inline bool cmpValsStable<double>(_kv<double> const & left, _kv<double> const & right)
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
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes keys and values by copying elements from the array
/// @param[out] keyValues - The key value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename T>
void setKeyValueArraysFromArray(host_device_ptr<_kv<T> > & keyValues,
                                const size_t len, const T* arr) ;

///////////////////////////////////////////////////////////////////////////
/// @author Benjamin Liu after Alan Dayton
/// @brief Initializes the KeyValueSorter by copying elements from the array
/// @param[out] keys   - The key array to set to the identity
/// @param[out] values - The value array to set
/// @param[in] len - The number of elements to allocate space for
/// @param[in] arr - An array to copy elements from
/// @return void
///////////////////////////////////////////////////////////////////////////
template <typename T>
void setKeyValueArraysFromManagedArray(host_device_ptr<_kv<T> > & keyValues,
                                       const size_t len, const host_device_ptr<const T>& arr) ;


///////////////////////////////////////////////////////////////////////////
/// @author Jeff Keasler, Alan Dayton
/// @brief Eliminates duplicate values
/// Assumes key value array sorted by values.
/// @param[in/out] keyValues - The key value array to eliminate duplicates in
/// @param[in/out] len - original length of key value array
/// @return new length of array
///////////////////////////////////////////////////////////////////////////
template <typename T>
size_t eliminateKeyValueDuplicates(host_device_ptr<_kv<T> > & keyValues, const size_t len) ;

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
template <typename T>
void initializeKeyArray(host_device_ptr<size_t>& keys, const host_device_ptr<const _kv<T> >& keyValues, const size_t len) ;

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
template <typename T>
void initializeValueArray(host_device_ptr<T>& values, const host_device_ptr<const _kv<T> >& keyValues, const size_t len);

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
      KeyValueSorter<T, RAJA::seq_exec>() {}

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
      KeyValueSorter<T, RAJA::seq_exec>(const size_t len, const host_device_ptr<const T> & arr)
      : m_len(len)
      , m_ownsPointers(true)
      , m_keys(nullptr)
      , m_values(nullptr)
      , m_keyValues(len, "m_keyValues")
      {
         setKeyValueArraysFromManagedArray(m_keyValues, len, arr);
      }

#if defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

      ///////////////////////////////////////////////////////////////////////////
      /// @author Alan Dayton
      ///
      /// @brief Constructor
      ///
      /// Allocates space and initializes the KeyValueSorter by copying
      ///    elements and ordering from the given managed array
      ///
      /// @note This overload is needed to prevent ambiguity when implicit
      ///       casts are enabled
      ///
      /// @param[in] len - The number of elements to allocate space for
      /// @param[in] arr - The managed array to copy elements from
      ///
      /// @return a KeyValueSorter instance
      ///
      ///////////////////////////////////////////////////////////////////////////
      KeyValueSorter<T, RAJA::seq_exec>(const size_t len, const host_device_ptr<T> & arr)
      : KeyValueSorter<T, RAJA::seq_exec>(len, host_device_ptr<const T>(arr))
      {
      }

#endif // defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

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
            initializeKeyArray(m_keys, (host_device_ptr<const _kv<T> >)m_keyValues, m_len);
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
            initializeValueArray(m_values, (host_device_ptr<const _kv<T> >)m_keyValues, m_len);
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


#ifdef CARE_GPUCC
template <typename T>
void IntersectKeyValueSorters(RAJADeviceExec exec, KeyValueSorter<T, RAJADeviceExec> sorter1, int size1,
                              KeyValueSorter<T, RAJADeviceExec> sorter2, int size2,
                              host_device_ptr<int> &matches1, host_device_ptr<int>& matches2,
                              int & numMatches) ;
#endif // defined(CARE_GPUCC)

// This assumes arrays have been sorted and unique. If they are not uniqued the GPU
// and CPU versions may have different behaviors (the index they match to may be different, 
// with the GPU implementation matching whatever binary search happens to land on, and the// CPU version matching the first instance. 

template <typename T>
void IntersectKeyValueSorters(RAJA::seq_exec exec, 
                              KeyValueSorter<T, RAJA::seq_exec> sorter1, int size1,
                              KeyValueSorter<T, RAJA::seq_exec> sorter2, int size2,
                              host_device_ptr<int> &matches1, host_device_ptr<int>& matches2, int & numMatches) ;

} // namespace care

#endif // !defined(_CARE_KEY_VALUE_SORTER_DECL_H_)

