//////////////////////////////////////////////////////////////////////////////
// Copyright (c) Lawrence Livermore National Security, LLC and other CARE
// contributors. See the CARE LICENSE and COPYRIGHT files for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_ZIP_ITERATOR_H
#define CARE_ZIP_ITERATOR_H

#include <cstddef>
#include <iterator>
#include <utility>

namespace care {

/// A random-access iterator over two same-length arrays.
template <typename KeyT, typename ValueT>
class zip_iterator {
 public:
   struct value_type {
      KeyT key;
      ValueT value;
   };

 private:
   struct reference {
      KeyT& key;
      ValueT& value;

      operator value_type() const { return {key, value}; }

      reference& operator=(value_type const& other)
      {
         key = other.key;
         value = other.value;
         return *this;
      }
      reference& operator=(reference const& other)
      { return *this = value_type(other); }
      reference& operator=(reference&& other)
      { return *this = value_type(other); }

      friend void swap(reference left, reference right) noexcept
      {
         using std::swap;
         swap(left.key, right.key);
         swap(left.value, right.value);
      }
   };

   KeyT* m_keys = nullptr;
   ValueT* m_values = nullptr;
   std::ptrdiff_t m_index = 0;

 public:
   using difference_type = std::ptrdiff_t;
   using reference_type = reference;
   using iterator_category = std::random_access_iterator_tag;

   zip_iterator() = default;
   zip_iterator(KeyT* keys, ValueT* values, std::ptrdiff_t index = 0)
      : m_keys(keys), m_values(values), m_index(index) {}

   reference operator*() const { return {m_keys[m_index], m_values[m_index]}; }
   reference operator[](difference_type offset) const { return *(*this + offset); }

   zip_iterator& operator++() { ++m_index; return *this; }
   zip_iterator operator++(int) { auto result = *this; ++*this; return result; }
   zip_iterator& operator--() { --m_index; return *this; }
   zip_iterator operator--(int) { auto result = *this; --*this; return result; }
   zip_iterator& operator+=(difference_type offset) { m_index += offset; return *this; }
   zip_iterator& operator-=(difference_type offset) { m_index -= offset; return *this; }

   friend zip_iterator operator+(zip_iterator it, difference_type offset) { return it += offset; }
   friend zip_iterator operator+(difference_type offset, zip_iterator it) { return it += offset; }
   friend zip_iterator operator-(zip_iterator it, difference_type offset) { return it -= offset; }
   friend difference_type operator-(zip_iterator left, zip_iterator right)
   { return left.m_index - right.m_index; }

   friend bool operator==(zip_iterator left, zip_iterator right) { return left.m_index == right.m_index; }
   friend bool operator!=(zip_iterator left, zip_iterator right) { return !(left == right); }
   friend bool operator<(zip_iterator left, zip_iterator right) { return left.m_index < right.m_index; }
   friend bool operator>(zip_iterator left, zip_iterator right) { return right < left; }
   friend bool operator<=(zip_iterator left, zip_iterator right) { return !(right < left); }
   friend bool operator>=(zip_iterator left, zip_iterator right) { return !(left < right); }
};

} // namespace care

#endif // CARE_ZIP_ITERATOR_H
