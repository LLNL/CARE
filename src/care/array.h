//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_ARRAY_H_
#define _CARE_ARRAY_H_

// CARE config header
#include "care/config.h"

#if defined(__HIPCC__)
#include "hip/hip_runtime.h"
#endif

namespace care {
   ////////////////////////////////////////////////////////////////
   ///
   /// @author Peter Robinson, Ben Liu, Alan Dayton
   ///
   /// Implements a subset of the functionality of std::array.
   /// It is readable and writable on the host, but read-only on
   /// the device.
   ///
   ////////////////////////////////////////////////////////////////
   template <class T, std::size_t N>
   struct array {
      using value_type = T;
      using size_type = std::size_t;
      using difference_type = std::ptrdiff_t;
      using reference = value_type&;
      using const_reference = const value_type&;
      using pointer = value_type*;
      using const_pointer = const value_type*;

      array() = default;

      explicit array(const T* values) {
         for (size_type i = 0; i < N; ++i) {
            elements[i] = values[i];
         }
      }

      explicit array(const std::array<T, N>& values) {
         for (size_type i = 0; i < N; ++i) {
            elements[i] = values[i];
         }
      }

      // Writeable only on the host
      reference operator[](size_type pos) {
         return elements[pos];
      }

      // Readable on the host and device
      CARE_HOST_DEVICE const_reference operator[](size_type pos) const {
         return elements[pos];
      }

      reference front() {
         return elements[0];
      }

      CARE_HOST_DEVICE const_reference front() const {
         return elements[0];
      }

      reference back() {
         return elements[N - 1];
      }

      CARE_HOST_DEVICE const_reference back() const {
         return elements[N - 1];
      }

      // Writeable only on the host
      pointer data() noexcept {
         return elements;
      }

      // Readable on the host and device
      CARE_HOST_DEVICE const_pointer data() const noexcept {
         return elements;
      }

      CARE_HOST_DEVICE constexpr bool empty() const noexcept {
         return N == 0;
      }

      CARE_HOST_DEVICE constexpr size_type size() const noexcept {
         return N;
      }

      void fill(const_reference value) {
         for (size_type i = 0; i < N; ++i) {
            elements[i] = value;
         }
      }

      void swap(array& other) noexcept {
         std::swap(elements, other.elements);
      }

      value_type elements[N];
   };

   template <class T, std::size_t N>
   CARE_HOST_DEVICE bool operator==(const array<T, N>& lhs,
                                const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] != rhs[i]) {
            return false;
         }
      }

      return true;
   }

   template <class T, std::size_t N>
   CARE_HOST_DEVICE inline bool operator!=(const array<T, N>& lhs,
                                       const array<T, N>& rhs) {
      return !(lhs == rhs);
   }

   template <class T, std::size_t N>
   CARE_HOST_DEVICE bool operator<(const array<T, N>& lhs,
                               const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] >= rhs[i]) {
            return false;
         }
      }

      return true;
   }

   template <class T, std::size_t N>
   CARE_HOST_DEVICE bool operator<=(const array<T, N>& lhs,
                                const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] > rhs[i]) {
            return false;
         }
      }

      return true;
   }

   template <class T, std::size_t N>
   CARE_HOST_DEVICE bool operator>(const array<T, N>& lhs,
                               const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] <= rhs[i]) {
            return false;
         }
      }

      return true;
   }

   template <class T, std::size_t N>
   CARE_HOST_DEVICE bool operator>=(const array<T, N>& lhs,
                                const array<T, N>& rhs) {
      for (std::size_t i = 0; i < N; ++i) {
         if (lhs[i] < rhs[i]) {
            return false;
         }
      }

      return true;
   }
} // namespace care

#endif // !defined(_CARE_ARRAY_H_)

