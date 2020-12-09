//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_HOST_PTR_H_
#define _CARE_HOST_PTR_H_

// CARE headers
#include "care/single_access_ptr.h"

// Other library headers
#include "chai/ManagedArray.hpp"

// Std library headers
#include <cstddef>

namespace care {
   ///
   /// @author Peter Robinson
   ///
   /// Designed to be used only on the host. If used in a device context,
   /// will produce a compile time error.
   ///
   template <typename T>
   class host_ptr : public single_access_ptr {
      private:
         using T_non_const = typename std::remove_const<T>::type;

      public:
         using value_type = T;

         ///
         /// @author Peter Robinson
         ///
         /// Default constructor
         ///
         host_ptr() noexcept : m_ptr(nullptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// nullptr constructor
         ///
         host_ptr(std::nullptr_t) noexcept : m_ptr(nullptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from raw pointer
         ///
         host_ptr(T* ptr) noexcept : m_ptr(ptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// Copy constructor
         ///
         host_ptr(host_ptr const & ptr) noexcept : m_ptr(ptr.data()) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from host_ptr containing non-const elements if T is const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         host_ptr<T>(host_ptr<T_non_const> const &ptr) noexcept : m_ptr(ptr.data()) {}

         // TODO: When CHAI has a new tagged version greather than v2.1.1,
         //       use .data instead of .getPointer. Also, we should be able
         //       to pass const references to chai::ManagedArrays.

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray
         ///
         host_ptr(chai::ManagedArray<T> ptr) : m_ptr(ptr.getPointer(chai::CPU)) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray containing non-const elements if T is const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         host_ptr<T>(chai::ManagedArray<T_non_const> ptr) : m_ptr(ptr.getPointer(chai::CPU)) {}

         ///
         /// @author Peter Robinson
         ///
         /// Return the element at the given index
         ///
         inline T & operator[](int index) const { return m_ptr[index]; }

#if defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)
         ///
         /// @author Peter Robinson
         ///
         /// Convert to a raw pointer
         ///
         operator T*() const { return m_ptr; }
#endif

         ///
         /// @author Peter Robinson
         ///
         /// Pointer arithmetic
         ///
         T* operator ++(int) { return m_ptr++; }

         ///
         /// @author Peter Robinson
         ///
         /// Pointer arithmetic
         ///
         T* operator ++() { return ++m_ptr; }

         ///
         /// @author Peter Robinson
         ///
         /// Pointer arithmetic
         ///
         template<typename Idx>
         host_ptr<T> & operator +=(Idx i) { m_ptr += i; return *this; }

         ///
         /// @author Danny Taller
         ///
         /// Get the underlying data array.
         ///
         T* data() const { return m_ptr; }

         ///
         /// @author Alan Dayton
         ///
         /// Get the underlying data array.
         ///
         const T* cdata() const { return m_ptr; }

         ///
         /// @author Alan Dayton
         ///
         /// Returns true if the contained pointer is not nullptr, false otherwise.
         ///
         inline explicit operator bool() const noexcept {
            return m_ptr != nullptr;
         }

      private:
         T * m_ptr; //!< Raw host pointer
   };

   /// Comparison operators

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison.
   ///
   /// @param[in] lhs The first host_ptr to compare
   /// @param[in] rhs The second host_ptr to compare
   ///
   template <typename T, typename U>
   bool operator==(const host_ptr<T>& lhs, const host_ptr<U>& rhs) noexcept {
      return lhs.cdata() == rhs.cdata();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison.
   ///
   /// @param[in] lhs The first host_ptr to compare
   /// @param[in] rhs The second host_ptr to compare
   ///
   template <typename T, typename U>
   bool operator!=(const host_ptr<T>& lhs, const host_ptr<U>& rhs) noexcept {
      return lhs.cdata() != rhs.cdata();
   }

   /// Comparison operators with nullptr

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison with nullptr.
   ///
   /// @param[in] lhs The host_ptr to compare to nullptr
   ///
   template <typename T>
   bool operator==(const host_ptr<T>& lhs, std::nullptr_t) noexcept {
      return lhs.cdata() == nullptr;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Equals comparison with nullptr.
   ///
   /// @param[in] rhs The host_ptr to compare to nullptr
   ///
   template <typename T>
   bool operator==(std::nullptr_t, const host_ptr<T>& rhs) noexcept {
      return nullptr == rhs.cdata();
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison with nullptr.
   ///
   /// @param[in] lhs The host_ptr to compare to nullptr
   ///
   template <typename T>
   bool operator!=(const host_ptr<T>& lhs, std::nullptr_t) noexcept {
      return lhs.cdata() != nullptr;
   }

   ///
   /// @author Alan Dayton
   ///
   /// Not equals comparison with nullptr.
   ///
   /// @param[in] rhs The host_ptr to compare to nullptr
   ///
   template <typename T>
   bool operator!=(std::nullptr_t, const host_ptr<T>& rhs) noexcept {
      return nullptr != rhs.cdata();
   }
} // namespace care

#endif // !defined(_CARE_HOST_PTR_H_)

