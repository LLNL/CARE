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

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray
         ///
         host_ptr(chai::ManagedArray<T> const &ptr) : m_ptr(ptr.data(chai::CPU)) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray containing non-const elements if T is const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         host_ptr<T>(chai::ManagedArray<T_non_const>& ptr) : m_ptr(ptr.data(chai::CPU)) {}

         ///
         /// @author Peter Robinson
         ///
         /// Return the element at the given index
         ///
         inline T & operator[](int index) const { return m_ptr[index]; }

// We will allow this implicit conversion since it is safe for host-only pointers. We could re-consider
// this in the future, however, but it may require many code changes for users.
//#if defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)
         ///
         /// @author Peter Robinson
         ///
         /// Convert to a raw pointer
         ///
         operator T*() const { return m_ptr; }
//#endif

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
         /// Get the underlying data array. In the future, this may replace operator T*()
         ///
         T* data() const { return m_ptr; }

      private:
         T * m_ptr; //!< Raw host pointer
   };
} // namespace care

#endif // !defined(_CARE_HOST_PTR_H_)

