//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_LOCAL_PTR_H_
#define _CARE_LOCAL_PTR_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/host_ptr.h"
#include "care/single_access_ptr.h"

// CHAI headers
#include "chai/ManagedArray.hpp"

// Std library headers
#include <cstddef>

namespace care {
   // local_ptr access can only occur on host or device, but [] const is not allowed, meaning
   // it can't be captured and used, (must be created within a loop to work).
   template <typename T>
   class local_ptr : public single_access_ptr {
      private:
         using T_non_const = typename std::remove_const<T>::type;

      public:
         using value_type = T;

         ///
         /// @author Peter Robinson
         ///
         /// Default constructor
         ///
         local_ptr() = default;

         ///
         /// @author Alan Dayton
         ///
         /// nullptr constructor
         ///
         CARE_HOST_DEVICE local_ptr(std::nullptr_t) noexcept : local_ptr() {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from raw pointer
         ///
         CARE_HOST_DEVICE local_ptr(T* ptr) noexcept : m_ptr(ptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// Copy constructor
         ///
         local_ptr(local_ptr const &ptr) = default;

         ///
         /// @author Peter Robinson
         ///
         /// Construct from local_ptr containing non-const elements if T is const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         CARE_HOST_DEVICE local_ptr(local_ptr<T_non_const> const &ptr) noexcept : m_ptr(ptr.data()) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from host_ptr
         ///
         CARE_HOST_DEVICE local_ptr(host_ptr<T> const &ptr) noexcept : m_ptr(ptr.data()) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from host_ptr containing non-const elements if T is const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         CARE_HOST_DEVICE local_ptr(host_ptr<T_non_const> const &ptr) noexcept : m_ptr(ptr.cdata()) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray
         ///
         CARE_HOST_DEVICE local_ptr(chai::ManagedArray<T> const &ptr) : m_ptr(ptr.data()) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray containing non-const elements if T is const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         CARE_HOST_DEVICE local_ptr(chai::ManagedArray<T_non_const> const &ptr) : m_ptr(ptr.data()) {}

         ///
         /// Copy assignment operator
         ///
         local_ptr& operator=(const local_ptr& other) = default;

         ///
         /// @author Peter Robinson
         ///
         /// Return the element at the given index
         ///
         CARE_HOST_DEVICE inline T & operator[](int index) { return m_ptr[index]; }

         ///
         /// @author Peter Robinson
         ///
         /// Convert to a raw pointer
         ///
         CARE_HOST_DEVICE operator T*() const { return m_ptr; }

         ///
         /// @author Danny Taller
         ///
         /// Convert to a raw pointer
         ///
         CARE_HOST_DEVICE T* data() const { return m_ptr; }

         ///
         /// @author Alan Dayton
         ///
         /// Convert to a raw pointer
         ///
         CARE_HOST_DEVICE const T* cdata() const { return m_ptr; }

      private:
         T* m_ptr = nullptr; //!< Raw pointer
   };
} // namespace care

#endif // !defined(_CARE_LOCAL_PTR_H_)

