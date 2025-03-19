//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_DEVICE_PTR_H_
#define _CARE_DEVICE_PTR_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/local_ptr.h"
#include "care/single_access_ptr.h"

// CHAI headers
#include "chai/ManagedArray.hpp"

// Std library headers
#include <cstddef>

namespace care {
   ///
   /// @author Peter Robinson
   ///
   /// Designed to be used only on the device. If CARE_GPUCC is defined and
   /// this is dereferenced in a host context, it will produce a compile time error.
   ///
   template <typename T>
   class device_ptr : public single_access_ptr {
      private:
         using T_non_const = typename std::remove_const<T>::type;

      public:
         using value_type = T;

         ///
         /// @author Peter Robinson
         ///
         /// Default constructor
         ///
         CARE_HOST_DEVICE device_ptr() noexcept : m_ptr(nullptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// nullptr constructor
         ///
         CARE_HOST_DEVICE device_ptr(std::nullptr_t) noexcept : m_ptr(nullptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from raw pointer
         ///
         CARE_HOST_DEVICE device_ptr(T* ptr) noexcept : m_ptr(ptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// Copy constructor
         ///
         CARE_HOST_DEVICE device_ptr(device_ptr const &ptr) noexcept : m_ptr(ptr.m_ptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from device_ptr containing non-const elements if we are const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         CARE_HOST_DEVICE device_ptr(device_ptr<T_non_const> const &ptr) noexcept : m_ptr(ptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray
         ///
         CARE_HOST_DEVICE device_ptr(chai::ManagedArray<T> const &ptr) : m_ptr(ptr.data()) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray containing non-const elements if T is const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         CARE_HOST_DEVICE device_ptr(chai::ManagedArray<T_non_const> const &ptr) : m_ptr(ptr.data()) {}

         ///
         /// Copy assignment operator
         ///
         device_ptr& operator=(const device_ptr& other) = default;
         
         ///
         /// @author Peter Robinson
         ///
         /// Explicit conversion to local_ptr
         ///
         explicit operator local_ptr<T>() const { return local_ptr<T>(m_ptr); }

         ///
         /// @author Peter Robinson
         ///
         /// Return the element at the given index
         ///
#ifdef CARE_GPUCC
         CARE_DEVICE 
#else
         CARE_HOST_DEVICE 
#endif
         inline T & operator[](int index) const { return m_ptr[index]; }

         ///
         /// @author Peter Robinson
         ///
         /// Convert to a raw pointer
         ///
#ifdef CARE_GPUCC
         CARE_DEVICE 
#else
         CARE_HOST_DEVICE 
#endif
         operator T*() const { return m_ptr; }

      private:
         T * m_ptr; //!< Raw device pointer
   };

} // namespace care

#endif // !defined(_CARE_DEVICE_PTR_H_)

