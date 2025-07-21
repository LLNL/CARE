//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_HOST_PTR_H_
#define _CARE_HOST_PTR_H_

// CARE headers
#include "care/single_access_ptr.h"

// Other library headers
#include "chai/ManagedArray.hpp"

// Std library headers
#include <cstddef>

#if defined(CARE_ENABLE_STALE_DATA_CHECK) && !defined(CHAI_DISABLE_RM) && (defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE))
#include <cstdlib>
#include <iostream>
#endif

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
         host_ptr() = default;

         ///
         /// @author Peter Robinson
         ///
         /// nullptr constructor
         ///
         host_ptr(std::nullptr_t) noexcept : host_ptr() {}

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
         host_ptr(host_ptr const & ptr) = default;

         ///
         /// @author Peter Robinson
         ///
         /// Construct from host_ptr containing non-const elements if T is const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         host_ptr(host_ptr<T_non_const> const &ptr) noexcept : m_ptr(ptr.data()) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray
         ///
         host_ptr(chai::ManagedArray<T> ptr) : m_ptr(ptr.data(chai::CPU)) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from chai::ManagedArray containing non-const elements if T is const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         host_ptr(chai::ManagedArray<T_non_const> ptr) : m_ptr(ptr.data(chai::CPU)) {}

         ///
         /// Copy assignment operator
         ///
         host_ptr& operator=(const host_ptr& other) = default;

         ///
         /// @author Peter Robinson
         ///
         /// Return the element at the given index
         ///
         inline T & operator[](int index) const {
#if defined(CARE_ENABLE_STALE_DATA_CHECK) && !defined(CHAI_DISABLE_RM) && (defined(CHAI_ENABLE_CUDA) || defined(CHAI_ENABLE_HIP) || defined(CHAI_ENABLE_GPU_SIMULATION_MODE))
            chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
            chai::PointerRecord* record = arrayManager->getPointerRecord((void*) m_ptr);

            if (record != &chai::ArrayManager::s_null_record &&
                record->m_touched[chai::ExecutionSpace::GPU]) {
               const char* name = CHAICallback::getName(record);

               if (name) {
                  std::cout << "[CARE] Error: Found stale data! "
                            << std::string(name) << std::endl;
               }
               else {
                  std::cout << "[CARE] Error: Found stale data!";
               }

               std::abort();
            }
#endif
            return m_ptr[index];
         }

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
         host_ptr& operator +=(Idx i) { m_ptr += i; return *this; }

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
         T* m_ptr = nullptr; //!< Raw host pointer
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

