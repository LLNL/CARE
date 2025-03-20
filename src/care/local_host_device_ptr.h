//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_LOCAL_HOST_DEVICE_PTR_H_
#define _CARE_LOCAL_HOST_DEVICE_PTR_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/host_device_ptr.h"

// CHAI headers
#include "chai/ManagedArray.hpp"

// Std library headers
#include <cstddef>

namespace care {
   // This is use to prevent clang-query from giving a false positive error for use
   // of operator[] from within a HOSTDEV function that is called from within a RAJA
   // loop. The user is responsible for ensuring that this is indeed true. If this is
   // used outside of a RAJA context, this may be a bug.
   // This should really only be used for multidimensional managed arrays.
   // TODO restrict this to MD arrays using template metaprogramming.
   template <typename T>
   class local_host_device_ptr : public host_device_ptr<T> {
      private:
         using T_non_const = typename std::remove_const<T>::type;
         using MA = chai::ManagedArray<T>;
         using MAU = chai::ManagedArray<T_non_const>;

      public:
         using value_type = T;

         ///
         /// @author Peter Robinson
         ///
         /// Default constructor
         ///
         CARE_HOST_DEVICE local_host_device_ptr() noexcept : host_device_ptr<T>() {}

         ///
         /// @author Alan Dayton
         ///
         /// nullptr constructor
         ///
         CARE_HOST_DEVICE local_host_device_ptr(std::nullptr_t) noexcept : host_device_ptr<T>(nullptr) {}

         ///
         /// @author Peter Robinson
         ///
         /// Copy constructor
         ///
         CARE_HOST_DEVICE local_host_device_ptr(local_host_device_ptr const & other) : host_device_ptr<T>(other) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from local_host_device_ptr containing non-const elements if we are const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         CARE_HOST_DEVICE local_host_device_ptr(local_host_device_ptr<T_non_const> const & other)
            : host_device_ptr<T>(other) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from host_device_ptr
         ///
         CARE_HOST_DEVICE local_host_device_ptr(host_device_ptr<T> const & other)
            : host_device_ptr<T> (other) {}

         ///
         /// @author Peter Robinson
         ///
         /// Construct from host_device_ptr containing non-const elements if we are const
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<B, int>::type = 1>
         CARE_HOST_DEVICE local_host_device_ptr(host_device_ptr<T_non_const> const & other)
            : host_device_ptr<T>(other) {}

         ///
         /// @author Peter Robinson
         ///
         /// Convert to local_host_device_ptr<const T>
         ///
         template <bool B = std::is_const<T>::value,
                   typename std::enable_if<!B, int>::type = 0>
         CARE_HOST_DEVICE operator local_host_device_ptr<const T> () const{
#if !defined(CARE_DEVICE_COMPILE)
#if defined(CHAI_DISABLE_RM)
             return host_device_ptr<const T>(const_cast<const T*>(MA::m_host_ptr));
#else

             return host_device_ptr<const T>(const_cast<const T*>(MA::m_host_ptr),
                                             const_cast<const T*>(MA::m_device_ptr),
                                             MA::m_ArrayManager,
                                             MA::getName());
#endif // CHAI_DISABLE_RM
#else
             return host_device_ptr<const T>(const_cast<const T*>(MA::m_device_ptr));
#endif
         }

         ///
         /// @author Peter Robinson
         ///
         /// Return the element at the given index
         ///
         template<typename Idx>
         inline CARE_HOST_DEVICE T& operator[](const Idx i) {
            return MA::operator[](i);
         }
   };
} // namespace care

#endif // !defined(_CARE_LOCAL_HOST_DEVICE_PTR_H_)

