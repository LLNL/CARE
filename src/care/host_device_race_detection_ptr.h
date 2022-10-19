//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_HOST_DEVICE_RACE_DETECTION_PTR_H_
#define _CARE_HOST_DEVICE_RACE_DETECTION_PTR_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/CHAICallback.h"
#include "care/DefaultMacros.h"
#include "care/ExecutionSpace.h"
#include "care/host_device_ptr.h"
#include "care/util.h"

// Other library headers
#include "chai/ManagedArray.hpp"

// Std library headers
#include <cstddef>

namespace care {
   ///
   /// @author Peter Robinson, Ben Liu, Alan Dayton, Arlie Capps
   ///
   template <typename T>
   class host_device_race_detection_ptr : host_device_ptr<T> {
     private:
      using T_non_const = typename std::remove_const<T>::type;
      using MA = chai::ManagedArray<T>;
      using MAU = chai::ManagedArray<T_non_const>;
      using HDP = host_device_ptr<T>;
      using HDPU = host_device_ptr<T_non_const>;

     public:
      using value_type = T;

      
      
      /// metadat used to track access patterns 
      host_device_ptr<int> m_read_by_thread_ids;
      host_device_ptr<int> m_written_to_thread_ids;
      ///
      /// @author Peter Robinson
      ///
      /// Default constructor
      ///
      CARE_HOST_DEVICE host_device_race_detection_ptr <T>() noexcept : HDP() , m_read_by_thread_ids(std::nullptr), m_written_to_thread_ids(std::nullptr) {}

      ///
      /// @author Peter Robinson
      ///
      /// nullptr constructor
      ///
      CARE_HOST_DEVICE host_device_race_detection_ptr<T>(std::nullptr_t from) noexcept : HDP (from) , m_read_by_thread_ids(from), m_written_to_thread_ids(from) {}

#if defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)
      ///
      /// @author Peter Robinson
      ///
      /// Construct from a raw pointer
      ///
      /// @note Only safe if the raw pointer is already registered with CHAI
      ///
      template <bool Q = 0>
      CARE_HOST_DEVICE host_device_race_detection_ptr<T>(
         T * from, //!< Raw pointer to construct from
         chai::CHAIDISAMBIGUATE name=chai::CHAIDISAMBIGUATE(), //!< Used to disambiguate this constructor
         bool foo=Q) //!< Used to disambiguate this constructor
      : MA(from, name, foo) , m_read_by_thread_ids(nullptr), m_written_to_thread_ids(nullptr){}
#endif

      ///
      /// @author Peter Robinson
      ///
      /// Copy constructor
      ///
      CARE_HOST_DEVICE host_device_race_detection_ptr<T>(host_device_race_detection_ptr<T> const & other) : HDP (other), m_read_by_thread_ids(other.m_read_by_thread_ids), m_written_to_thread_ids(other.m_written_to_thread_ids) {}

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a chai::ManagedArray
      ///
      CARE_HOST_DEVICE host_device_race_detection_ptr<T>(MA const & other) : MA (other), m_read_by_thread_ids(nullptr), m_written_to_thread_ids(nullptr) {}

      ///
      /// @author Peter Robinson
      /// @note   cannot make this CARE_HOST_DEVICE due to use of standard library.
      //          The test file TestArrayUtils.cpp gives a warning about calling a
      //          __host__ function from a __host__ __device__ function when compiled on with CUDA.
      /// Construct from a chai::ManagedArray containing non-const elements
      ///
      template <bool B = std::is_const<T>::value,
                typename std::enable_if<B, int>::type = 1>
      host_device_race_detection_ptr<T>(MAU const & other) : HDP (other) , m_read_by_thread_ids(nullptr), m_written_to_thread_ids(nullptr) {}

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a raw pointer, size, and name
      /// This is defined when the CHAI resource manager is disabled
      ///
      host_device_race_detection_ptr<T>(T* from, size_t size, const char * name)
         : HDP (from, nullptr, size, nullptr), m_read_by_thread_ids(nullptr), m_written_to_thread_ids(nullptr) {}
      {
      }

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a size and name
      ///
      host_device_race_detection_ptr<T>(size_t size, const char * name) : HDP (size, name), m_read_by_thread_ids(nullptr), m_written_to_thread_ids(nullptr)  {}

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a size, initial value, and name
      /// Optionally inititialize on device rather than the host
      ///
      CARE_HOST_DEVICE host_device_race_detection_ptr<T>(size_t size, T initial, const char * name, bool initOnDevice=false) : HDP (size, initial, name, initOnDevice) { }

      ///
      /// @author Peter Robinson
      ///
      /// Convert to a host_device_race_detection_ptr containing const elements
      ///
      template<bool B = std::is_const<T>::value,
               typename std::enable_if<!B, int>::type = 0>
      CARE_HOST_DEVICE operator host_device_race_detection_ptr<const T> () const {
         return *reinterpret_cast<host_device_race_detection_ptr<const T> const *> (this);
      }

      ///
      /// Copy assignment operator
      ///
      host_device_race_detection_ptr & operator=(const host_device_race_detection_ptr & other) = default;

      ///
      /// @author Peter Robinson
      ///
      /// Return the value at the given index
      ///
      template<typename Idx>
      inline CARE_HOST_DEVICE T& operator[](const Idx i)
#if !CARE_LEGACY_COMPATIBILITY_MODE
      const
#endif
      {
         return HDP::operator[](i);
      }

#if CARE_LEGACY_COMPATIBILITY_MODE
     template<typename Idx>
     inline CARE_HOST_DEVICE T& operator[](const Idx i) const {
        return HDP::operator[](i);
     }
#endif

      CARE_HOST_DEVICE void pick(int idx, T_non_const& val) const  {
         val = HDP::pick(idx);
      }

      CARE_HOST_DEVICE T pick(int idx) const {
         return HDP::pick((size_t) idx);
      }

   }; // class host_device_race_detection_ptr

} // namespace care

#if defined(CARE_ENABLE_MANAGED_PTR)
#if !defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)

// TODO: Declaring these functions causes problems with a project that depends on CARE
//       and has not eliminated implicit casts. Having this macro guard around these
//       functions is a temporary workaround. A better solution needs to be found.
//       Perhaps having an object wrapper like chai::ManagedDataSplitter would be
//       a good way to indicate to chai::make_managed that raw pointers actually
//       should be extracted. Basically, we break if a constructor takes both
//       c-style arrays and ManagedArrays/host_device_ptrs. There is a reproducer
//       of this issue in the reproducers directory.

///
/// @author Danny Taller
///
/// This implementation of getRawPointers handles the CARE host_device_ptr type.
/// Note that without this function, care host_device_ptrs will go to the The non-CHAI type
/// version of this function (NOT the ManagedArray version), so the make managed paradigm
/// will fail unless implicit conversions are allowed. See also similar getRawPointers
/// functions within CHAI managed_ptr header.
///
/// @param[in] arg The host_device_ptr from which to extract a raw pointer
///
/// @return arg cast to a raw pointer
/// 
namespace chai {
   namespace detail {
      template <typename T>
      CARE_HOST_DEVICE T* getRawPointers(care::host_device_race_detection_ptr<T> arg) {
         return arg.data();
      }
   } // namespace detail
} // namespace chai

#endif // !defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)
#endif // defined(CARE_ENABLE_MANAGED_PTR)

#endif // !defined(_CARE_HOST_DEVICE_RACE_DETECTION_PTR_H_)

