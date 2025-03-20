//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_DEBUG_H_
#define _CARE_DEBUG_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/host_device_ptr.h"

// Other library headers
#if CARE_HAVE_LLNL_GLOBALID
#include "LLNL_GlobalID.h"
#endif // CARE_HAVE_LLNL_GLOBALID

// Include this file into a source file that is being compiled and linked
// into your executable. You must explicitly instantiate the TV_ttf_display_type
// template function for every type used in your executable.
// tv_data_display.h must be included before this file
// tv_data_display.c must also be compiled and linked into your executable

namespace care {
   template <typename T>
   struct type_name {
      static constexpr const char* value = "value_type";
   };

   template <>
   struct type_name<bool> { static constexpr const char* value = "bool"; };

   template <>
   struct type_name<const bool> { static constexpr const char* value = "const bool"; };

   template <>
   struct type_name<short> { static constexpr const char* value = "short"; };

   template <>
   struct type_name<const short> { static constexpr const char* value = "const short"; };

   template <>
   struct type_name<unsigned short> { static constexpr const char* value = "unsigned short"; };

   template <>
   struct type_name<const unsigned short> { static constexpr const char* value = "const unsigned short"; };

   template <>
   struct type_name<int> { static constexpr const char* value = "int"; };

   template <>
   struct type_name<const int> { static constexpr const char* value = "const int"; };

   template <>
   struct type_name<unsigned int> { static constexpr const char* value = "unsigned int"; };

   template <>
   struct type_name<const unsigned int> { static constexpr const char* value = "const unsigned int"; };

   template <>
   struct type_name<long> { static constexpr const char* value = "long"; };

   template <>
   struct type_name<const long> { static constexpr const char* value = "const long"; };

   template <>
   struct type_name<unsigned long> { static constexpr const char* value = "unsigned long"; };

   template <>
   struct type_name<const unsigned long> { static constexpr const char* value = "const unsigned long"; };

   template <>
   struct type_name<long long> { static constexpr const char* value = "long long"; };

   template <>
   struct type_name<const long long> { static constexpr const char* value = "const long long"; };

   template <>
   struct type_name<unsigned long long> { static constexpr const char* value = "unsigned long long"; };

   template <>
   struct type_name<const unsigned long long> { static constexpr const char* value = "const unsigned long long"; };

   template <>
   struct type_name<float> { static constexpr const char* value = "float"; };

   template <>
   struct type_name<const float> { static constexpr const char* value = "const float"; };

   template <>
   struct type_name<double> { static constexpr const char* value = "double"; };

   template <>
   struct type_name<const double> { static constexpr const char* value = "const double"; };

   template <>
   struct type_name<long double> { static constexpr const char* value = "long double"; };

   template <>
   struct type_name<const long double> { static constexpr const char* value = "const long double"; };

#if CARE_HAVE_LLNL_GLOBALID
   template <>
   struct type_name<globalID> { static constexpr const char* value = GIDTYPENAME; };

   template <>
   struct type_name<const globalID> { static constexpr const char* value = GIDCONSTTYPENAME; };
#endif // CARE_HAVE_LLNL_GLOBALID
} // namespace care

template <typename T>
int TV_ttf_display_type(const care::host_device_ptr<T> * a) {
#if defined(CARE_GPUCC)
   return TV_ttf_format_raw;
#else // defined(CARE_GPUCC)
   // Add a row for the data
   char type[4096];
   snprintf(type, sizeof(type), "%s[%lu]", care::type_name<T>::value, a->size());

   if (TV_ttf_add_row("data", type, a->getActivePointer()) != TV_ttf_ec_ok) {
      return TV_ttf_format_raw;
   }

   return TV_ttf_format_ok;
#endif // defined(CARE_GPUCC)
}

#endif // !defined(_CARE_DEBUG_H_)

