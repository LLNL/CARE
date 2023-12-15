//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#include <mdspan/mdspan.hpp>

#include <type_traits>


// mfh 2022/08/04: This is based on my comments on reference mdspan
// implementation issue https://github.com/kokkos/mdspan/issues/169.

namespace care {


#if defined(_MDSPAN_COMPILER_MSVC) || defined(__INTEL_COMPILER)
#  define _MDSPAN_RESTRICT_KEYWORD __restrict
#elif defined(__GNUC__) || defined(__clang__)
#  define _MDSPAN_RESTRICT_KEYWORD __restrict__
#else
#  define _MDSPAN_RESTRICT_KEYWORD
#endif

#define _MDSPAN_RESTRICT_POINTER( ELEMENT_TYPE ) ELEMENT_TYPE * _MDSPAN_RESTRICT_KEYWORD

// https://en.cppreference.com/w/c/language/restrict gives examples
// of the kinds of optimizations that may apply to restrict.  For instance,
// "[r]estricted pointers can be assigned to unrestricted pointers freely,
// the optimization opportunities remain in place
// as long as the compiler is able to analyze the code:"
//
// void f(int n, float * restrict r, float * restrict s) {
//   float * p = r, * q = s; // OK
//   while(n-- > 0) *p++ = *q++; // almost certainly optimized just like *r++ = *s++
// }
//
// This is relevant because restrict_accessor<ElementType>::reference is _not_ restrict.
// (It's not formally correct to apply C restrict wording to C++ references.
// However, GCC defines this extension:
//
// https://gcc.gnu.org/onlinedocs/gcc/Restricted-Pointers.html
//
// In what follows, I'll assume that this has a reasonable definition.)
// The idea is that even though p[i] has type ElementType& and not ElementType& restrict,
// the compiler can figure out that the reference comes from a pointer based on p,
// which is marked restrict.
//
// Note that any performance improvements can only be determined by experiment.
// Compilers are not required to do anything with restrict.
// Any use of this keyword is not Standard C++,
// so you'll have to refer to the compiler's documentation,
// look at the assembler output, and do performance experiments.
template<class ElementType>
struct restrict_accessor {
  using offset_policy = restrict_accessor;
  using element_type = ElementType;
  using reference = ElementType&;
  using data_handle_type = _MDSPAN_RESTRICT_POINTER( ElementType );

  constexpr restrict_accessor() noexcept = default;

  MDSPAN_TEMPLATE_REQUIRES(
    class OtherElementType,
    /* requires */ (std::is_convertible<OtherElementType(*)[], element_type(*)[]>::value)
    )
  constexpr restrict_accessor(restrict_accessor<OtherElementType>) noexcept {}

  constexpr reference access(data_handle_type p, size_t i) const noexcept {
    return p[i];
  }
  constexpr typename offset_policy::data_handle_type
  offset(data_handle_type p, size_t i) const noexcept {
    return p + i;
  }
};

// Use int, not size_t, as the index_type.
// Some compilers have trouble optimizing loops with unsigned or 64-bit index types.
using index_type = int;

template<class ElementType>
using restrict_mdspan_1d =
  care::mdspan<ElementType, care::dextents<index_type, 1>, care::layout_right, restrict_accessor<ElementType>>;


} // namespace care
