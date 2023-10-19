#if !defined(CARE_UTILITY_H)
#define CARE_UTILITY_H

#include "care/config.h"

#include <cstddef>
#include <type_traits>

namespace care {
   template <class T>
   CARE_HOST_DEVICE constexpr std::remove_reference_t<T>&& move(T&& t) noexcept {
      return static_cast<typename std::remove_reference<T>::type&&>(t);
   }

   template <class T>
   CARE_HOST_DEVICE constexpr void swap(T& a, T& b) noexcept(
                                                       std::is_nothrow_move_constructible<T>::value &&
                                                       std::is_nothrow_move_assignable<T>::value
                                                    )
   {
      T tmp = move(a);
      a = move(b);
      b = move(tmp);
   }

   template <class T, std::size_t N>
   CARE_HOST_DEVICE constexpr void swap(T (&a)[N], T (&b)[N]) noexcept(std::is_nothrow_swappable_v<T>)
   {
      for (std::size_t i = 0; i < N; ++i) {
         care::swap(a[i], b[i]);
      }
   }
} // namespace care

#endif // !defined(CARE_UTILITY_H)

