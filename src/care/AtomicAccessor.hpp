#ifndef CARE_ATOMIC_ACCESSOR_HPP
#define CARE_ATOMIC_ACCESSOR_HPP

#include "RAJA/RAJA.hpp"

namespace care {
   template <class T>
   class AtomicRef {
      public:
         explicit AtomicRef(T& ref) : m_ref(ref);

         AtomicRef(const AtomicRef&) = delete;
         AtomicRef& operator=(const AtomicRef&) = delete;

         T atomic_add(T arg) const noexcept {
            return RAJA::atomicAdd<RAJA::auto_atomic>(&m_ref, arg);
         }

      private:
         T& m_ref;
   };

   template <class ElementType>
   struct AtomicAccessor {
      using offset_policy = AtomicAccessor;
      using element_type = ElementType;
      using reference = care::AtomicRef<ElementType>;
      using data_handle_type = ElementType*;

      constexpr AtomicAccessor() noexcept = default;

      constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept {
        return p + i;
      }

      constexpr reference access(data_handle_type p, size_t i) const noexcept {
         return reference(p[i]);
      }
   };
} // namespace care

#endif // CARE_ATOMIC_ACCESSOR_HPP

