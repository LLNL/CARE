#ifndef CARE_STALE_CHECK_ACCESSOR_HPP
#define CARE_STALE_CHECK_ACCESSOR_HPP

#include <iostream>
#include <string>

namespace care {
   template <class ElementType>
   struct StaleCheckAccessor {
      using offset_policy = StaleCheckAccessor;
      using element_type = ElementType;
      using reference = ElementType&;
      using data_handle_type = ElementType*;

      StaleCheckAccessor(const bool* touchedOnDevice,
                         const std::string& label) noexcept
         : m_touched_on_device{touchedOnDevice},
           m_label{label}
      {}

      template <class OtherElementType>
      constexpr StaleCheckAccessor(StaleCheckAccessor<OtherElementType>) noexcept {}

      ~StaleCheckAccessor() {
         if (m_stale) {
            std::cout << "[CARE] Stale data: " << m_label << "\n";
         }
      }

      constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept {
        return p + i;
      }

      constexpr reference access(data_handle_type p, size_t i) const noexcept {
         if (m_touched_on_device && *m_touched_on_device) {
            m_stale = true;
         }

         return p[i];
      }

      private:
         const bool* m_touched_on_device = nullptr;
         std::string m_label;
         mutable bool m_stale = false;
   };
} // namespace care

#endif // CARE_STALE_CHECK_ACCESSOR_HPP

