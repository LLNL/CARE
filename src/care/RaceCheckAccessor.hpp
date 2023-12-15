#ifndef CARE_RACE_CHECK_ACCESSOR_HPP
#define CARE_RACE_CHECK_ACCESSOR_HPP

#include "care/config.h"

#include <iostream>
#include <string>

namespace care {
   template <class T>
   class RaceRef {
      public:
         CARE_HOST_DEVICE RaceRef(T& ref, bool& write);

         // Write all the assignment operators as update
         CARE_HOST_DEVICE RaceRef& operator=(const T& other) {
            m_ref = other;
            m_write = true;
            return *this;
         }

         CARE_HOST_DEVICE T operator+(const T& other) {
            return m_ref + other;
         }

      private:
         T& m_ref;
         bool& m_write;
   };

   template <class ElementType>
   struct RaceCheckAccessor {
      using offset_policy = RaceCheckAccessor;
      using element_type = ElementType;
      using reference = RaceRef<ElementType>;
      using data_handle_type = ElementType*;

      RaceCheckAccessor(size_t* m_threadIDs,
                        bool* read,
                        bool* written,
                        bool* potentialRace,
                        bool& race) noexcept
         : m_threadIDs{threadIDs},
           m_read{read},
           m_written{written},
           m_potentialRace{potentialRace},
           m_race{race}
      {}

      CARE_HOST_DEVICE ~RaceCheckAccessor() {
         for (size_t i = 0; i < m_size; ++i) {
            if (m_potentialRace[i] && m_written[i]) {
               m_race = true;
               break;
            }
         }
      }

      CARE_HOST_DEVICE constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept {
        return p + i;
      }

      CARE_HOST_DEVICE constexpr reference access(data_handle_type p, size_t i) const noexcept {
         if (!m_read[i] && !m_written[i]) {
            m_threadIDs[i] = blockIdx.x*blockDim.x + threadIdx.x;
         }
         else if (m_threadIDs[i] != blockIdx.x*blockDim.x + threadIdx.x) {
            m_potentialRace[i] = true;
         }

         m_read[i] = true;
         return p[i];
      }

      private:
         size_t* m_threadIDs = nullptr;
         bool* m_read = nullptr;
         bool* m_written = nullptr;
         bool* m_potentialRace = nullptr;
         bool& m_race = false;
   };
} // namespace care

#endif // CARE_RACE_CHECK_ACCESSOR_HPP

