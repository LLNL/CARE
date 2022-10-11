#ifndef CARE_ARRAY_H
#define CARE_ARRAY_H

namespace care {
   template <class T,
             template class <class> Storage>
   class Array : public Storage<T> {
      public:
         Array() = default;
         Array(Array const &) = default;
         Array(Array &&) = default;
         ~Array() = default;

         Array& operator=(Array const &) = default;
         Array& operator=(Array &&) = default;

         Storage<T>::size_type size() const noexcept {
            return m_data.size();
         }

         Storage<T>::reference operator[](Storage<T>::size_type pos) const {
            return m_data[pos];
         }

      private:
         Storage<T> m_data;
   };
} // namespace care

#endif //
