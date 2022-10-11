#ifndef CARE_DEFAULT_ARRAY_STORAGE_H
#define CARE_DEFAULT_ARRAY_STORAGE_H

template <class T>
class DefaultArrayStorage {
   public:
      using size_type = std::size_t;
      using reference = T &;

      DefaultArrayStorage() = default;

      DefaultArrayStorage(size_type count) :
         m_size{count},
         m_data{new T[count]} {
      }

      DefaultArrayStorage(DefaultArrayStorage const & other) :
         m_size{other.m_size},
         m_data{new T[other.m_size]} {
         for (size_type i = 0; i < m_size; ++i) {
            m_data[i] = other.m_data[i];
         }
      }

      DefaultArrayStorage(DefaultArrayStorage && other) :
         m_size{other.m_size},
         m_data{other.m_data} {
         other.m_size = 0;
         other.m_data = nullptr;
      }

      ~DefaultArrayStorage() {
         m_size = 0;
         delete[] m_data;
      }

      DefaultArrayStorage & operator=(DefaultArrayStorage const & other) {
         if (this != &other) {
            m_size = other.m_size;
            m_data = new T[m_size];

            for (size_type i = 0; i < m_size; ++i) {
               m_data[i] = other.m_data[i];
            }
         }

         return *this;
      }

      DefaultArrayStorage & operator=(DefaultArrayStorage && other) {
         if (this != &other) {
            m_size = other.m_size;
            m_data = other.m_data;
            
            other.m_size = 0;
            other.m_data = nullptr;
         }
         
         return *this;
      }

      size_type size() const {
         return m_size;
      }

      reference operator[](size_type pos) const {
         return m_data[pos];
      }

      void resize(size_type count) {
         T* newData = new T[count];

         if (m_size < count) {
            for (size_type i = 0; i < m_size; ++i) {
               newData[i] = m_data[i];
            }
         }
         else {
            for (size_type i = 0; i < count; ++i) {
               newData[i] = m_data[i];
            }
         }

         m_size = count;

         delete[] m_data;
         m_data = newData;
      }

   private:
      size_type m_size = 0;
      T* m_data = nullptr;
};

#endif // CARE_DEFAULT_ARRAY_STORAGE
