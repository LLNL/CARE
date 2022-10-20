//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_HOST_DEVICE_PTR_H_
#define _CARE_HOST_DEVICE_PTR_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
#include "care/Accessor.h"
#include "care/CHAICallback.h"
#include "care/DefaultMacros.h"
#include "care/ExecutionSpace.h"
#include "care/util.h"

// Other library headers
#include "chai/ManagedArray.hpp"

// Std library headers
#include <cstddef>

namespace care {
   ///////////////////////////////////////////////////////////////////////////
   /// @struct _kv
   /// @author Peter Robinson
   /// @brief The key value pair struct used by the sequential version of
   ///    KeyValueSorter
   ///////////////////////////////////////////////////////////////////////////
   template <typename KeyType, typename ValueType>
   struct _kv {
      KeyType key;
      ValueType value;

      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Less than comparison operator
      /// Used as a comparator in the STL
      /// @param right - right _kv to compare
      /// @return true if this value is less than right's value, false otherwise
      ///////////////////////////////////////////////////////////////////////////
      inline bool operator <(_kv const && right) { return value < right.value; };
      ///////////////////////////////////////////////////////////////////////////
      /// @author Peter Robinson
      /// @brief Equality operator
      /// Used as a comparator care/Accessor.h::detectRaceCondition
      /// @param right - right _kv to compare
      /// @return true if this value is less than right's value, false otherwise
      ///////////////////////////////////////////////////////////////////////////
      inline bool operator ==(_kv const & right) const { return value == right.value && key == right.key;  };
   };

   ///
   /// @author Alan Dayton
   ///
   /// @brief Overload operator<< for the _kv struct
   ///
   /// @param[in] os   The output stream
   /// @param[in] kv   The struct to process
   ///
   /// @return   The output stream for chaining of operations
   ///
   template <typename KeyType, typename ValueType>
   inline std::ostream& operator<<(std::ostream& os, const _kv<KeyType, ValueType>& kv) {
      os << kv.key << ": " << kv.value;
      return os;
   }

   ///
   /// @author Peter Robinson, Ben Liu, Alan Dayton, Arlie Capps
   ///
   template <typename T, template <class Anyfoo> class Accessor=RaceConditionAccessor>
   class host_device_ptr : public chai::ManagedArray<T>, Accessor<T> {
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
      CARE_HOST_DEVICE host_device_ptr<T, Accessor<T>>() noexcept : MA(), Accessor<T>() {}

      ///
      /// @author Peter Robinson
      ///
      /// nullptr constructor
      ///
      CARE_HOST_DEVICE host_device_ptr<T, Accessor<T>>(std::nullptr_t from) noexcept : MA (from), Accessor<T>() {}

#if defined(CARE_ENABLE_IMPLICIT_CONVERSIONS)
      ///
      /// @author Peter Robinson
      ///
      /// Construct from a raw pointer
      ///
      /// @note Only safe if the raw pointer is already registered with CHAI
      ///
      template <bool Q = 0>
      CARE_HOST_DEVICE host_device_ptr<T, Accessor<T>>(
         T * from, //!< Raw pointer to construct from
         chai::CHAIDISAMBIGUATE name=chai::CHAIDISAMBIGUATE(), //!< Used to disambiguate this constructor
         bool foo=Q) //!< Used to disambiguate this constructor
      : MA(from, name, foo) , Accessor<T>() {Accessor<T>::set_data(MA::data(chai::CPU, false));}
#endif

      ///
      /// @author Peter Robinson
      ///
      /// Copy constructor
      ///
      CARE_HOST_DEVICE host_device_ptr<T, Accessor<T>>(host_device_ptr<T> const & other) : MA (other) , Accessor<T>(other) {}

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a chai::ManagedArray
      ///
      CARE_HOST_DEVICE host_device_ptr<T,Accessor<T>>(MA const & other) : MA (other) , Accessor<T>() {Accessor<T>::set_data(MA::data(chai::CPU, false));}

      ///
      /// @author Peter Robinson
      /// @note   cannot make this CARE_HOST_DEVICE due to use of standard library.
      //          The test file TestArrayUtils.cpp gives a warning about calling a
      //          __host__ function from a __host__ __device__ function when compiled on with CUDA.
      /// Construct from a chai::ManagedArray containing non-const elements
      ///
      template <bool B = std::is_const<T>::value,
                typename std::enable_if<B, int>::type = 1>
      host_device_ptr<T, Accessor<T>>(MAU const & other) : MA (other), Accessor<T>() {Accessor<T>::set_data(MA::data(chai::CPU, false));}

#if defined (CHAI_DISABLE_RM)
      ///
      /// @author Peter Robinson
      ///
      /// Construct from a raw pointer, size, and name
      /// This is defined when the CHAI resource manager is disabled
      ///
      host_device_ptr<T, Accessor<T>>(T* from, size_t size, const char * name)
         : MA(from, nullptr, size, nullptr), Accessor<T>(size)
      {
         Accessor<T>::set_data(MA::data(chai::CPU, false));
      }
#else
      ///
      /// @author Peter Robinson
      ///
      /// Construct from a raw pointer, size, and name
      /// This is defined when the CHAI resource manager is enabled
      ///
      host_device_ptr<T, Accessor<T>>(T* from, size_t size, const char * name)
         : MA(size == 0 ? nullptr : from,
              chai::ArrayManager::getInstance(),
              size,
              chai::ArrayManager::getInstance()->getPointerRecord((void *) (size == 0 ? nullptr : from))),
           Accessor<T>(size)
      {
         registerCallbacks(name);
         sanityCheckRecords((void *) from, MA::m_pointer_record);

         // This fix probably belongs in CHAI proper, but there is a reasonable debate as
         // to what is more "correct", constructor wins or previous state wins. CARE is
         // taking the stance that constructor wins, regardless of what CHAI decides.
         if ( size > 0 && from != nullptr) {
            chai::ArrayManager::getInstance()->getPointerRecord((void *)from)->m_size = size*sizeof(T);
         }
      }
#endif

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a size and name
      ///
      host_device_ptr<T, Accessor<T>>(size_t size, const char * name) : MA (size), Accessor<T>(size){
         registerCallbacks(name);
         Accessor<T>::set_data(MA::data(chai::CPU,false));
      }

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a size, initial value, and name
      /// Optionally inititialize on device rather than the host
      ///
      CARE_HOST_DEVICE host_device_ptr<T, Accessor<T>>(size_t size, T initial, const char * name, bool initOnDevice=false) : MA (size), Accessor<T>(size) { 
         registerPointerName(name); 
         initialize(size, initial, 0, initOnDevice);
         Accessor<T>::set_data(MA::data(chai::CPU,false));
      }

      ///
      /// @author Peter Robinson
      ///
      /// Convert to a host_device_ptr containing const elements
      ///
      template<bool B = std::is_const<T>::value,
               typename std::enable_if<!B, int>::type = 0>
      CARE_HOST_DEVICE operator host_device_ptr<const T, Accessor> () const {
         return *reinterpret_cast<host_device_ptr<const T, Accessor> const *> (this);
      }

#if defined(CARE_ENABLE_BOUNDS_CHECKING)
      template <class Index>
      inline void boundsCheck(const Index i) const {
         if (i < 0
#if !defined (CHAI_DISABLE_RM)
             || i >= (Index) (MA::m_pointer_record->m_size / sizeof(T))
#endif
            ) {
            const char* name = CHAICallback::getName(MA::m_pointer_record);

            if (name) {
               std::cerr << "[CARE] Error: Index " << i << " is out of bounds for array '" << std::string(name) << "'!" << std::endl;
            }
            else {
               std::cerr << "[CARE] Error: Index " << i << " is out of bounds for array!" << std::endl;
            }
         }
      }
#endif

      ///
      /// Copy assignment operator
      ///
      host_device_ptr& operator=(const host_device_ptr& other) = default;

      ///
      /// @author Peter Robinson
      ///
      /// Return the value at the given index
      ///
     template<typename Idx>
     inline CARE_HOST_DEVICE T& operator[](const Idx i) const {
#if !defined(CARE_DEVICE_COMPILE) && defined(CARE_ENABLE_BOUNDS_CHECKING)
         boundsCheck(i);
#endif
         Accessor<T>::operator[](i);
         return MA::operator[](i);
      }

      host_device_ptr<T> & realloc(size_t elems) {
         // If the managed array is empty, we register the callback on reallocation.
         bool doRegisterCallback = (MA::m_elems == 0 && MA::m_active_base_pointer == nullptr);
         MA::reallocate(elems);
         Accessor<T>::set_size(elems);
         Accessor<T>::set_data(MA::data(chai::CPU,false));
         if (doRegisterCallback) {
            registerCallbacks();
         }
         return *this;
      }

      void alloc(size_t elems) {
         MA::allocate(elems);
         Accessor<T>::set_size(elems);
         Accessor<T>::set_data(MA::data(chai::CPU,false));
         registerCallbacks();
      }

      void registerPointerName(const char * name = nullptr) const {
#if !defined(CHAI_DISABLE_RM)
         if (CHAICallback::isActive()) {
            const chai::PointerRecord * pointer_record = MA::m_pointer_record;
            std::string pointerName;

            if (name == nullptr) {
               const char* temp = CHAICallback::getName(pointer_record);

               if (temp) {
                  pointerName = temp;
               }
               else {
                  pointerName = "UNKNOWN";
               }
            }
            else {
               pointerName = name;
            }

            // Only use the folder and file name for friendlier diffs (i.e. util/Util.cxx)
            if (CHAICallback::isDiffFriendly() && pointerName != "UNKNOWN") {
#ifdef WIN32
               size_t position = pointerName.rfind("\\") - 1;

               if (position != std::string::npos) {
                  position = pointerName.rfind("\\", position) + 1;
               }
#else
               size_t position = pointerName.rfind("/") - 1;

               if (position != std::string::npos) {
                  position = pointerName.rfind("/", position) + 1;
               }
#endif
               if (position != std::string::npos && position < pointerName.size()) {
                  pointerName = pointerName.substr(position);
               }
            }

            CHAICallback::setName(pointer_record, pointerName);
            CHAICallback::setTypeIndex(pointer_record, typeid(T));
         }
#endif
      }

      void registerCallbacks(const char * name = nullptr) {
#if !defined(CHAI_DISABLE_RM)
         if (CHAICallback::isActive()) {
            registerPointerName(name);

            /* we capture the pointers instead of the values so that it is runtime
             * conditions that determine behavior instead of instantiation time
             * conditions. */
            const chai::PointerRecord * pointer_record = MA::m_pointer_record;
            MA::setUserCallback(CHAICallback(pointer_record));
         }
#endif
      }

      void sanityCheckRecords(void * pointer, const chai::PointerRecord * pointer_record) const {
#if !defined(CHAI_DISABLE_RM) && defined(CARE_DEBUG)
         void * allocationPtr = MA::m_resource_manager->frontOfAllocation(pointer);
         if (pointer == allocationPtr) {
            // we are happy
         }
         // if we are not nullptr and allocation is nullptr, then we don't have a valid umpire allocation
         // (should not happen since the chai constructors are supposed to register unknown pointers with
         // umpire)
         else if (allocationPtr  == nullptr) {
            const char* name = CHAICallback::getName(pointer_record);

            if (name) {
               printf("[CARE] %p %s associated with null allocation record!\n",
                      pointer, name);
            }
            else {
               printf("[CARE] %p associated with null allocation record!\n",
                      pointer);
            }
         }
         // sad if they are different and we are not a slice
         else if (!MA::m_is_slice) {
            const char* name = CHAICallback::getName(pointer_record);

            if (name) {
               printf("[CARE] %p %s associated with allocation record %p %s !\n",
                      pointer, name, allocationPtr, name);
            }
            else {
               printf("[CARE] %p associated with allocation record %p !\n",
                      pointer, allocationPtr);
            }
         }
         // we are happy
#else // !defined(CHAI_DISABLE_RM) && defined(CARE_DEBUG)
         // Quiet the compiler warnings
         (void) pointer;
         (void) pointer_record;
#endif // !defined(CHAI_DISABLE_RM) && defined(CARE_DEBUG)
         return;
      }

      const char* getName() const {
         return CHAICallback::getName(MA::m_pointer_record);
      }

      ///
      /// Names this object and also updates the size.
      /// Trick for when these pointers are cast to different underlying types.
      ///
      void namePointer(const char * name) {
         // Register the callbacks with the new name
         registerCallbacks(name);

#if !defined(CHAI_DISABLE_RM)
         // Update the size if cast from a different underlying data type.
         if (MA::m_pointer_record && MA::m_pointer_record->m_size > 0) {
            MA::m_elems = MA::m_pointer_record->m_size / sizeof(T);

            if (MA::m_elems * sizeof(T) != MA::m_pointer_record->m_size) {
               fprintf(stderr, "[CARE] host_device_ptr<T>::namePointer performed an unsafe cast to a different underlying type. Expect errors!\n");
            }
         }
#endif
      }

      void initialize(const size_t N, const T initial,
                      const int startIndx = 0,
                      const bool initOnDevice = false) {
         registerCallbacks();
         MA & me = *this;
         if (initOnDevice) { 
            CARE_STREAM_LOOP(i,startIndx,N) {
               me[i] = initial;
            } CARE_STREAM_LOOP_END
         }
         else {
            CARE_SEQUENTIAL_LOOP(i,startIndx,N) {
               me[i] = initial;
            } CARE_SEQUENTIAL_LOOP_END
         }
      }

      void freeDeviceMemory(bool deregisterPointer=true) {
#if !defined(CHAI_DISABLE_RM) 
#if defined(CHAI_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE
         MA::move(chai::CPU);
         MA::free(chai::GPU);
#endif
         if (deregisterPointer) {
            auto arrayManager = chai::ArrayManager::getInstance();
            arrayManager->deregisterPointer(MA::m_pointer_record,true);
            CHAICallback::deregisterRecord(MA::m_pointer_record);
         }
#endif
      }

      CARE_HOST_DEVICE void pick(int idx, T_non_const& val) const  {
#if !defined(CARE_DEVICE_COMPILE) && defined(CARE_ENABLE_BOUNDS_CHECKING)
         boundsCheck(idx);
#endif
         val = MA::pick((size_t) idx);
      }

      CARE_HOST_DEVICE T pick(int idx) const {
#if !defined(CARE_DEVICE_COMPILE) && defined(CARE_ENABLE_BOUNDS_CHECKING)
         boundsCheck(idx);
#endif
         return MA::pick((size_t) idx);
      }

      CARE_HOST T* getPointer(ExecutionSpace space, bool moveToSpace = true) {
         return MA::getPointer(chai::ExecutionSpace((int)space), moveToSpace);
      }

      CARE_HOST void registerTouch(ExecutionSpace space) {
         MA::registerTouch(chai::ExecutionSpace((int) space));
      }

      CARE_HOST void move(ExecutionSpace space) {
         MA::move(chai::ExecutionSpace((int) space));
      }
   }; // class host_device_ptr
} // namespace care


#endif // !defined(_CARE_HOST_DEVICE_PTR_H_)

