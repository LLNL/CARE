//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_HOST_DEVICE_PTR_H_
#define _CARE_HOST_DEVICE_PTR_H_

// CARE config header
#include "care/config.h"

// Other CARE headers
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
      /// Used as a comparator
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
   template <typename T>
   class host_device_ptr : public chai::ManagedArray<T> {
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
      CARE_HOST_DEVICE host_device_ptr() noexcept : MA() {}

      ///
      /// @author Peter Robinson
      ///
      /// nullptr constructor
      ///
      CARE_HOST_DEVICE host_device_ptr(std::nullptr_t from) noexcept : MA (from) {}

      ///
      /// @author Peter Robinson
      ///
      /// Copy constructor
      ///
      CARE_HOST_DEVICE host_device_ptr(host_device_ptr<T> const & other) : MA (other) {}

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a chai::ManagedArray
      ///
      CARE_HOST_DEVICE host_device_ptr(MA const & other) : MA (other) {}

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a chai::ManagedArray containing non-const elements
      ///
      template <bool B = std::is_const<T>::value,
                typename std::enable_if<B, int>::type = 1>
      CARE_HOST_DEVICE host_device_ptr(MAU const & other)
         : MA (other)
      {
      }

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a raw pointer, size, and name
      /// This is defined when the CHAI resource manager is disabled
      ///
#if defined(CARE_DEEP_COPY_RAW_PTR)
      host_device_ptr(T* from, size_t size, const char * name)
         : MA(size)
      {
         std::copy_n(from, size, (T_non_const*)MA::data());
      }
#else /* defined(CARE_DEEP_COPY_RAW_PTR) */
#if defined (CHAI_DISABLE_RM) || defined(CHAI_THIN_GPU_ALLOCATE)
      host_device_ptr(T* from, size_t size, const char * name)
         : MA(from, nullptr, size, nullptr)
      {
      }
#else
      ///
      /// @author Peter Robinson
      ///
      /// Construct from a raw pointer, size, and name
      /// This is defined when the CHAI resource manager is enabled
      ///
      host_device_ptr(T* from, size_t size, const char * name)
         : MA(size == 0 ? nullptr : from,
              chai::ArrayManager::getInstance(),
              size,
              chai::ArrayManager::getInstance()->getPointerRecord((void *) (size == 0 ? nullptr : from)))
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
#endif /* defined(CARE_DEEP_COPY_RAW_PTR) */

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a size and name
      ///
      host_device_ptr(size_t size, const char * name) : MA (size) {
         registerCallbacks(name);
      }

      ///
      /// @author Peter Robinson
      ///
      /// Construct from a size, initial value, and name
      /// Optionally inititialize on device rather than the host
      ///
      CARE_HOST_DEVICE host_device_ptr(size_t size, T initial, const char * name, bool initOnDevice=false) : MA (size) {
         registerPointerName(name); 
         initialize(size, initial, 0, initOnDevice);
      }

      ///
      /// @author Peter Robinson
      ///
      /// Convert to a host_device_ptr<const T>
      ///
      template<bool B = std::is_const<T>::value,
               typename std::enable_if<!B, int>::type = 0>
      CARE_HOST_DEVICE operator host_device_ptr<const T> () const {
         return *reinterpret_cast<host_device_ptr<const T> const *> (this);
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
         return MA::operator[](i);
      }

      host_device_ptr<T> & realloc(size_t elems) {
         // If the managed array is empty, we register the callback on reallocation.
         bool doRegisterCallback = (MA::m_size == 0 && MA::m_active_base_pointer == nullptr);
         MA::reallocate(elems);
         if (doRegisterCallback) {
            registerCallbacks();
         }
         return *this;
      }

      void alloc(size_t elems) {
         MA::allocate(elems);
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
         // Let the pointer record be source of truth for the size.
         if (MA::m_pointer_record && MA::m_pointer_record->m_size > 0) {
            MA::m_size = MA::m_pointer_record->m_size;
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

      // frees device memory, ensuring that *CPU_destination is updated with valid CPU data.
      // if CPU_destination is nullptr, that indicates CPU data is not needed so no work should be done to get it.
      //    This best supports use cases where the user already has a handle on CPU data that they know is up to date.
      // if *CPU_destination is nullptr, the semantics is 0-copy if you can, so after this call *CPU_destination will
      //     be aliased to the CPU data of this host_device_ptr. if CHAI_THIN_GPU_ALLOCATE is defined, this is the same
      //     as the GPU data. It's up to the user to sort out whether that pointer needs to be freed with umpire calls
      //     or with raw free commands.
      // if *CPU_destination is not nullptr and is a different address from the host_device_ptr CPU data, then
      //     *CPU_destination will be updated with a deep copy of this data
      //
      // TODO: Should this really live in chai::ManagedArray?
      void freeDeviceMemory(T_non_const ** CPU_destination,
                            size_t elems,
                            bool deregisterPointer=true) {
#if defined(CARE_DEEP_COPY_RAW_PTR)
         // if there is a pointer to update ...
         if (CPU_destination != nullptr) {
            if (*CPU_destination == nullptr) {
               *CPU_destination = (T_non_const *) std::malloc(elems*sizeof(T));
            }
            std::copy_n(MA::cdata(), elems, *CPU_destination);
         }
         MA::free();
#else /* defined(CARE_DEEP_COPY_RAW_PTR) */
#if !defined(CHAI_DISABLE_RM) 
#if defined(CHAI_GPUCC) || CARE_ENABLE_GPU_SIMULATION_MODE
         if (CPU_destination != nullptr) {
            MA::move(chai::CPU);

            // if our active pointer is different than the CPU destination
            if (MA::m_active_pointer != *CPU_destination) {
               // and the cpu destination is nullptr,
               if (*CPU_destination == nullptr) {
                  // semantics is moving our pointer to that pointer
                  *CPU_destination = const_cast<T_non_const *> (MA::m_active_pointer);
               }
               // if the CPU destination is not nullptr
               else {
                  // semantics is copying from our pointer to the CPU_destination
                  std::copy_n(MA::m_active_pointer, elems, *CPU_destination);
               }
            }
            // otherwise our active pointer is the cpu destination so we don't
            // have to do anything other than the move that just happened
         }

         MA::free(chai::GPU);
#endif
         if (deregisterPointer) {
            auto arrayManager = chai::ArrayManager::getInstance();
            arrayManager->deregisterPointer(MA::m_pointer_record,true);
            CHAICallback::deregisterRecord(MA::m_pointer_record);
         }
         
#else // no resource manager active
#if defined(CHAI_THIN_GPU_ALLOCATE) // GPU allocated thin wrapped
         // ... then sync to ensure data is up to date
         // this needs to be called even without a CPU_destination in case the pointer is reused
         chai::ArrayManager::getInstance()->syncIfNeeded();
#endif
         // if there is a pointer to update ...
         if (CPU_destination != nullptr) {
            // if our active pointer is different than the CPU destination
            if (MA::m_active_pointer != *CPU_destination) {
               // and the cpu destination is nullptr,
               if (*CPU_destination == nullptr) {
                  // semantics is moving our pointer to that pointer
                  *CPU_destination = const_cast<T_non_const *> (MA::m_active_pointer);
               }
               // if the CPU destination is not nullptr
               else {
                  // semantics is copying from our pointer to the CPU_destination
                  std::copy_n(MA::m_active_pointer, elems, *CPU_destination);
               }
            }
         }
#endif
#endif /* defined(CARE_DEEP_COPY_RAW_PTR) */
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

      using MA::data;
      using MA::cdata;

      CARE_HOST T* data(ExecutionSpace space, bool moveToSpace = true) {
         return MA::data(chai::ExecutionSpace((int)space), moveToSpace);
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

