#ifndef CARE_CUDA_UMPIRE_RESOURCE_H
#define CARE_CUDA_UMPIRE_RESOURCE_H

#include "camp/defines.hpp"

#ifdef CAMP_ENABLE_CUDA

#include <mutex> // camp/resource/cuda.hpp is missing this include

#include "camp/resource/cuda.hpp"

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"

#include <cuda_runtime.h>

namespace care {
  class CudaUmpireResource : public camp::resources::Cuda {
    public:
      CudaUmpireResource() :
         m_resourceManager{&umpire::ResourceManager::getInstance()}
      {
         m_deviceAllocator  = m_resourceManager->getAllocator("DEVICE");
         m_pinnedAllocator  = m_resourceManager->getAllocator("PINNED");
         m_managedAllocator = m_resourceManager->getAllocator("UM");
      }

      CudaUmpireResource(const umpire::Allocator& deviceAllocator,
                         const umpire::Allocator& pinnedAllocator,
                         const umpire::Allocator& managedAllocator) :
         m_resourceManager{&umpire::ResourceManager::getInstance()},
         m_deviceAllocator{deviceAllocator},
         m_pinnedAllocator{pinnedAllocator},
         m_managedAllocator{managedAllocator}
      {
      }

      // Memory
      template <typename T>
      T *allocate(size_t size, camp::resources::MemoryAccess ma = camp::resources::MemoryAccess::Device) {
        T *ret = nullptr;

        if (size > 0) {
          auto d{camp::resources::device_guard(get_device())};

          switch (ma) {
            case camp::resources::MemoryAccess::Unknown:
            case camp::resources::MemoryAccess::Device:
              ret = static_cast<T*>(m_deviceAllocator.allocate(sizeof(T) * size));
              break;
            case camp::resources::MemoryAccess::Pinned:
              // TODO: do a test here for whether managed is *actually* shared
              // so we can use the better performing memory
              ret = static_cast<T*>(m_pinnedAllocator.allocate(sizeof(T) * size));
              break;
            case camp::resources::MemoryAccess::Managed:
              ret = static_cast<T*>(m_managedAllocator.allocate(sizeof(T) * size));
              break;
          }
        }

        return ret;
      }

      void *calloc(size_t size, camp::resources::MemoryAccess ma = camp::resources::MemoryAccess::Device) {
        void *p = allocate<char>(size, ma);
        this->memset(p, 0, size);
        return p;
      }

      void deallocate(void *p, camp::resources::MemoryAccess ma = camp::resources::MemoryAccess::Unknown) {
        auto d{camp::resources::device_guard(get_device())};

        if (ma == camp::resources::MemoryAccess::Unknown) {
          ma = get_access_type(p);
        }

        switch (ma) {
          case camp::resources::MemoryAccess::Device:
            m_deviceAllocator.deallocate(p);
            break;
          case camp::resources::MemoryAccess::Pinned:
            // TODO: do a test here for whether managed is *actually* shared
            // so we can use the better performing memory
            m_pinnedAllocator.deallocate(p);
            break;
          case camp::resources::MemoryAccess::Managed:
            m_managedAllocator.deallocate(p);
            break;
          case camp::resources::MemoryAccess::Unknown:
            ::camp::throw_re("Unknown memory access type, cannot free");
        }
      }

      void memcpy(void *dst, const void *src, size_t size) {
        if (size > 0) {
          auto d{camp::resources::device_guard(get_device())};
          m_resourceManager->copy(dst, src, *this, size);
        }
      }

      void memset(void *p, int val, size_t size)
      {
        if (size > 0) {
          auto d{camp::resources::device_guard(get_device())};
          m_resourceManager->memset(p, val, *this, size);
        }
      }

    private:
      // TODO: Make this a public or protected method in camp
      camp::resources::MemoryAccess get_access_type(void *p) {
        cudaPointerAttributes a;
        cudaError_t status = cudaPointerGetAttributes(&a, p);
        if (status == cudaSuccess) {
          switch(a.type){
            case cudaMemoryTypeUnregistered:
              return camp::resources::MemoryAccess::Unknown;
            case cudaMemoryTypeHost:
              return camp::resources::MemoryAccess::Pinned;
            case cudaMemoryTypeDevice:
              return camp::resources::MemoryAccess::Device;
            case cudaMemoryTypeManaged:
              return camp::resources::MemoryAccess::Managed;
          }
        }
        ::camp::throw_re("invalid pointer detected");
        // This return statement exists because compilers do not determine the
        // above unconditionally throws
        // related: https://stackoverflow.com/questions/64523302/cuda-missing-return-statement-at-end-of-non-void-function-in-constexpr-if-fun
        return camp::resources::MemoryAccess::Unknown;
      }

      umpire::ResourceManager* m_resourceManager;

      umpire::Allocator m_deviceAllocator;
      umpire::Allocator m_pinnedAllocator;
      umpire::Allocator m_managedAllocator;
  }; // class CudaUmpireResource
} // namespace care

#endif // CAMP_ENABLE_CUDA

#endif // CARE_CUDA_UMPIRE_RESOURCE_H
