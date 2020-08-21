//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#define GPU_ACTIVE

#include "care/config.h"

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/care.h"
#include "care/host_device_ptr.h"
#include "care/KeyValueSorter.h"
#include "care/PointerTypes.h"

class BaseClass {
   public:
      CARE_HOST_DEVICE BaseClass() {}
      CARE_HOST_DEVICE virtual ~BaseClass() {}
      CARE_HOST_DEVICE virtual int getData(int index) = 0;
      CARE_HOST_DEVICE virtual void setData(int index, int value) = 0;
};

class DerivedClass : public BaseClass {
   public:
      CARE_HOST_DEVICE DerivedClass(int* data) : BaseClass(), m_data(data) {}
      CARE_HOST_DEVICE virtual ~DerivedClass() {}
      CARE_HOST_DEVICE int getData(int index) override { return m_data[index]; }
      CARE_HOST_DEVICE void setData(int index, int value) override { m_data[index] = value; }

   private:
      int* m_data;
};

TEST(ManagedPtr, SplitHostDevicePointer)
{
   // Set up data
   int length = 10;
   care::host_device_ptr<int> data(length);
   care_utils::ArrayFill<int>(data, length, 0);

   // This will construct an instance of DerivedClass on the host and an instance of
   // DerivedClass on the device. It is aware of host_device_ptr types, so it gives
   // the host pointer in data to the host instance and the device pointer in data to
   // the device instance.
   care::managed_ptr<BaseClass> base = care::make_managed<DerivedClass>(data);

   // Now if data is changed on the host or the device and we want the changes to be
   // reflected in the other execution space, we need to set a callback that triggers
   // the copy constructor of data. We can also use this callback to free the
   // host_device_ptr.
   base.set_callback([=] (chai::Action action,
                          chai::ExecutionSpace space,
                          void* /* pointer */) mutable {
      if (action == chai::ACTION_MOVE) {
         auto dataTemp = data; // Trigger move of data in host_device_ptr
         (void) dataTemp; // Quiet compiler
         return true;
      }
      else if (action == chai::ACTION_FREE && space == chai::NONE) {
         data.free(); // Free data in host_device_ptr
         return true;
      }
      else {
         return false; // Let the default actions be taken
      }
   });

   LOOP_SEQUENTIAL(i, 0, length) {
      base->setData(i, i);
   } LOOP_SEQUENTIAL_END

   LOOP_SEQUENTIAL(i, 0, length) {
      EXPECT_EQ(base->getData(i), i);
   } LOOP_SEQUENTIAL_END

   base.free();
}

TEST(ManagedPtr, RawPointer)
{
   // Set up data
   int length = 10;
   int data[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

   // This will construct an instance of DerivedClass on the host and an instance of
   // DerivedClass on the device. This gives the host pointer to both the host instance
   // and the device instance. Therefore, it is not safe to use on the device. This
   // is a common pattern in some codes (i.e. a c-style string that is only ever
   // accessed on the host, enforced by __host__ only accessors), and to avoid forcing
   // users to make separate constructors for the host and device, we need to ensure
   // that this works.
   care::managed_ptr<BaseClass> base = care::make_managed<DerivedClass>(data);

   LOOP_SEQUENTIAL(i, 0, length) {
      base->setData(i, i);
   } LOOP_SEQUENTIAL_END

   LOOP_SEQUENTIAL(i, 0, length) {
      EXPECT_EQ(base->getData(i), i);
   } LOOP_SEQUENTIAL_END

   base.free();
}


#if defined(__GPUCC__)

// Adapted from CHAI
#define GPU_TEST(X, Y) \
   static void gpu_test_##X##Y(); \
   TEST(X, gpu_test_##Y) { gpu_test_##X##Y(); } \
   static void gpu_test_##X##Y()


#endif // __GPUCC__

