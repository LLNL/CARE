//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#include "care/config.h"

// other library headers
#include "gtest/gtest.h"

// care headers
#include "care/algorithm.h"
#include "care/host_device_ptr.h"
#include "care/PointerTypes.h"
#include "care/detail/test_utils.h"

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Custom base class used to verify managed_ptr and make_managed
///        behave correctly.
///
/////////////////////////////////////////////////////////////////////////
class BaseClass {
   public:
      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Constructor
      ///
      /////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE BaseClass() {}

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Destructor
      ///
      /////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE virtual ~BaseClass() {}

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Data accessor.
      ///
      /// @param[in] index   The index into the contained data
      ///
      /// @return   The value of the data at the given index
      ///
      /////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE virtual int getData(int index) = 0;

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Data setter.
      ///
      /// @param[in] index   The index into the contained data
      /// @param[in] value   The new value of the data at the given index
      ///
      /////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE virtual void setData(int index, int value) = 0;
};

/////////////////////////////////////////////////////////////////////////
///
/// @author Alan Dayton
///
/// @brief Custom derived class used to verify managed_ptr and make_managed
///        behave correctly.
///
/////////////////////////////////////////////////////////////////////////
class DerivedClass : public BaseClass {
   public:
      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Constructor
      ///
      /// @param[in] data    The data to store
      ///
      /////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE DerivedClass(int* data) : BaseClass(), m_data(data) {}

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Destructor
      ///
      /////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE virtual ~DerivedClass() {}

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Data accessor.
      ///
      /// @param[in] index   The index into the contained data
      ///
      /// @return   The value of the data at the given index
      ///
      /////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE int getData(int index) override { return m_data[index]; }

      /////////////////////////////////////////////////////////////////////////
      ///
      /// @author Alan Dayton
      ///
      /// @brief Data setter.
      ///
      /// @param[in] index   The index into the contained data
      /// @param[in] value   The new value of the data at the given index
      ///
      /////////////////////////////////////////////////////////////////////////
      CARE_HOST_DEVICE void setData(int index, int value) override { m_data[index] = value; }

   private:
      int* m_data; //!< The data to store
};

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that initializes GPU memory pools. 
///
/////////////////////////////////////////////////////////////////////////
#if defined(CARE_GPUCC)
GPU_TEST(ManagedPtr, gpu_initialization) {
   printf("Initializing\n");
   init_care_for_testing();
   printf("Initialized... Testing care::managed_ptr\n");
}
#endif

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that splitting a host_device_ptr through
///        the call to make_managed behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
TEST(ManagedPtr, SplitHostDevicePointer)
{
   // Set up data
   int length = 10;
   care::host_device_ptr<int> data(length);
   care::fill_n(data, length, 0);

   // This will construct an instance of DerivedClass on the host and an instance of
   // DerivedClass on the device. It is aware of host_device_ptr types, so it gives
   // the host pointer in data to the host instance and the device pointer in data to
   // the device instance.
   care::managed_ptr<BaseClass> base = care::make_managed<DerivedClass>(chai::unpack(data));

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

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      base->setData(i, i);
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(base->getData(i), i);
   } CARE_SEQUENTIAL_LOOP_END

   base.free();
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief Test case that checks that passing a c-style array to
///        make_managed behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
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

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      base->setData(i, i);
   } CARE_SEQUENTIAL_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(base->getData(i), i);
   } CARE_SEQUENTIAL_LOOP_END

   base.free();
}


#if defined(CARE_GPUCC)


/////////////////////////////////////////////////////////////////////////
///
/// @brief GPU test case that checks that splitting a host_device_ptr
///        through the call to make_managed behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
GPU_TEST(ManagedPtr, SplitHostDevicePointer)
{
   // Set up data
   int length = 10;
   care::host_device_ptr<int> data(length);
   care::fill_n(data, length, 0);

   // This will construct an instance of DerivedClass on the host and an instance of
   // DerivedClass on the device. It is aware of host_device_ptr types, so it gives
   // the host pointer in data to the host instance and the device pointer in data to
   // the device instance.
   care::managed_ptr<BaseClass> base = care::make_managed<DerivedClass>(chai::unpack(data));

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

   CARE_MANAGED_PTR_LOOP(i, 0, length) {
      base->setData(i, i);
   } CARE_MANAGED_PTR_LOOP_END

   CARE_SEQUENTIAL_LOOP(i, 0, length) {
      EXPECT_EQ(base->getData(i), i);
   } CARE_SEQUENTIAL_LOOP_END

   base.free();
}

/////////////////////////////////////////////////////////////////////////
///
/// @brief GPU test case that checks that passing a c-style array to
///        make_managed behaves correctly.
///
/////////////////////////////////////////////////////////////////////////
GPU_TEST(ManagedPtr, RawPointer)
{
   // Set up data
   int length = 10;
   int* data;

#ifdef __CUDACC__
   cudaMalloc(&data, length * sizeof(int));
   cudaMemset(data, 0, length * sizeof(int));
#else
   hipMalloc(&data, length * sizeof(int));
   hipMemset(data, 0, length * sizeof(int));
#endif

   // This will construct an instance of DerivedClass on the host and an instance
   // of DerivedClass on the device. This gives the device pointer to both the host
   // instance and the device instance. Therefore, it is not safe to use on the host.
   // This could be a common pattern in some codes (i.e. a specific value that is only
   // ever accessed on the device, enforced by __device__ only accessors), and to avoid
   // forcing users to make separate constructors for the host and device, we need to
   // ensure that this works.
   care::managed_ptr<BaseClass> base = care::make_managed<DerivedClass>(data);

   CARE_MANAGED_PTR_LOOP(i, 0, length) {
      base->setData(i, i);
   } CARE_MANAGED_PTR_LOOP_END

   RAJAReduceMin<bool> passed{true};

   CARE_MANAGED_PTR_LOOP(i, 0, length) {
      if (base->getData(i) != i) {
         passed.min(false);
      }
   } CARE_MANAGED_PTR_LOOP_END

   EXPECT_TRUE((bool) passed);

   base.free();
}

#endif // CARE_GPUCC
