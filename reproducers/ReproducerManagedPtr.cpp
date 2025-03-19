//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

//
// care::make_managed falls down when there are some c-style arrays and some
// care::host_device_ptrs as arguments to the same constructor and implicit
// casts are disabled.
//
// A potential solution would be to provide a chai::ManagedDataSplitter
// wrapper for arguments where the raw pointer needs to be extracted from
// host_device_ptrs. To avoid specifying the template parameter, a function
// called chai::make_managed_data_splitter could be used for template
// deduction.
//

// CARE library headers
#include "care/config.h"
#include "care/care.h"
#include "care/host_device_ptr.h"
#include "care/PointerTypes.h"

class BaseClass {
   public:
      CARE_HOST_DEVICE BaseClass() {}
      CARE_HOST_DEVICE virtual ~BaseClass() {}
      CARE_HOST_DEVICE virtual int getData(int index) = 0;
      CARE_HOST_DEVICE virtual void setData(int index, int value) = 0;
      CARE_HOST_DEVICE virtual int getManagedData(int index) = 0;
      CARE_HOST_DEVICE virtual void setManagedData(int index, int value) = 0;
};

class DerivedClass : public BaseClass {
   public:
      CARE_HOST_DEVICE DerivedClass(int* data, care::host_device_ptr<int> managedData)
      : BaseClass(), m_data(data), m_managedData(managedData) {}
      CARE_HOST_DEVICE virtual ~DerivedClass() {}
      CARE_HOST_DEVICE int getData(int index) override { return m_data[index]; }
      CARE_HOST_DEVICE void setData(int index, int value) override { m_data[index] = value; }
      CARE_HOST_DEVICE int getManagedData(int index) override { return m_managedData[index]; }
      CARE_HOST_DEVICE void setManagedData(int index, int value) override { m_managedData[index] = value; }

   private:
      int* m_data;
      care::host_device_ptr<int> m_managedData;
};


int main(int, char**) {
   // Set up data
   int length = 10;
   int data[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
   care::host_device_ptr<int> managedData(length);
   care::fill_n(managedData, length, 0);

   care::managed_ptr<BaseClass> base = care::make_managed<DerivedClass>(data, managedData);

   LOOP_SEQUENTIAL(i, 0, length) {
      base->setData(i, i);
   } LOOP_SEQUENTIAL_END

   LOOP_SEQUENTIAL(i, 0, length) {
      EXPECT_EQ(base->getData(i), i);
   } LOOP_SEQUENTIAL_END

   base.free();
   managedData.free();

   return 0;
}

