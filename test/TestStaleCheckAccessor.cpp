//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#include "care/mdspan.hpp"
#include "care/StaleCheckAccessor.hpp"

#include "chai/ManagedArray.hpp"
#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

#include <string>

/////////////////////////////////////////////////////////////////////////
///
/// @brief Macro that allows extended __host__ __device__ lambdas (i.e.
///        CARE_STREAM_LOOP) to be used in google tests. Essentially,
///        extended __host__ __device__ lambdas cannot be used in
///        private or protected members, and the TEST macro creates a
///        protected member function. We get around this by creating a
///        function that the TEST macro then calls.
///
/// @note  Adapted from CHAI
///
/////////////////////////////////////////////////////////////////////////
#define GPU_TEST(X, Y) static void cuda_test_ ## X_ ## Y(); \
   TEST(X, gpu_test_##Y) { cuda_test_ ## X_ ## Y(); } \
   static void cuda_test_ ## X_ ## Y()

#define STR(X) #X

#if defined(__CUDACC__) || defined(__HIPCC__)
GPU_TEST(StaleDataChecker, NoStaleData)
{
   constexpr int size = 10;
   chai::ManagedArray<int> a(size);
   auto host_a = care::mdspan(a.data(chai::CPU), size);

   for (int i = 0; i < size; ++i) {
      host_a[i] = i;
   }
}

namespace care {
   template <class T>
   auto makeHostView(const chai::ManagedArray<T>& container,
                     const std::string& label) {
      auto data = container.data(chai::CPU);

      auto mapping = care::layout_right::template mapping<care::dextents<size_t, 1>>(care::dextents<size_t, 1>(container.size()));

      chai::ArrayManager* arrayManager = chai::ArrayManager::getInstance();
      chai::PointerRecord* record = arrayManager->getPointerRecord(data);
      care::StaleCheckAccessor<int> accessor(&record->m_touched[chai::ExecutionSpace::GPU], label);

      return care::mdspan(data, mapping, accessor);
   }

} // namespace care

#define CARE_HOST_VIEW(DATA) care::makeHostView(DATA, std::string(__FILE__) + std::string(":") + std::to_string(__LINE__));

GPU_TEST(StaleDataChecker, StaleData)
{
   constexpr int size = 10;
   chai::ManagedArray<int> a(size);
   auto host_a = CARE_HOST_VIEW(a);

   RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, size), [=] (int i) {
      host_a[i] = i;
   });

   RAJA::forall<RAJA::cuda_exec<256, true>>(RAJA::TypedRangeSegment<int>(0, size), [=] __device__ (int i) {
      a[i] = size - i;
   });

   for (int i = 0; i < size; ++i) {
      host_a[i] += i;
   }

   RAJA::forall<RAJA::seq_exec>(RAJA::TypedRangeSegment<int>(0, size), [=] (int i) {
      a[i] += i;
   });
}
#endif
