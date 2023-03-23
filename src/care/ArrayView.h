#ifndef CARE_ARRAY_VIEW_H
#define CARE_ARRAY_VIEW_H

#include "care/ExecutionSpace.h"
#include "care/host_device_ptr.h"
#include "RAJA/util/View.hpp"

namespace care {
#if defined(CARE_GPUCC)
   constexpr care::ExecutionSpace DefaultExecutionSpace = care::GPU;

   using Layout2D = RAJA::Layout<2, int, 1>;
#else
   constexpr care::ExecutionSpace DefaultExecutionSpace = care::CPU;

   using Layout2D = RAJA::Layout<2, int, 0>;
#endif

   template <class T>
   using ArrayView1D = RAJA::View<T, RAJA::Layout<1>>;

   template <class T>
   using ArrayView2D = RAJA::View<T, Layout2D>;

   template <class T>
   ArrayView1D<T> makeArrayView1D(int extent,
                                  care::host_device_ptr<T> data);

   template <class T>
   ArrayView2D<T> makeArrayView2D(int extent1, int extent2,
                                  care::host_device_ptr<T> data);
   
   template <class T>
   ArrayView1D<T> makeArrayView1D(int extent,
                                  care::host_device_ptr<T> data) {
      return ArrayView1D(data.data(ExecutionSpace), extent);
   }

   template <class T>
   ArrayView2D<T> makeArrayView2D(int extent1, int extent2,
                                  care::host_device_ptr<T> data) {
#if defined(CARE_GPUCC)
      std::array< RAJA::idx_t, 2> perm {{1, 0}};
      RAJA::Layout<2> layout =
         RAJA::make_permuted_layout( {{extent1, extent2}}, perm );

      return ArrayView2D<T>(data.data(ExecutionSpace), layout);
#else
      return ArrayView2D<T>(data.data(ExecutionSpace), extent1, extent2);
#endif
   }
} // namespace care

#endif // CARE_ARRAY_VIEW_H

