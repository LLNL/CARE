#ifndef CARE_ARRAY_VIEW_H
#define CARE_ARRAY_VIEW_H

#include "care/host_device_ptr.h"
#include "chai/ExecutionSpaces.hpp"
#include "RAJA/util/View.hpp"

namespace care {
#if defined(CARE_GPUCC)
   constexpr chai::ExecutionSpace DefaultExecutionSpace = chai::GPU;

   using Layout2D = RAJA::Layout<2, int, 1>;
#else
   constexpr chai::ExecutionSpace DefaultExecutionSpace = chai::CPU;

   using Layout2D = RAJA::Layout<2, int, 0>;
#endif

   template <class T>
   using ArrayView1D = RAJA::View<T, RAJA::Layout<1>>;

   template <class T>
   using ArrayView2D = RAJA::View<T, Layout2D>;

   template <class T>
   ArrayView1D<T> makeArrayView1D(care::host_device_ptr<T> data,
                                  int extent);

   template <class T>
   ArrayView2D<T> makeArrayView2D(care::host_device_ptr<T> data,
                                  int extent1, int extent2);
   
   template <class T>
   ArrayView1D<T> makeArrayView1D(care::host_device_ptr<T> data,
                                  int extent) {
      return ArrayView1D<T>(data.data(DefaultExecutionSpace), extent);
   }

   template <class T>
   ArrayView2D<T> makeArrayView2D(care::host_device_ptr<T> data,
                                  int extent1, int extent2) {
#if defined(CARE_GPUCC)
      std::array< RAJA::idx_t, 2> perm {{1, 0}};
      RAJA::Layout<2> layout =
         RAJA::make_permuted_layout( {{extent1, extent2}}, perm );

      return ArrayView2D<T>(data.data(DefaultExecutionSpace), layout);
#else
      return ArrayView2D<T>(data.data(DefaultExecutionSpace), extent1, extent2);
#endif
   }
} // namespace care

#endif // CARE_ARRAY_VIEW_H

