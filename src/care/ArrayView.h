//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef CARE_ARRAY_VIEW_H
#define CARE_ARRAY_VIEW_H

#include "care/host_device_ptr.h"
#include "chai/ExecutionSpaces.hpp"
#include "RAJA/util/View.hpp"

namespace care {
   template <class T>
   using ArrayView1D = RAJA::View<T, RAJA::Layout<1>>;

   template <class T>
   using ArrayCView1D = ArrayView1D<const T>;

#if defined(CARE_GPUCC)
   template <class T>
   using ArrayView2D = RAJA::View<T, RAJA::Layout<2, int, 1>>;
#else
   template <class T>
   using ArrayView2D = RAJA::View<T, RAJA::Layout<2, int, 0>>;
#endif

   template <class T>
   using ArrayCView2D = ArrayView2D<const T>;

   ///
   /// Creates a 1D view on top of a flat allocation (non-owning semantics)
   ///
   /// Moves the data to the default execution space (determined at
   /// compile time) and wraps it in a RAJA::View.
   ///
   /// @param[in] data Container for the allocation
   /// @param[in] extent Length of the view
   ///
   /// @return A RAJA::View object
   ///
   template <class T>
   auto makeArrayView1D(care::host_device_ptr<T> data,
                        int extent);

   ///
   /// Creates a 1D const view on top of a flat allocation (non-owning semantics)
   ///
   /// Moves the data to the default execution space (determined at
   /// compile time) and wraps it in a RAJA::View.
   ///
   /// @param[in] data Container for the allocation
   /// @param[in] extent Length of the view
   ///
   /// @return A RAJA::View object
   ///
   template <class T>
   auto makeArrayCView1D(care::host_device_ptr<T> data,
                         int extent);

   ///
   /// Creates a 2D view on top of a flat allocation (non-owning semantics)
   ///
   /// Moves the data to the default execution space (determined at
   /// compile time) and wraps it in a RAJA::View. When using with a
   /// RAJA loop, extent2 should correspond to the length of the RAJA
   /// loop for optimal performance.
   ///
   /// @param[in] data Container for the allocation
   /// @param[in] extent1 Length of the 1st dimension
   /// @param[in] extent2 Length of the 2nd dimension
   ///
   /// @return A RAJA::View object
   ///
   template <class T>
   auto makeArrayView2D(care::host_device_ptr<T> data,
                        int extent1, int extent2);

   ///
   /// Creates a 2D const view on top of a flat allocation (non-owning semantics)
   ///
   /// Moves the data to the default execution space (determined at
   /// compile time) and wraps it in a RAJA::View. When using with a
   /// RAJA loop, extent2 should correspond to the length of the RAJA
   /// loop for optimal performance.
   ///
   /// @param[in] data Container for the allocation
   /// @param[in] extent1 Length of the 1st dimension
   /// @param[in] extent2 Length of the 2nd dimension
   ///
   /// @return A RAJA::View object
   ///
   template <class T>
   auto makeArrayCView2D(care::host_device_ptr<T> data,
                         int extent1, int extent2);
   
   template <class T>
   auto makeArrayView1D(care::host_device_ptr<T> data,
                        int extent) {
#if defined(CARE_GPUCC)
      return RAJA::View<T, RAJA::Layout<1>>(data.data(chai::GPU), extent);
#else
      return RAJA::View<T, RAJA::Layout<1>>(data.data(chai::CPU), extent);
#endif
   }

   template <class T>
   auto makeArrayCView1D(care::host_device_ptr<T> data,
                         int extent) {
      return makeArrayView1D(care::host_device_ptr<T const>(data), extent);
   }

   template <class T>
   auto makeArrayView2D(care::host_device_ptr<T> data,
                        int extent1, int extent2) {
#if defined(CARE_GPUCC)
      return RAJA::View<T, RAJA::Layout<2, int, 1>>(data.data(chai::GPU), extent1, extent2);
#else
      std::array< RAJA::idx_t, 2> perm {{1, 0}};
      RAJA::Layout<2> layout =
         RAJA::make_permuted_layout( {{extent1, extent2}}, perm );

      return RAJA::View<T, RAJA::Layout<2, int, 0>>(data.data(chai::CPU), layout);
#endif
   }

   template <class T>
   auto makeArrayCView2D(care::host_device_ptr<T> data,
                         int extent1, int extent2) {
      return makeArrayView2D(care::host_device_ptr<T const>(data), extent1, extent2);
   }
} // namespace care

#endif // CARE_ARRAY_VIEW_H

