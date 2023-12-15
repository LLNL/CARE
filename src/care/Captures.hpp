#ifndef CARE_CAPTURES_HPP
#define CARE_CAPTURES_HPP

#include "care/FOREACHMACRO.h"
#include "care/local_ptr.h"
#include "care/mdspan.hpp"
#include "care/RestrictAccessor.hpp"
#include "chai/ManagedArray.hpp"

namespace care {
   template <class T>
   auto makeView(const chai::ManagedArray<T>& value) {
      // return care::mdspan(value.data(), value.size());
      // return care::mdspan(value.data(chai::ExecutionSpace::GPU), value.size());
      // return care::restrict_mdspan_1d<T>(value.data(chai::ExecutionSpace::GPU), value.size());
      return care::restrict_mdspan_1d<T>(value.data(), value.size());
      // return care::local_ptr(value.data());
      // return care::local_ptr(value.data(chai::ExecutionSpace::GPU));
   }

   template <class T>
   auto makeConstView(const chai::ManagedArray<T>& value) {
      // return care::mdspan(value.cdata(), value.size());
      // return care::mdspan(value.cdata(chai::ExecutionSpace::GPU), value.size());
      // return care::restrict_mdspan_1d<const T>(value.cdata(chai::ExecutionSpace::GPU), value.size());
      return care::restrict_mdspan_1d<const T>(value.cdata(), value.size());
      // return care::local_ptr(value.cdata());
      // return care::local_ptr(value.cdata(chai::ExecutionSpace::GPU));
   }
} // namespace care

#define CARE_CAPTURE_AS_VIEW_DETAIL(X) , X = care::makeView(X)
#define CARE_CAPTURE_AS_CONST_VIEW_DETAIL(X) , X = care::makeConstView(X)
#define CARE_CAPTURE_AS_ATOMIC_VIEW_DETAIL(X) , X = care::makeAtomicView(X)
#define CARE_GENERALIZED_CAPTURE_DETAIL(X) , X

#define CARE_CAPTURE_AS_VIEW(...) FOR_EACH(CARE_CAPTURE_AS_VIEW_DETAIL, __VA_ARGS__)
#define CARE_CAPTURE_AS_CONST_VIEW(...) FOR_EACH(CARE_CAPTURE_AS_CONST_VIEW_DETAIL, __VA_ARGS__)
#define CARE_CAPTURE_AS_ATOMIC_VIEW(...) FOR_EACH(CARE_CAPTURE_AS_ATOMIC_VIEW_DETAIL, __VA_ARGS__)
#define CARE_GENERALIZED_CAPTURES(...) FOR_EACH(CARE_GENERALIZED_CAPTURE_DETAIL, __VA_ARGS__)



#endif // CARE_CAPTURES_HPP
