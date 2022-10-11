#ifndef CARE_ARRAY_2_H
#define CARE_ARRAY_2_H

namespace care {
   template <class T, class SizeType = std::size_t>
   class Array2 {
      public:
         Array2() {
            // 1. Could do nothing
         }

         // Only if not a view
         Array2(SizeType count) {
            // 1. Could just allocate space
            // 2. Could allocate space and initialize to a default value
         }

         // Only if not a view
         Array2(SizeType count, T initial) {

         }

         Array2(Array2 const & other) {
            // 1. Could be a deep copy
            // 2. Could be a shallow copy
            // 3. Could have CHAI semantics
            // 4. Could be disabled
         }

         Array2(Array2 && other) {
            // 1. Could steal resources
            // 2. Could do something like CHAI?
         }

         // This is always needed
         ~Array2() {
            // 1. Could clean up memory
            // 2. Could do nothing

         }

         SizeType size() {
            // 1. Could return the number of elements
            // 2. Could return the size in bytes
         }

         // Only if not a view
         void resize(SizeType count) {

         }

         // In the case of CHAI, could have allocate, reallocate, and free methods
   };
} // namespace care

#endif // CARE_ARRAY_2_H
