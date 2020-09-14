.. ######################################################################################
   # Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
   # See the top-level LICENSE file for details.
   #
   # SPDX-License-Identifier: BSD-3-Clause
   ######################################################################################

==============================
CARE: CHAI and RAJA Extensions
==============================

Data Structures
===============

CARE provides a helpful wrapper for chai::ManagedArray and also adds other data structures for convenience and to help safeguard against accessing memory in the wrong execution space. These data structures include care::host_device_ptr, care::host_ptr, care::device_ptr, care::local_ptr, care::array, care::KeyValueSorter, care::LocalKeyValueSorter, and care::managed_ptr (a type alias for chai::managed_ptr). Each of these is discussed in more detail below.

care::host_device_ptr
---------------------

The most commonly used data structure in CARE is care::host_device_ptr. It is a lightweight wrapper for chai::ManagedArray. As such, it provides the usual data movement semantics when the execution space is set and when the copy constructor is called, which happens when it is captured into a lambda used with RAJA. The loop macros provided by CARE take care of setting the correct execution space and creating the lambda behind the scenes.

What this wrapper adds is a compiler error if the data in care::host_device_ptr is accessed outside of a RAJA loop. This prevents the possibility of reading/writing stale data. When used in combination with the CARE loop macros, clang queries can also be constructed that find any instance of care::host_device_ptr used outside of those loops.

Another helpful feature of this wrapper is the ability to name these pointers. This is highly useful for debugging and can be used to great effect in combination with CHAI callbacks. For example, if care::set_log_chai_leaks is called, all the leaks in a certain memory space or all of the memory spaces will be written to stdout with the name supplied to the care::host_device_ptr. Similar functions enable logging of memory transfers and even allocations and deallocations.

Some sample code is provided below for proper use of care::host_device_ptr.

.. code-block:: c++

   #define GPU_ACTIVE // If this is not defined in the compilation unit, CARE loop macros will not run on the device

   template <typename T>
   void allocatePtr(int length, care::host_device_ptr<T>& ptr, std::string name) {
      ptr = care::host_device_ptr<T>(length, name.c_str());
   }

   #define ALLOCATE_PTR(LENGTH, PTR) allocatePtr(LENGTH, PTR, __FILE__ ":" std::to_string(__LINE__) ":" #PTR);

   #include "care/care.h"

   int main(int argc, char* argv[]) {
      // Set up logging for leaks. The integer argument corresponds to
      // chai::ExecutionSpace (NUM_SPACES means all in this context)
      chai::set_log_chai_leaks(4);
      chai::set_callback_output_file("leaks.txt");

      // Initialize data structure
      int length = 10;
      care::host_device_ptr<int> myArray;
      ALLOCATE_PTR(length, myArray);

      // Will run on the GPU if this is compiled with nvcc or hip.
      // Otherwise will run sequentially on the host.
      CARE_GPU_LOOP(i, 0, length) {
         myArray[i] = i;
      } CARE_GPU_LOOP_END

      // With chai::ManagedArray this would happily access stale host data which is
      // an extremely difficult bug to find, but with care::host_device_ptr we get a
      // compile-time error.
      // myArray[0] = 4;

      // Will always run sequentially on the host.
      CARE_SEQUENTIAL_LOOP(i, 0, length) {
         printf("myArray[%d]: %d", i, myArray[i]);
      } CARE_SEQUENTIAL_LOOP_END

      // Free memory
      myArray.free();

      return 0;
   }

care::host_ptr
--------------

This type is a lightweight wrapper for a raw host array that prevents the host memory from erroneously being accessed on the device. For obvious reasons, the data it contains cannot be accessed in STREAM or WORK loops, but it can be accessed in plain host code outside of the loop macros and within SEQUENTIAL loops.

Some sample code is provided below for proper use of care::host_ptr.

.. code-block:: c++

   #define GPU_ACTIVE // If this is not defined in the compilation unit, CARE loop macros will not run on the device

   template <typename T>
   void allocatePtr(int length, care::host_device_ptr<T>& ptr, std::string name) {
      ptr = care::host_device_ptr<T>(length, name.c_str());
   }

   #define ALLOCATE_PTR(LENGTH, PTR) allocatePtr(LENGTH, PTR, __FILE__ ":" std::to_string(__LINE__) ":" #PTR);

   #include "care/care.h"

   int main(int argc, char* argv[]) {
      // wrap a raw array
      int length = 100;
      int* array1 = new int[length];
      care::host_ptr<int> host_array1 = array1.data();
      host_array1[5] = 5;
      delete[] array1;

      // move data to host
      int length = 100;
      care::host_device_ptr<int> array2;
      ALLOCATE_PTR(length, array2);
      care::host_ptr<int> host_array2 = array2;
      host_array2[5] = 5; // If only reading/writing one element, using pick/set directly on array2 would be a better choice. But if reading/writing multiple elements, this pattern is preferred since it does fewer memory transfers.
      array2.free();

      return 0;
   }

care::device_ptr
----------------

This type is a lightweight wrapper for a raw device array that prevents the device memory from erroneously being accessed on the host. For obvious reasons, the data it contains cannot be accessed in SEQUENTIAL loops, but it can be accessed in plain device code outside of the loop macros and within STREAM or WORK loops.

Some sample code is provided below for proper use of care::device_ptr.

.. code-block:: c++

   #define GPU_ACTIVE // If this is not defined in the compilation unit, CARE loop macros will not run on the device

   template <typename T>
   void allocatePtr(int length, care::host_device_ptr<T>& ptr, std::string name) {
      ptr = care::host_device_ptr<T>(length, name.c_str());
   }

   #define ALLOCATE_PTR(LENGTH, PTR) allocatePtr(LENGTH, PTR, __FILE__ ":" std::to_string(__LINE__) ":" #PTR);

   #include "care/care.h"

   int main(int argc, char* argv[]) {
      // wrap a raw array
      int length = 100;
      int* array1;
      cudaMalloc(&((void*) array1), length * sizeof(int));
      care::device_ptr<int> device_array1 = array1;

      CARE_GPU_KERNEL {
         device_array1[5] = 5;
      } CARE_GPU_KERNEL_END

      cudaFree(array1);

      // move data to device (i
      int length = 100;
      care::host_device_ptr<int> array2;
      ALLOCATE_PTR(length, array2);
      care::device_ptr<int> device_array2 = array2.data(chai::GPU);

      // In this case, it is completely unnecessary to extract the device pointer since
      // care::host_device_ptr could be used directly, but this gives the general idea.
      CARE_GPU_KERNEL {
         device_array2[5] = 5;
      } CARE_GPU_KERNEL_END

      array2.free();

      return 0;
   }
