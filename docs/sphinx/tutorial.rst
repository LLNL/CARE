.. ##############################################################################
   # Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
   # project contributors. See the CARE LICENSE file for details.
   #
   # SPDX-License-Identifier: BSD-3-Clause
   ##############################################################################

========
Tutorial
========

CARE provides an abstraction layer built on top of CHAI and RAJA. Like RAJA, CARE has a set of execution policies, but these are simplified. These policies are the following: sequential, openmp, gpu, parallel, gpu_simulation, raja_chai_everywhere, raja_fusible, and raja_fusible_seq. One of the main difference between these policies and RAJA's policies is that the openmp and gpu policies fallback to the sequential policy if OpenMP or CUDA/HIP is not available. The parallel policy first attempts to use the GPU if available, then falls back to OpenMP, then falls back to the sequential policy. The main benefit of this approach is that clang-query can be used to detect what would be illegal memory accesses on the device using a regular CPU build.

It should also be noted that there is a gpu_simulation mode. It is generally much harder to debug issues on the device, so we have a mode that runs the code on the host, and if CHAI is configured with -DCHAI_ENABLE_GPU_SIMULATION_MODE=ON, the "GPU" memory space is actually on the host as well. This enables some stale data to be caught more easily when CHAI is used as a memory manager, and also enables address sanitizer or valgrind to be used on the simulated GPU memory space. CARE must also be configured with -DCARE_ENABLE_GPU_SIMULATION_MODE=ON.

Another policy of note is raja_chai_everywhere. This runs the code on both the host and the device and is intended to be used with managed_ptr to keep both the host and device object instance in sync.

Finally, there are fusible policies. Loops can be "fused" together into a single kernel to reduce kernel launch overhead. More details will be forthcoming on this.

These policies can be used with care::forall, which then forwards the arguments to RAJA::forall. CARE also provides convenient macros that can fall back to vanilla for loops if CARE is configured with -DENABLE_LEGACY_CARE=ON. These macros generally start with CARE and end with LOOP. For example, CARE_SEQUENTIAL_LOOP always executes sequentially. CARE_STREAM_LOOP executes on the GPU if configured with -DENABLE_CUDA=ON or -DENABLE_HIP=ON and nvcc or hip is used to compile the code, otherwise it falls back to executing using OpenMP on the host if configured with -DENABLE_OPENMP=ON and the correct OpenMP flags are passed to the compiler, otherwise it falls back to running sequentially on the host.

Some sample code is provided below to demonstrate the CARE concepts described above. Please note that the library must be configured with -DENABLE_CUDA=ON and/or -DENABLE_OPENMP=ON to run with CUDA or OpenMP.

.. code-block:: c++

   #include "care/care.h"

   int main(int argc, char* argv[]) {
      int length = 10;
      care::host_device_ptr<int> array(length, "array");

      // Will run on the GPU if this is compiled with nvcc or hip.
      // Otherwise will run sequentially on the host.
      CARE_GPU_LOOP(i, 0, length) {
         array[i] = i;
      } CARE_GPU_LOOP_END

      // Will run in parallel on the host if compiled with the OpenMP flags.
      // Otherwise will run sequentially on the host.
      CARE_OPENMP_LOOP(i, 0, length) {
         array[i] = array[i] * i;
      } CARE_OPENMP_LOOP_END

      // Will run on the GPU if compiled with nvcc or hip.
      // Otherwise, will run in parallel on the host if compiled with the OpenMP flags.
      // Otherwise, will run sequentially on the host.
      CARE_PARALLEL_LOOP(i, 0, length) {
         array[i] = array[i] * i;
      } CARE_PARALLEL_LOOP_END

      // Will always run sequentially on the host.
      CARE_SEQUENTIAL_LOOP(i, 0, length) {
         printf("array[%d]: %d", i, array[i]);
      } CARE_SEQUENTIAL_LOOP_END

      // The policy can be specified.
      CARE_LOOP(care::sequential{}, i, 0, length) {
         printf("array[%d]: %d", i, array[i]);
      } CARE_LOOP_END

      return 0;
   }
