.. ##############################################################################
   # Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
   # project contributors. See the CARE LICENSE file for details.
   #
   # SPDX-License-Identifier: BSD-3-Clause
   ##############################################################################

==============
Infrastructure
==============

There are certain macros that are helpful for anyone wishing to write portable code. CARE provides some of these for convenience.

.. code-block:: c++

   #if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
   #define CARE_DEVICE_COMPILE
   #endif

   #if defined(__CUDACC__) || defined(__HIPCC__)
   #define CARE_GPUCC
   #endif

   #if defined(CARE_GPUCC)
   #define CARE_HOST_DEVICE __host__ __device__
   #define CARE_DEVICE __device__
   #define CARE_HOST __host__
   #define CARE_GLOBAL __global__
   #else // defined(CARE_GPUCC)
   #define CARE_HOST_DEVICE
   #define CARE_DEVICE
   #define CARE_HOST
   #define CARE_GLOBAL
   #endif // defined(CARE_GPUCC)
