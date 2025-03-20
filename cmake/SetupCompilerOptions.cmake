##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

if(ENABLE_CUDA)
   # nvcc dies if compiler flags are duplicated, and RAJA adds these flags
   set(RAJA_CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

   # Only add these flags if they are not already present or nvcc will die
   if(NOT CMAKE_CUDA_FLAGS MATCHES "--expt-extended-lambda")
      set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
   endif()

   set(CARE_CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
endif()
