##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

include(CMakeDependentOption)

# Advanced configuration options
# TODO: If these are disabled, the code will not compile or run correctly.
#       Fix those issues so that these options are actually configurable.
option(ENABLE_PINNED "Enable pinned memory space" ON)
option(CARE_ENABLE_PINNED_MEMORY_FOR_SCANS "Use pinned memory for scan lengths" ON)
option(CARE_GPU_MEMORY_IS_ACCESSIBLE_ON_CPU "Allows default memory spaces for ZERO_COPY and PAGEABLE to be the GPU memory space" OFF)
option(CARE_LEGACY_COMPATIBILITY_MODE "Enable legacy compatibility mode" OFF)
option(CARE_DEEP_COPY_RAW_PTR "Use deep copy for managed array initialization from raw pointer" OFF)
option(CARE_ENABLE_MANAGED_PTR "Enable managed_ptr aliases, tests, and reproducer" ON)
option(CARE_DISABLE_RAJAPLUGIN "Disable use of the RAJA plugin. WILL ALSO DISABLE MEMORY MOTION." OFF)
option(CARE_ENABLE_EXTERN_INSTANTIATE "Enable extern instantiation of template functions" OFF)
# Bounds checking only checks for negative indices
# if the resource manager is disabled (CHAI_DISABLE_RM defined).
option(CARE_ENABLE_BOUNDS_CHECKING "Enable bounds checking for host_device_ptr on the host" OFF)
option(CARE_ENABLE_GPU_SIMULATION_MODE "Enable GPU simulation mode." OFF )
option(CARE_NEVER_USE_RAJA_PARALLEL_SCAN "Disable RAJA parallel scans in SCAN loops." OFF )
#  Known to not compile for Debug configurations, have low probability of working in 
#  practice for Release / RelWithDebInfo configurations.
option(CARE_ENABLE_FUSER_BIN_32 "Enable the 32 register fusible loop bin." OFF)
option(CARE_ENABLE_PARALLEL_LOOP_BACKWARDS "Reverse the start and end for parallel loops." OFF)
option(CARE_ENABLE_STALE_DATA_CHECK "Enable checking for stale host data. Only applicable for GPU (or GPU simulation) builds." OFF)

# Extra components
cmake_dependent_option(CARE_ENABLE_TESTS "Build CARE tests"
                       ON "ENABLE_TESTS" OFF)

cmake_dependent_option(CARE_ENABLE_BENCHMARKS "Build CARE benchmarks"
                       ON "ENABLE_BENCHMARKS" OFF)

cmake_dependent_option(CARE_ENABLE_DOCS "Build CARE documentation"
                       ON "ENABLE_DOCS" OFF)

cmake_dependent_option(CARE_ENABLE_EXAMPLES "Build CARE examples"
                       ON "ENABLE_EXAMPLES" OFF)

option(CARE_ENABLE_REPRODUCERS "Build CARE reproducers" OFF)

# Extra submodule components
cmake_dependent_option(CARE_ENABLE_SUBMODULE_TESTS "Build submodule tests"
                       OFF "ENABLE_TESTS" OFF)

cmake_dependent_option(CARE_ENABLE_SUBMODULE_BENCHMARKS "Build submodule benchmarks"
                       OFF "ENABLE_BENCHMARKS" OFF)

cmake_dependent_option(CARE_ENABLE_SUBMODULE_DOCS "Build submodule documentation"
                       OFF "ENABLE_DOCS" OFF)

cmake_dependent_option(CARE_ENABLE_SUBMODULE_EXAMPLES "Build submodule examples"
                       OFF "ENABLE_EXAMPLES" OFF)

option(CARE_ENABLE_SUBMODULE_TOOLS "Build submodule tools" OFF)
option(CARE_ENABLE_SUBMODULE_EXERCISES "Build submodule exercises" OFF)
option(CARE_ENABLE_SUBMODULE_REPRODUCERS "Build submodule reproducers" OFF)

# CUDA specific options
if (ENABLE_CUDA)
   # CUDA configuration
   option(CUDA_SEPARABLE_COMPILATION "Enable CUDA separable compilation" ON)

   # Optional features
   set(CARE_ENABLE_LOOP_FUSER ON CACHE STRING "Enable the loop fuser")

   # This is needed for the loop fuser to work
   option(CHAI_ENABLE_RAJA_PLUGIN "Build plugin to set RAJA execution spaces" ON)
else()
   set(CARE_ENABLE_LOOP_FUSER OFF CACHE STRING "Enable the loop fuser")
endif()

set(CARE_LOOP_FUSER_FLUSH_LENGTH "838608" CACHE STRING "Loop fuser flush length")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
   set(CMAKE_CUDA_ARCHITECTURES 75)
endif()
