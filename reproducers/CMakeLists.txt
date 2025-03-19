##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

set(care_depends
    care)

if (ENABLE_CUDA)
   list(APPEND care_depends cuda)
endif()

if (ENABLE_HIP)
   list(APPEND care_depends blt::hip)
endif ()

if (ENABLE_OPENMP)
   list(APPEND care_depends openmp)
endif()

if (CARE_ENABLE_MANAGED_PTR)
   blt_add_executable(NAME ReproducerManagedPtr
                      SOURCES ReproducerManagedPtr.cpp
                      DEPENDS_ON ${care_depends})

   target_include_directories(ReproducerManagedPtr PRIVATE ${PROJECT_SOURCE_DIR}/src)
   target_include_directories(ReproducerManagedPtr PRIVATE ${PROJECT_BINARY_DIR}/include)
endif ()
