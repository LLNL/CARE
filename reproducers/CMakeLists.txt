##############################################################################
# Copyright (c) 2020-24, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

if(CARE_ENABLE_MANAGED_PTR)
   blt_add_executable(NAME ReproducerManagedPtr
                      SOURCES ReproducerManagedPtr.cpp
                      DEPENDS_ON care)
endif()

blt_add_executable(NAME ReproducerArray
                   SOURCES ReproducerArray.cpp
                   DEPENDS_ON care)
