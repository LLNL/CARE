######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

################################
# BASIL
################################
if (BASIL_DIR)
    include(cmake/libraries/FindBasil.cmake)
    if (BASIL_FOUND)
        set(BASIL_DEPENDS )
        blt_list_append(TO BASIL_DEPENDS ELEMENTS cuda IF ${ENABLE_CUDA})
        blt_register_library( NAME       basil
                                TREAT_INCLUDES_AS_SYSTEM ON
                                INCLUDES   ${BASIL_INCLUDE_DIR}
                                LIBRARIES  ${BASIL_LIBRARY}
                                DEPENDS_ON ${BASIL_DEPENDS})

        set(CARE_HAVE_BASIL "1" CACHE STRING "")
    else()
        message(FATAL_ERROR "Unable to find BASIL with given path: ${BASIL_DIR}")
    endif()
else()
    message(STATUS "Library Disabled: BASIL")
    set(CARE_HAVE_BASIL "0" CACHE STRING "")
endif()

################################
# CUB
################################
if (CUB_DIR)
    include(cmake/libraries/FindCUB.cmake)
    if (CUB_FOUND)
        blt_register_library( NAME       cub
                                TREAT_INCLUDES_AS_SYSTEM ON
                                INCLUDES   ${CUB_INCLUDE_DIRS})

        set(CARE_HAVE_CUB "1" CACHE STRING "")
    else()
        message(FATAL_ERROR "Unable to find CUB with given path: ${CUB_DIR}")
    endif()
else()
    message(STATUS "Library Disabled: CUB")
    set(CARE_HAVE_CUB "0" CACHE STRING "")
endif()

################################
# NVTOOLSEXT
################################
if (NVTOOLSEXT_DIR)
    if (ENABLE_CUDA)
       include(cmake/libraries/FindNvToolsExt.cmake)
       if (NVTOOLSEXT_FOUND)
           blt_register_library( NAME       nvtoolsext
                                   TREAT_INCLUDES_AS_SYSTEM ON
                                   INCLUDES   ${NVTOOLSEXT_INCLUDE_DIRS}
                                   LIBRARIES  ${NVTOOLSEXT_LIBRARY}
                                   )

           set(CARE_HAVE_NVTOOLSEXT "0" CACHE STRING "")
       else()
           message(FATAL_ERROR "Unable to find NVTOOLSEXT with given path: ${NVTOOLSEXT_DIR}")
       endif()
    endif()
else()
    message(STATUS "Library Disabled: NVTOOLSEXT")
    set(CARE_HAVE_NVTOOLSEXT "0" CACHE STRING "")
endif()

################################
# LLNL_GlobalID
################################
if (LLNL_GLOBALID_DIR)
    include(cmake/libraries/FindLLNL_GlobalID.cmake)

    if (LLNL_GLOBALID_FOUND)
        blt_register_library( NAME       llnl_globalid
                                TREAT_INCLUDES_AS_SYSTEM ON
                                INCLUDES   ${LLNL_GLOBALID_INCLUDE_DIRS})
        set(CARE_HAVE_LLNL_GLOBALID "1" CACHE STRING "")
    else()
        message(FATAL_ERROR "Unable to find LLNL_GlobalID with given path: ${LLNL_GLOBALID_DIR}")
    endif()
else()
    message(STATUS "Library Disabled: LLNL_GlobalID")
    set(CARE_HAVE_LLNL_GLOBALID "0" CACHE STRING "")
endif()

