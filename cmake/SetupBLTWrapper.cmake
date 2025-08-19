##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

if(NOT BLT_LOADED)
   if(BLT_SOURCE_DIR)
       message(STATUS "CARE: Using external BLT")

       if(NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
           message(FATAL_ERROR "CARE: Given BLT_SOURCE_DIR does not contain SetupBLT.cmake")
       endif()
   else()
       message(STATUS "CARE: Using BLT submodule")

       set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/blt")

       if(NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake)
          message(FATAL_ERROR "CARE: BLT submodule is not initialized. Run `git submodule update --init` in git repository or set BLT_SOURCE_DIR to external BLT.")
       endif()
   endif()

   include(CMakeDependentOption)

   option(ENABLE_DOCS       "Enables documentation" OFF)
   option(ENABLE_EXAMPLES   "Enables examples" OFF)
   option(ENABLE_TESTS      "Enables tests" ON)
   option(ENABLE_BENCHMARKS "Enables benchmarks" OFF)

   option(ENABLE_GIT "Enables Git support" OFF)

   cmake_dependent_option(ENABLE_DOXYGEN "Enables Doxygen support"
                          ON "ENABLE_DOCS" OFF)

   cmake_dependent_option(ENABLE_SPHINX "Enables Sphinx support"
                          ON "ENABLE_DOCS" OFF)

   option(ENABLE_CLANGQUERY   "Enables clang-query support" OFF)
   option(ENABLE_CLANGTIDY    "Enables clang-tidy support" OFF)
   option(ENABLE_CPPCHECK     "Enables Cppcheck support" OFF)
   option(ENABLE_VALGRIND     "Enables Valgrind support" OFF)

   option(ENABLE_ASTYLE       "Enables AStyle support" OFF)
   option(ENABLE_CLANGFORMAT  "Enables ClangFormat support" OFF)
   option(ENABLE_UNCRUSTIFY   "Enables Uncrustify support" OFF)
   option(ENABLE_YAPF         "Enables Yapf support" OFF)
   option(ENABLE_CMAKEFORMAT  "Enables CMakeFormat support" OFF)

   option(ENABLE_FORTRAN      "Enables Fortran compiler support" OFF)

   set(BLT_CXX_STD "c++17" CACHE STRING "Set the c++ standard to use")

   if(("${BLT_CXX_STD}" STREQUAL "c++98") OR
      ("${BLT_CXX_STD}" STREQUAL "c++11") OR
      ("${BLT_CXX_STD}" STREQUAL "c++14"))
      message(FATAL_ERROR "CARE requires a minimum C++ standard of c++17. Please set BLT_CXX_STD accordingly.")
   endif()

   include(${BLT_SOURCE_DIR}/SetupBLT.cmake)
endif()
