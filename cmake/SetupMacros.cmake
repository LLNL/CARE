##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

##########################################################################
# care_convert_to_system_includes(TARGETS <targets> [RECURSIVE])
#
# Convert include directories to system include directories.
# Used to suppress warnings from external library headers.
#
# TARGETS
#    A list of CMake targets
#
# RECURSE
#    Whether to recursively convert interface link library includes to
#    system includes
##########################################################################

macro(care_convert_to_system_includes)
   set(options RECURSIVE)
   set(singleValuedArgs)
   set(multiValuedArgs TARGETS)

   ## parse the arguments to the macro
   cmake_parse_arguments(
      arg "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN})

   if(NOT DEFINED arg_TARGETS)
      message(FATAL_ERROR "TARGETS is a required parameter for the care_convert_to_system_includes macro.")
   endif()

   foreach(_target ${arg_TARGETS})
      if(TARGET _target)
         if(${arg_RECURSIVE})
            get_target_property(_libs ${_target} INTERFACE_LINK_LIBRARIES)

            foreach(_lib ${_libs})
               care_convert_to_system_includes(TARGETS ${_lib} RECURSIVE)
            endforeach()

            unset(_libs)
         endif()

         blt_convert_to_system_includes(TARGET ${_target})
      endif()
   endforeach()
endmacro()


##########################################################################
# care_find_package(NAME <name> [REQUIRED])
#
# Finds an external package.
#
# NAME
#    Name of the package
#
# REQUIRED
#    Whether the package is required
##########################################################################

macro(care_find_package)
   set(options REQUIRED)
   set(singleValuedArgs NAME)
   set(multiValuedArgs TARGETS)

   ## parse the arguments to the macro
   cmake_parse_arguments(
      care_arg "${options}" "${singleValuedArgs}" "${multiValuedArgs}" ${ARGN})

   if(NOT DEFINED care_arg_NAME)
      message(FATAL_ERROR "NAME is a required parameter for the care_find_package macro.")
   endif()

   string(TOUPPER ${care_arg_NAME} care_name_upper)
   find_package(${care_arg_NAME} QUIET NO_DEFAULT_PATH PATHS ${${care_name_upper}_DIR})
   set(${care_name_upper}_FOUND ${${care_arg_NAME}_FOUND})

   if(${care_arg_REQUIRED} AND NOT ${${care_name_upper}_FOUND})
      message(FATAL_ERROR "Could not find ${care_arg_NAME}. Set ${care_name_upper}_DIR to the install location of ${care_arg_NAME}.")
   endif()

   unset(care_name_upper)

   if(${care_arg_TARGETS})
      care_convert_to_system_includes(TARGETS ${care_arg_TARGETS} RECURSIVE)
   endif()

endmacro()
