######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

#------------------------------------------------------------------------------
# add a target to generate documentation with Doxygen
#------------------------------------------------------------------------------

if(DOXYGEN_FOUND)
   blt_add_doxygen_target(care-doxygen)
endif()

