##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CARE
# contributors. See the CARE LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

#------------------------------------------------------------------------------
# add a target to generate documentation with Doxygen
#------------------------------------------------------------------------------

if(DOXYGEN_FOUND)
   blt_add_doxygen_target(care-doxygen)
endif()

