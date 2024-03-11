######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

#------------------------------------------------------------------------------
# add a target to generate documentation with sphinx
#------------------------------------------------------------------------------

if(SPHINX_FOUND)
   blt_add_sphinx_target(care_sphinx)
endif()

