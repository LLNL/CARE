##############################################################################
# Copyright (c) Lawrence Livermore National Security, LLC and other CARE
# contributors. See the CARE LICENSE and COPYRIGHT files for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

if (SPHINX_FOUND)
   add_subdirectory(sphinx)
endif()

if (DOXYGEN_FOUND)
   add_subdirectory(doxygen)
else()
   message(WARNING "ENABLE_DOCUMENTATION=On, but Doxygen is not found. \
                    Documentation will not be built.")
endif()

