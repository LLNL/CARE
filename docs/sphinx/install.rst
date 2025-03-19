.. ##############################################################################
   # Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
   # project contributors. See the CARE LICENSE file for details.
   #
   # SPDX-License-Identifier: BSD-3-Clause
   ##############################################################################

=========================
Installation Instructions
=========================

``mkdir build && cd build``

``cmake -DBLT_SOURCE_DIR=/path/to/blt -DCHAI_DIR=/path/to/chai -DRAJA_DIR=/path/to/raja -DUMPIRE_DIR=/path/to/umpire ../``

``make -j``

CMake Options
=============

``-DCARE_DEBUG=ON|OFF`` Controls whether or not certain debug behavior is enabled (i.e. synchronizing after every CUDA kernel).

``-DCARE_LOOP_VVERBOSE_ENABLED=ON|OFF`` Controls whether or not the contents of all host_device_ptr's can be dumped after every RAJA loop.

``-DCARE_ENABLE_LOOP_FUSER=ON|OFF`` Controls whether or not the loop fusion (CUDA kernel fusion) capability is turned on.
