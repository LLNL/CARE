#!/usr/bin/env bash

######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

mkdir build-cuda
cd build-cuda
cmake -C ../configs/lc/blueos/nvcc10_clang12.cmake ../
make -j
rm -rf build-cuda
