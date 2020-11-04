#!/usr/bin/env bash

######################################################################################
# Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
######################################################################################

mkdir build
cd build
cmake -C ../configs/lc/blueos/nvcc10.1.243_clang8.0.1.cmake ../
make -j
rm -rf build
