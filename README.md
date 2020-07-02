CARE: CHAI and RAJA Extensions
===============================
CHAI and RAJA provide an excellent base on which to build portable code. CARE expands that functionality, adding new features such as loop fusion capability and a portable interface for many numerical algorithms. It provides all the basics for anyone wanting to write portable code.

Getting Started
===============
```bash
mkdir build && cd build
git submodule update --init
cmake -DCHAI_DIR=/path/to/chai -DRAJA_DIR=/path/to/raja -DUMPIRE_DIR=/path/to/umpire ../
make -j
```

Note: CHAI must be built with `-DENABLE_PICK=ON -DENABLE_PINNED=ON`.

For quick reference, the paths to CHAI, RAJA, and Umpire are the same as CMAKE\_INSTALL\_PREFIX when building those dependencies. That location defaults to /usr/local, but can be specified by passing `-DCMAKE_INSTALL_PREFIX=/desired/path` and running `make install`.

To build with CUDA support, use `-DENABLE_CUDA -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda/toolkit`. CHAI, RAJA, and Umpire must also be built with those options.
 
When compiling, please set `-DENABLE_HIP=On -DHIP_ROOT_DIR=/path/to/rocm/hip/ -DHIP_CLANG_PATH=/path/to/rocm/clang`. 
BLT needs to be updated to the development version for the HIP build to work. Even when BLT is updated, however, be sure to avoid 
running BLT's hip smoke test, since it dies with a compiler error. 
Other compilers besides the hip vendor-supplied clang compiler have not yet been tried for HIP builds.
 
To build with OpenMP support, use `-DENABLE_OPENMP`. RAJA must also be built with that option.

License
=======
CARE is release under the BSD-3-Clause License. See the LICENSE and NOTICE files for more details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-809741
