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

To build with HIP support, be sure to module load opt, module load rocm, and setenv HCC\_AMDGPU\_TARGET gfx900 prior to compiling. 
When compiling, please set `-DENABLE_HIP=On -DHIP_ROOT_DIR=/opt/rocm-3.5.0/hip -DHIP_CLANG_PATH=/opt/rocm-3.5.0/llvm/bin/ 
-DCMAKE_C_COMPILER=/opt/rocm-3.5.0/llvm/bin/clang -DCMAKE_CXX_COMPILER=/opt/rocm-3.5.0/llvm/bin/clang++ `. 
BLT needs to be updated to the development version for the HIP build to work. Even when BLT is updated, however, be sure to avoid 
running BLT's hip smoke test, since it dies with a compiler error. 
Other compilers besides the vendor-supplied clang compiler (packaged in /opt/rocm-3.5.0/llvm/bin/) have not yet been tried for HIP builds.
 
To build with OpenMP support, use `-DENABLE_OPENMP`. RAJA must also be built with that option.

License
=======
CARE is release under the BSD-3-Clause License. See the LICENSE and NOTICE files for more details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-809741
