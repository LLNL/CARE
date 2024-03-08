[comment]: # (#################################################################)
[comment]: # (Copyright 2020-24, Lawrence Livermore National Security, LLC and CARE)
[comment]: # (project contributors. See the CARE LICENSE file for details.)
[comment]: #
[comment]: # (SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

CARE: CHAI and RAJA Extensions
===============================
CHAI and RAJA provide an excellent base on which to build portable code. CARE expands that functionality, adding new features such as loop fusion capability, a portable interface for many numerical algorithms, and additional data structures. It provides all the basics for anyone wanting to write portable code.

Getting Started
===============
```bash
git submodule update --init   # Note that --recursive is not used since all the dependencies are at the top level
mkdir build
cd build
cmake ../   # May need to pass -DCMAKE_INSTALL_PREFIX=/path/to/install/in if the next instruction fails
make -j install
```

External libraries can be used instead of the submodules if desired. For example, an external CHAI can be specified with -Dchai_DIR=/directory/containing/chai-config.cmake or -DCHAI_DIR=/path/to/chai/install/prefix. Note that if using an external CHAI, it must be configured with `-DENABLE_PICK=ON -DENABLE_PINNED=ON`.

To build with CUDA support, use `-DENABLE_CUDA=ON -DCUDA_TOOLKIT_ROOT_DIR=/path/to/cuda/toolkit`. If using external libraries, note that CAMP, Umpire, RAJA, and CHAI must also be configured with those options.
 
To build with HIP support, set `-DENABLE_HIP=ON -DHIP_ROOT_DIR=/path/to/rocm/hip/ -DHIP_CLANG_PATH=/path/to/rocm/clang`. If using external libraries, note that CAMP, Umpire, RAJA, and CHAI must also be configured with those options. If using an external BLT, note that version 0.3.6 or later is required. Other compilers besides the hip vendor-supplied clang compiler have not yet been tried for HIP builds.
 
To build with OpenMP support, use `-DENABLE_OPENMP=ON`. If using an external RAJA, it must also be configured with that option.

By default, the tests, documentation, benchmarks, and examples are built but can be turned off with -DENABLE_TESTS=OFF, -DENABLE_DOCS=OFF, -DENABLE_BENCHMARKS=OFF, and -DENABLE_EXAMPLES=OFF.

License
=======
CARE is release under the BSD-3-Clause License. See the LICENSE and NOTICE files for more details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-809741
