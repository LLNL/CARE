CARE: CHAI and RAJA Extensions
===============================
CHAI and RAJA provide an excellent base on which to build portable code. CARE expands that functionality, adding new features such as loop fusion capability and a portable interface for many numerical algorithms. It provides all the basics for anyone wanting to write portable code.

Getting Started
===============
```bash
mkdir build && cd build
cmake -DBLT_SOURCE_DIR=/path/to/blt -DCHAI_DIR=/path/to/chai -DRAJA_DIR=/path/to/raja -DUMPIRE_DIR=/path/to/umpire ../
make -j
```

License
=======
CARE is release under the BSD-3-Clause License. See the LICENSE and NOTICE files for more details.

SPDX-License-Identifier: BSD-3-Clause

LLNL-CODE-809741
