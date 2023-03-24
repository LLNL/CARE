set(GCC_HOME "/usr/tce/packages/gcc/gcc-8.3.1" CACHE PATH "")

set(CMAKE_CXX_COMPILER "/usr/tce/packages/clang/clang-14.0.4/bin/clang++" CACHE FILEPATH "Path to clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --gcc-toolchain=${GCC_HOME}" CACHE STRING "C++ compiler flags")

set(BLT_EXPORT_THIRDPARTY OFF CACHE BOOL "")

