[comment]: # (#################################################################)
[comment]: # (Copyright 2024, Lawrence Livermore National Security, LLC and CARE)
[comment]: # (project contributors. See the CARE LICENSE file for details.)
[comment]: # 
[comment]: # (SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# CARE Software Release Notes

Notes describing significant changes in each CARE release are documented
in this file.

The format of this file is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Version 0.14.0] - Release date 2024-09-11

### Added
- Added default and move constructors and move assignment operator to host\_device\_map

### Changed
- Default policies concerning reductions were updated to RAJA's newly recommended policies.
- Now using '<' in care::BinarySearch.

### Fixed
- Removed C++17 features so that CARE is C++14 compliant. The next release of CARE will require C++17.
- LLNL\_GlobalID is no longer required downstream from CARE if it is disabled in CARE.
- Const correctness fix in uniqArray API.
- Sequential IntersectArrays now allocates data in a way that is consistent with the memory model CHAI uses for Hip GPU builds.

## [Version 0.13.3] - Release date 2024-07-31

### Fixed
- Replaced loop\_work alias with seq\_work (loop\_work was removed in RAJA v2024.02.2)
- Fixed CHUNKED loop macro implementations

## [Version 0.13.2] - Release date 2024-07-29

### Changed
- Updated to Umpire/RAJA/CHAI v2024.07.0
- Updated minimum required CMake to 3.23

## [Version 0.13.1] - Release date 2024-06-27

### Changed
- Updated to CHAI v2024.02.2

## [Version 0.13.0] - Release date 2024-06-11

### Added
- Alias for execution policy specificially for kernels with reductions
- Chunked loop policies

## [Version 0.12.0] - Release date 2024-03-11

### Added
- Support for APUs with a single memory space.
- ArrayDup overloads
- LocalSortPairs for a thread local simultaneous sort
- Better support for unsigned and 64 bit integers (explicit instantiations of some algorithms, a SCAN\_LOOP\_64 macro)

### Removed
- Camp submodule
- radiuss-ci submodule
- The `chai_force_sync` function used for debugging (the corresponding functionality in CHAI has also been removed)

### Changed
- When building with submodules, they now need to be initialized recursively
- When building with external libraries, specify the install location with `-D<uppercase name>_DIR` (previously, the lowercase version was also accepted)
- Only tests are built by default now. Docs, examples, and benchmarks must be enabled explicitly.
- The `CARE_ENABLE_*` options for tests, docs, examples, and benchmarks now CMake dependent options based on the corresponding `ENABLE_*` options
- care-config.cmake has been moved to ${INSTALL\_PREFIX}/lib/cmake/care and now properly exports CMake targets for care

### Fixed
- Eliminated some unnecessary data motion for GPU builds
- Several fixes were added for shared library builds on Windows
- Some warnings have been fixed
