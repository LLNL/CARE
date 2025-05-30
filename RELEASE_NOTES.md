[comment]: # (#################################################################)
[comment]: # (Copyright 2020-25, Lawrence Livermore National Security, LLC and CARE)
[comment]: # (project contributors. See the CARE LICENSE file for details.)
[comment]: # 
[comment]: # (SPDX-License-Identifier: BSD-3-Clause)
[comment]: # (#################################################################)

# CARE Software Release Notes

Notes describing significant changes in each CARE release are documented
in this file.

The format of this file is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [Unreleased] - Release date YYYY-MM-DD

### Fixed
- Fixed build errors when CARE is configured with ENABLE\_OPENMP or CARE\_ENABLE\_GPU\_SIMULATION\_MODE
- Fixed some build warnings

## [Version 0.15.1] - Release date 2025-04-07

### Added
- Added support for RAJA MultiReducers (Min/Max/Sum).

### Changed
- Changed RAJA reduce policy for CUDA to RAJA::cuda\_reduce\_atomic.
- Rearranged template parameters of care::sortKeyValueArrays (used by care::KeyValueSorter) for ease of use

### Fixed
- Only enable calls to cub::DeviceMergeSort when it is available (used by care::sortArray and care::KeyValueSorter when the type is not arithmetic)
- Fixes inputs to [hip]cub::DeviceMergeSort::StableSortKeys (used by care::sortArray when the type is not arithmetic)
- Avoids hardcoding one overload of care::sortArray to use [hip]cub::DeviceRadixSort
- Fixes a case where care::sort\_uniq should not modify the input array
- Miscellaneous fixes for care::host\_device\_map
- Clarified documentation for care::BinarySearch
- Added missing attributes to functions for building as a shared library on Windows
- Moved helper function to be accessible when the loop fuser is disabled

### Removed
- Removed dead ENABLE\_PICK option (corresponding option has been removed from CHAI)

## [Version 0.15.0] - Release date 2025-03-20

### Added
- Added CARE\_DEEP\_COPY\_RAW\_PTR configuration option.
- Added ATOMIC\_SUB, ATOMIC\_LOAD, ATOMIC\_STORE, ATOMIC\_EXCHANGE, and ATOMIC\_CAS macros.

### Removed
- Removed Accessor template parameter from host\_device\_ptr.
- Removed NoOpAccessor and RaceConditionAccessor. It is recommended to use ThreadSanitizer (TSAN) instead to locate race conditions.
- Removed CARE\_ENABLE\_RACE\_DETECTION configuration option.
- Removed implicit conversions between raw pointers and host\_device\_ptrs/host\_ptrs and the corresponding CARE\_ENABLE\_IMPLICIT\_CONVERSIONS configuration option.

### Changed
- Renamed host\_device\_ptr::getPointer to host\_device\_ptr::data.

### Fixed
- Replaced calls to chai::ManagedArray::getPointer (previously deprecated and now removed) with calls to chai::ManagedArray::data.

### Updated
- Updated to BLT v0.7.0
- Updated to Umpire v2025.03.0
- Updated to RAJA v2025.03.0
- Updated to CHAI v2025.03.0

## [Version 0.14.1] - Release date 2024-10-15

### Fixed
- Explicitly define host\_device\_map constructors since some versions of CUDA do not properly generate them.

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
