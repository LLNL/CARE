//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_DEFAULT_MACROS_H_
#define _CARE_DEFAULT_MACROS_H_

// CARE config header
#include "care/config.h"

// for OMP CARE loops, only used in compatibility mode
#if defined(CARE_LEGACY_COMPATIBILITY_MODE)
#include "care/openmp.h"
#endif

// Other CARE headers
#include "care/forall.h"
#include "care/policies.h"

// This makes sure the lambdas get decorated with the right __host__ and or
// __device__ specifiers
#if defined(CARE_GPUCC) && defined(GPU_ACTIVE)
#define CARE_HOST_DEVICE_ACTIVE __host__ __device__
#define CARE_DEVICE_ACTIVE __device__
#define CARE_HOST_ACTIVE __host__
#define CARE_GLOBAL_ACTIVE __global__
#else // defined CARE_GPUCC
#define CARE_HOST_DEVICE_ACTIVE
#define CARE_DEVICE_ACTIVE
#define CARE_HOST_ACTIVE
#define CARE_GLOBAL_ACTIVE
#endif // defined CARE_GPUCC

/// Used to make sure the start and end macros match
#ifndef NDEBUG
#define CARE_NEST_BEGIN(x) { int x ;
#define CARE_NEST_END(x) x = 1 ; (void) ++x ; }
#else // !NDEBUG
#define CARE_NEST_BEGIN(x) {
#define CARE_NEST_END(x) }
#endif // !NDEBUG

/// Used to capture variables by reference into a lambda (combine with FOR_EACH)
#define CARE_REF_CAPTURE(X) , &X










////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a vanilla for loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) for (int INDEX = START_INDEX; INDEX < (int)END_INDEX; ++INDEX) CARE_NEST_BEGIN(CHECK)

#define CARE_CHECKED_FOR_LOOP_END(CHECK) CARE_NEST_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a region of raw host code.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_HOST_CODE_START(CHECK) CARE_NEST_BEGIN(CHECK)

#define CARE_CHECKED_HOST_CODE_END(CHECK) CARE_NEST_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a vanilla OpenMP 3.0 for loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) { int const __end_ndx = END_INDEX; OMP_FOR_BEGIN for (int INDEX = START_INDEX; INDEX < __end_ndx; ++INDEX) CARE_NEST_BEGIN(CHECK)

#define CARE_CHECKED_OPENMP_FOR_LOOP_END(CHECK) CARE_NEST_END(CHECK) OMP_FOR_END }










#if defined(CARE_LEGACY_COMPATIBILITY_MODE)

////////////////////////////////////////////////////////////////////////////////
///
/// Legacy compatibility mode is more performant on host-only builds.
/// It also make for easier debugging.
///
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a call to forall with the given execution policy.
///        The legacy version uses a raw for loop.
///
/// @arg[in] POLICY The execution policy
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_LOOP_START(POLICY, INDEX, START_INDEX, END_INDEX, CHECK) CARE_CHECKED_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_LOOP_END(CHECK) CARE_CHECKED_FOR_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop. The legacy version
///        uses a raw for loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_SEQUENTIAL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) CARE_CHECKED_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_SEQUENTIAL_LOOP_END(CHECK) CARE_CHECKED_FOR_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop that captures some
///        variables by reference. The legacy version uses a raw for loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
/// @arg[in] __VA_ARGS__ The variables to capture by reference
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_START(INDEX, START_INDEX, END_INDEX, CHECK, ...) CARE_CHECKED_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_END(CHECK) CARE_CHECKED_FOR_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop of length one.
///        The legacy version executes raw host code.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_HOST_KERNEL_START(CHECK) CARE_CHECKED_HOST_CODE_START(CHECK)

#define CARE_CHECKED_HOST_KERNEL_END(CHECK) CARE_CHECKED_HOST_CODE_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop of length one that
///        captures some variables by reference. The legacy version uses a raw
///        for loop.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
/// @arg[in] __VA_ARGS__ The variables to capture by reference
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_HOST_KERNEL_WITH_REF_START(CHECK, ...) CARE_CHECKED_HOST_CODE_START(CHECK)

#define CARE_CHECKED_HOST_KERNEL_WITH_REF_END(CHECK) CARE_CHECKED_HOST_CODE_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end an OpenMP RAJA loop. If OpenMP is not
///        available, executes sequentially on the host. The legacy version
///        uses raw OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_OPENMP_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) CARE_CHECKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_OPENMP_LOOP_END(CHECK) CARE_CHECKED_OPENMP_FOR_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end an OpenMP RAJA loop that captures some
///        variables by reference. The legacy version uses raw OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
/// @arg[in] __VA_ARGS__ The variables to capture by reference
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_OPENMP_LOOP_WITH_REF_START(INDEX, START_INDEX, END_INDEX, CHECK, ...) CARE_CHECKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_OPENMP_LOOP_WITH_REF_END(CHECK) CARE_CHECKED_OPENMP_FOR_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a GPU RAJA loop. If GPU is not available,
///        executes sequentially on the host. The legacy version uses raw OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_GPU_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) CARE_CHECKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_GPU_LOOP_END(CHECK) CARE_CHECKED_OPENMP_FOR_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a GPU RAJA loop of length one. If GPU is
///        not available, executes on the host. The legacy version uses raw
///        host code.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_GPU_KERNEL_START(CHECK) CARE_CHECKED_HOST_CODE_START(CHECK)

#define CARE_CHECKED_GPU_KERNEL_END(CHECK) CARE_CHECKED_HOST_CODE_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host. The legacy version uses raw OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) CARE_CHECKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_PARALLEL_LOOP_END(CHECK) CARE_CHECKED_OPENMP_FOR_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a GPU RAJA loop of length one. If GPU is
///        not available, executes on the host. The legacy version uses raw
///        host code (no need for openmp since it would only use one thread
///        anyway).
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_PARALLEL_KERNEL_START(CHECK) CARE_CHECKED_HOST_CODE_START(CHECK)

#define CARE_CHECKED_PARALLEL_KERNEL_END(CHECK) CARE_CHECKED_HOST_CODE_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential loop of length one. If GPU is
///        available, also executes on the device. The legacy version only
///        executes code on the host.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_EVERYWHERE_KERNEL_START(CHECK) CARE_CHECKED_HOST_CODE_START(CHECK)

#define CARE_CHECKED_EVERYWHERE_KERNEL_END(CHECK) CARE_CHECKED_HOST_CODE_END(CHECK)










#else // defined(CARE_LEGACY_COMPATIBILITY_MODE)

////////////////////////////////////////////////////////////////////////////////
///
/// If we are not in legacy compatibility mode, define the macros as they
/// are intended.
///
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///
/// Big thanks to Danielle Sikich for putting the initial macro definitions
/// together for OpenMP, OpenACC, and GPU
///
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a call to forall with the given execution policy.
///
/// @arg[in] POLICY The execution policy
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_LOOP_START(POLICY, INDEX, START_INDEX, END_INDEX, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(POLICY, __FILE__, __LINE__, START_INDEX, END_INDEX, [=] CARE_HOST_DEVICE (const int INDEX) {

#define CARE_CHECKED_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_SEQUENTIAL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::sequential{}, __FILE__, __LINE__, START_INDEX, END_INDEX, [=] (const int INDEX) {

#define CARE_CHECKED_SEQUENTIAL_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop that captures some
///        variables by reference.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
/// @arg[in] __VA_ARGS__ The variables to capture by reference
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_START(INDEX, START_INDEX, END_INDEX, CHECK, ...) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::sequential{}, __FILE__, __LINE__, START_INDEX, END_INDEX, [= FOR_EACH(CARE_REF_CAPTURE, __VA_ARGS__)] (const int INDEX) {

#define CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop of length one.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_HOST_KERNEL_START(CHECK) { \
   CARE_NEST_BEGIN(CHECK) \
   care::forall(care::sequential{}, __FILE__, __LINE__, 0, 1, [=] (const int) {

#define CARE_CHECKED_HOST_KERNEL_END(CHECK) }); \
   CARE_NEST_END(CHECK) }

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop of length one that
///        captures some variables by reference.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
/// @arg[in] __VA_ARGS__ The variables to capture by reference
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_HOST_KERNEL_WITH_REF_START(CHECK, ...) { \
   CARE_NEST_BEGIN(CHECK) \
   care::forall(care::sequential{}, __FILE__, __LINE__, 0, 1, [= FOR_EACH(CARE_REF_CAPTURE, __VA_ARGS__)] (const int) {

#define CARE_CHECKED_HOST_KERNEL_WITH_REF_END(CHECK) }); \
   CARE_NEST_END(CHECK) }

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end an OpenMP RAJA loop. If OpenMP is not
///        available, executes sequentially on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_OPENMP_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::openmp{}, __FILE__, __LINE__, START_INDEX, END_INDEX, [=] (const int INDEX) {

#define CARE_CHECKED_OPENMP_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end an OpenMP RAJA loop that captures some
///        variables by reference.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
/// @arg[in] __VA_ARGS__ The variables to capture by reference
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_OPENMP_LOOP_WITH_REF_START(INDEX, START_INDEX, END_INDEX, CHECK, ...) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::openmp{}, __FILE__, __LINE__, START_INDEX, END_INDEX, [= FOR_EACH(CARE_REF_CAPTURE, __VA_ARGS__)] (const int INDEX) {

#define CARE_CHECKED_OPENMP_LOOP_WITH_REF_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end an OpenMP RAJA loop of length one. If
///        OpenMP is not available, executes sequentially.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_OPENMP_KERNEL_START(CHECK) { \
   CARE_NEST_BEGIN(CHECK) \
   care::forall(care::openmp{}, __FILE__, __LINE__, 0, 1, [=] (const int) {

#define CARE_CHECKED_OPENMP_KERNEL_END(CHECK) }); \
   CARE_NEST_END(CHECK) }

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a GPU RAJA loop. If GPU is not available,
///        executes sequentially on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_GPU_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::gpu{}, __FILE__, __LINE__, START_INDEX, END_INDEX, [=] CARE_DEVICE_ACTIVE (const int INDEX) {

#define CARE_CHECKED_GPU_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a GPU RAJA loop of length one. If GPU is
///        not available, executes on the host.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_GPU_KERNEL_START(CHECK) { \
   CARE_NEST_BEGIN(CHECK) \
   care::forall(care::gpu{}, __FILE__, __LINE__, 0, 1, [=] CARE_DEVICE_ACTIVE (const int) {

#define CARE_CHECKED_GPU_KERNEL_END(CHECK) }); \
   CARE_NEST_END(CHECK) }

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::parallel{}, __FILE__, __LINE__, START_INDEX, END_INDEX, [=] CARE_DEVICE_ACTIVE (const int INDEX) {

#define CARE_CHECKED_PARALLEL_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a GPU RAJA loop of length one. If GPU is
///        not available, executes on the host.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_PARALLEL_KERNEL_START(CHECK) { \
   CARE_NEST_BEGIN(CHECK) \
   care::forall(care::parallel{}, __FILE__, __LINE__, 0, 1, [=] CARE_DEVICE_ACTIVE (const int) {

#define CARE_CHECKED_PARALLEL_KERNEL_END(CHECK) }); \
   CARE_NEST_END(CHECK) }

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential loop of length one. If GPU is
///        available, also executes on the device.
///
/// @note This should execute on the device even if GPU_ACTIVE is not defined.
///       The reason for this is that managed_ptrs are always constructed on
///       both the host and device, and this macro is used to update both the
///       host and device objects to keep them in sync.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_EVERYWHERE_KERNEL_START(CHECK) { \
   CARE_NEST_BEGIN(CHECK) \
   care::forall(care::raja_chai_everywhere{}, __FILE__, __LINE__, 0, 1, [=] CARE_HOST_DEVICE (const int) {

#define CARE_CHECKED_EVERYWHERE_KERNEL_END(CHECK) }); \
   CARE_NEST_END(CHECK) }

#endif // defined(CARE_LEGACY_COMPATIBILITY_MODE)










////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a call to forall.
///
/// @arg[in] POLICY The execution policy to use
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_LOOP(POLICY, INDEX, START_INDEX, END_INDEX) CARE_CHECKED_LOOP_START(POLICY, INDEX, START_INDEX, END_INDEX, care_loop_check)

#define CARE_LOOP_END CARE_CHECKED_LOOP_END(care_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_SEQUENTIAL_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_SEQUENTIAL_LOOP_START(INDEX, START_INDEX, END_INDEX, care_sequential_loop_check)

#define CARE_SEQUENTIAL_LOOP_END CARE_CHECKED_SEQUENTIAL_LOOP_END(care_sequential_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop that captures some
///        variables by reference.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] __VA_ARGS__ The variables to capture by reference
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_SEQUENTIAL_REF_LOOP(INDEX, START_INDEX, END_INDEX, ...) CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_START(INDEX, START_INDEX, END_INDEX, care_sequential_ref_loop_check, __VA_ARGS__)

#define CARE_SEQUENTIAL_REF_LOOP_END CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_END(care_sequential_ref_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop. The P indicates
///        that pointers are used in the loop (and thus it is unsafe to run
///        on the device).
///
/// @note The P is used to indicate to uncrustify that pointers can be used
///       in this loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_SEQUENTIAL_P_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_SEQUENTIAL_LOOP_START(INDEX, START_INDEX, END_INDEX, care_sequential_p_loop_check)

#define CARE_SEQUENTIAL_P_LOOP_END CARE_CHECKED_SEQUENTIAL_LOOP_END(care_sequential_p_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop that captures some
///        variables by reference. The P indicates that pointers are used in
///        the loop (and thus it is unsafe to run on the device).
///
/// @note The P is used to indicate to uncrustify that pointers can be used
///       in this loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] __VA_ARGS__ The variables to capture by reference
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_SEQUENTIAL_REF_P_LOOP(INDEX, START_INDEX, END_INDEX, ...) CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_START(INDEX, START_INDEX, END_INDEX, care_sequential_ref_p_loop_check, __VA_ARGS__)

#define CARE_SEQUENTIAL_REF_P_LOOP_END CARE_CHECKED_SEQUENTIAL_LOOP_WITH_REF_END(care_sequential_ref_p_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop. The P indicates
///        that pointers are used in the loop (and thus it is unsafe to run
///        on the device).
///
///        STREAM is an alias to PARALLEL that indicates not much work is taking place.
///        The P indicates it is not safe to run on the device (yet).
///
/// @note The P is used to indicate to uncrustify that pointers can be used
///       in this loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_STREAM_P_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_SEQUENTIAL_LOOP_START(INDEX, START_INDEX, END_INDEX, care_stream_p_loop_check)

#define CARE_STREAM_P_LOOP_END CARE_CHECKED_SEQUENTIAL_LOOP_END(care_stream_p_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end an OpenMP RAJA loop. If OpenMP is not
///        available, executes sequentially on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_OPENMP_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_OPENMP_LOOP_START(INDEX, START_INDEX, END_INDEX, care_openmp_loop_check)

#define CARE_OPENMP_LOOP_END CARE_CHECKED_OPENMP_LOOP_END(care_openmp_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end an OpenMP RAJA loop that captures some
///        variables by reference.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] __VA_ARGS__ The variables to capture by reference
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_OPENMP_REF_LOOP(INDEX, START_INDEX, END_INDEX, ...) CARE_CHECKED_OPENMP_LOOP_WITH_REF_START(INDEX, START_INDEX, END_INDEX, ravi_openmp_ref_loop_check, __VA_ARGS__)

#define CARE_OPENMP_REF_LOOP_END CARE_CHECKED_OPENMP_LOOP_WITH_REF_END(ravi_openmp_ref_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host.
///
///        WORK is an alias to PARALLEL that indicates a lot of work is taking place.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_WORK_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, care_work_loop_check)

#define CARE_WORK_LOOP_END CARE_CHECKED_PARALLEL_LOOP_END(care_work_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host.
///
///        STREAM is an alias to PARALLEL that indicates not much work is taking place.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_STREAM_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, care_stream_loop_check)

#define CARE_STREAM_LOOP_END CARE_CHECKED_PARALLEL_LOOP_END(care_stream_loop_check)


////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host.
///
///        REDUCE is an alias to PARALLEL that indicates a reduction is taking place.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_REDUCE_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, care_reduce_loop_check)

#define CARE_REDUCE_LOOP_END CARE_CHECKED_PARALLEL_LOOP_END(care_reduce_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential RAJA loop of length one.
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_HOST_KERNEL CARE_CHECKED_HOST_KERNEL_START(care_host_kernel_check)

#define CARE_HOST_KERNEL_END CARE_CHECKED_HOST_KERNEL_END(care_host_kernel_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end an OpenMP RAJA loop of length one. If
///        OpenMP is not available, executes sequentially.
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_OPENMP_KERNEL CARE_CHECKED_OPENMP_KERNEL_START(care_openmp_kernel_check)

#define CARE_OPENMP_KERNEL_END CARE_CHECKED_OPENMP_KERNEL_END(care_openmp_kernel_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a GPU RAJA loop of length one. If GPU is
///        not available, executes on the host.
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_GPU_KERNEL CARE_CHECKED_GPU_KERNEL_START(care_cuda_kernel_check)

#define CARE_GPU_KERNEL_END CARE_CHECKED_GPU_KERNEL_END(care_cuda_kernel_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a GPU RAJA loop of length one. If GPU is
///        not available, executes on the host.
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_PARALLEL_KERNEL CARE_CHECKED_PARALLEL_KERNEL_START(care_parallel_kernel_check)

#define CARE_PARALLEL_KERNEL_END CARE_CHECKED_PARALLEL_KERNEL_END(care_parallel_kernel_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a sequential loop of length one. If GPU is
///        available, also executes on the device.
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_EVERYWHERE_KERNEL { CARE_CHECKED_EVERYWHERE_KERNEL_START(care_everywhere_kernel_check)

#define CARE_EVERYWHERE_KERNEL_END CARE_CHECKED_EVERYWHERE_KERNEL_END(care_everywhere_kernel_check) }

#endif // !defined(_CARE_DEFAULT_MACROS_H_)

