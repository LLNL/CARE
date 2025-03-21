//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

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
#include "care/FOREACHMACRO.h"
#include "care/GPUMacros.h"
#include "care/policies.h"

/// Used to make sure the start and end macros match
#ifndef NDEBUG
#define CARE_NEST_BEGIN(x) { int x ;
#define CARE_NEST_END(x) x = 1 ; (void) ++x ; x = x;}
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
#define CARE_CHECKED_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) CARE_NEST_BEGIN(CHECK) \
auto _care_for_loop_end_index = END_INDEX; \
decltype(_care_for_loop_end_index) _care_for_loop_begin_index = START_INDEX; \
for (auto INDEX = _care_for_loop_begin_index; INDEX < _care_for_loop_end_index; ++INDEX)  {

#define CARE_CHECKED_FOR_LOOP_END(CHECK) } CARE_NEST_END(CHECK)

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
#define CARE_CHECKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) {\
 CARE_NEST_BEGIN(CHECK) \
 auto const _care_openmp_for_loop_end_ndx = END_INDEX; \
 decltype(_care_openmp_for_loop_end_ndx) _care_openmp_for_loop_begin_ndx = START_INDEX; \
OMP_FOR_BEGIN for (auto INDEX = _care_openmp_for_loop_begin_ndx; INDEX < _care_openmp_for_loop_end_ndx; ++INDEX) {\

#define CARE_CHECKED_OPENMP_FOR_LOOP_END(CHECK) } OMP_FOR_END CARE_NEST_END(CHECK) }

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked vanilla OpenMP 3.0 for loop.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) {\
 CARE_NEST_BEGIN(CHECK) \
 auto const _care_openmp_for_loop_end_ndx = END_INDEX; \
 decltype(_care_openmp_for_loop_end_ndx) _care_openmp_for_loop_ndx = START_INDEX; \
 decltype(_care_openmp_for_loop_end_ndx) _care_open_chunked_for_loop_chunk_size = CHUNK_SIZE > 0 ? CHUNK_SIZE : END_INDEX - START_INDEX ; \
 while (_care_openmp_for_loop_begin_ndx < _care_openmp_for_loop_end_ndx) { \
    decltype(_care_openmp_for_loop_end_ndx) _care_openmp_for_loop_chunk_begin_ndx = _care_openmp_for_loop_ndx ; \
    decltype(_care_openmp_for_loop_end_ndx) _care_openmp_for_loop_chunk_end_ndx = (_care_openmp_for_loop_ndx + _care_open_chunked_for_loop_chunk_size) ? _care_openmp_for_loop_ndx + _care_open_chunked_for_loop_chunk_size : _care_openmp_for_loop_end_ndx ; \
OMP_FOR_BEGIN for (auto INDEX = _care_openmp_for_loop_chunk_begin_ndx; INDEX < _care_openmp_for_loop_chunk_end_ndx; ++INDEX) {\

#define CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_END(CHECK) } OMP_FOR_END } CARE_NEST_END(CHECK) }









#if CARE_LEGACY_COMPATIBILITY_MODE

////////////////////////////////////////////////////////////////////////////////
///
/// Legacy compatibility mode is more performant on host-only builds.
/// It also make for easier debugging.
///
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macro for skipping the remainder of the current iteration in a CARE loop.
///        The legacy version uses a "continue" statement.
///
/// @note This should only be used within the outermost scope inside a CARE loop
///       (a regular "continue" can be used inside nested loops within a CARE loop).
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_LOOP_CONTINUE continue

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
/// @brief Macros that start and end a call to forall with the given execution policy.
///        This is for compatibility with chunked GPU loops.
///        The legacy version uses a raw for loop.
///
/// @arg[in] POLICY The execution policy
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Not used
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_LOOP_START(POLICY, INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) CARE_CHECKED_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_CHUNKED_LOOP_END(CHECK) CARE_CHECKED_FOR_LOOP_END(CHECK)

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
/// @brief Macros that start and end a chunked OpenMP RAJA loop. If OpenMP is not
///        available, executes sequentially on the host. The legacy version
///        uses raw OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_OPENMP_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK)

#define CARE_CHECKED_CHUNKED_OPENMP_LOOP_END(CHECK) CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_END(CHECK)

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
/// @brief Macros that start and end a chunked GPU RAJA loop. If GPU is not available,
///        executes sequentially on the host. The legacy version uses raw OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_GPU_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK)

#define CARE_CHECKED_CHUNKED_GPU_LOOP_END(CHECK) CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_END(CHECK)

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

#define CARE_CHECKED_REDUCE_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) \
   CARE_CHECKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_REDUCE_LOOP_END(CHECK) CARE_CHECKED_PARALLEL_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host. The legacy version uses raw OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK)

#define CARE_CHECKED_CHUNKED_PARALLEL_LOOP_END(CHECK) CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_END(CHECK)

#define CARE_CHECKED_CHUNKED_REDUCE_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) \
   CARE_CHECKED_CHUNKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK)

#define CARE_CHECKED_CHUNKED_REDUCE_LOOP_END(CHECK) CARE_CHECKED_CHUNKED_PARALLEL_LOOP_END(CHECK)

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
/// @brief Macros that start and end a RAJA loop that uses at least one
///        managed_ptr. If GPU is available, and managed_ptr is available
///        on the device, executes on the device. If GPU is not available
///        but OpenMP is, executes in parallel on the host. Otherwise,
///        executes sequentially on the host. The legacy version uses raw
///        OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_MANAGED_PTR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) CARE_CHECKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_MANAGED_PTR_LOOP_END(CHECK) CARE_CHECKED_OPENMP_FOR_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked RAJA loop that uses at least one
///        managed_ptr. If GPU is available, and managed_ptr is available
///        on the device, executes on the device. If GPU is not available
///        but OpenMP is, executes in parallel on the host. Otherwise,
///        executes sequentially on the host. The legacy version uses raw
///        OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_MANAGED_PTR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK)

#define CARE_CHECKED_CHUNKED_MANAGED_PTR_LOOP_END(CHECK) CARE_CHECKED_CHUNKED_OPENMP_FOR_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros for updating/initializing managed_ptrs.
///        Will start and end a sequential loop of length one.
///        If GPU is available, also executes on the device.
///        The legacy version only executes code on the host.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_MANAGED_PTR_UPDATE_KERNEL_START(CHECK) CARE_CHECKED_HOST_CODE_START(CHECK)

#define CARE_CHECKED_MANAGED_PTR_UPDATE_KERNEL_END(CHECK) CARE_CHECKED_HOST_CODE_END(CHECK)










#else // CARE_LEGACY_COMPATIBILITY_MODE

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
/// @brief Macro for skipping the remainder of the current iteration in a CARE loop.
///        In a lambda, this is done with a "return" statement.
///
/// @note This should only be used within the outermost scope inside a CARE loop
///       (a regular "continue" can be used inside nested loops within a CARE loop).
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_LOOP_CONTINUE return

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
      care::forall(POLICY, __FILE__, __LINE__, START_INDEX, END_INDEX, 0, [=] CARE_HOST_DEVICE (const int INDEX) {

#define CARE_CHECKED_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a call to a chunked forall with the given execution policy.
///
/// @arg[in] POLICY The execution policy
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum chunk size for each kernel
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_LOOP_START(POLICY, INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(POLICY, __FILE__, __LINE__, START_INDEX, END_INDEX, CHUNK_SIZE, [=] CARE_HOST_DEVICE (const int INDEX) {

#define CARE_CHECKED_CHUNKED_LOOP_END(CHECK) }); \
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
      care::forall(care::sequential{}, __FILE__, __LINE__, START_INDEX, END_INDEX, 0, [=] (const int INDEX) {

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
      care::forall(care::sequential{}, __FILE__, __LINE__, START_INDEX, END_INDEX, 0, [= FOR_EACH(CARE_REF_CAPTURE, __VA_ARGS__)] (const int INDEX) {

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
   care::forall(care::sequential{}, __FILE__, __LINE__, 0, 1, 0, [=] (const int) {

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
   care::forall(care::sequential{}, __FILE__, __LINE__, 0, 1, 0, [= FOR_EACH(CARE_REF_CAPTURE, __VA_ARGS__)] (const int) {

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
      care::forall(care::openmp{}, __FILE__, __LINE__, START_INDEX, END_INDEX, 0, [=] (const int INDEX) {

#define CARE_CHECKED_OPENMP_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked OpenMP RAJA loop. If OpenMP is not
///        available, executes sequentially on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum size of kernel
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_OPENMP_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::openmp{}, __FILE__, __LINE__, START_INDEX, END_INDEX, CHUNK_SIZE, [=] (const int INDEX) {

#define CARE_CHECKED_CHUNKED_OPENMP_LOOP_END(CHECK) }); \
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
      care::forall(care::openmp{}, __FILE__, __LINE__, START_INDEX, END_INDEX, 0, [= FOR_EACH(CARE_REF_CAPTURE, __VA_ARGS__)] (const int INDEX) {

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
   care::forall(care::openmp{}, __FILE__, __LINE__, 0, 1, 0, [=] (const int) {

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
      care::forall(care::gpu{}, __FILE__, __LINE__, START_INDEX, END_INDEX, 0, [=] CARE_DEVICE (const int INDEX) {

#define CARE_CHECKED_GPU_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked GPU RAJA loop. If GPU is not available,
///        executes sequentially on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_GPU_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::gpu{}, __FILE__, __LINE__, START_INDEX, END_INDEX, CHUNK_SIZE, [=] CARE_DEVICE (const int INDEX) {

#define CARE_CHECKED_CHUNKED_GPU_LOOP_END(CHECK) }); \
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
   care::forall(care::gpu{}, __FILE__, __LINE__, 0, 1, 0, [=] CARE_DEVICE (const int) {

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

#define CARE_CHECKED_POLICY_LOOP_START(POLICY, INDEX, START_INDEX, END_INDEX, CHECK) { \
   auto _care_checked_loop_end = END_INDEX; \
   decltype(_care_checked_loop_end) _care_checked_loop_begin = START_INDEX; \
   if (_care_checked_loop_end > _care_checked_loop_begin) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(POLICY{}, __FILE__, __LINE__, _care_checked_loop_begin, _care_checked_loop_end, 0, [=] CARE_DEVICE (decltype(_care_checked_loop_end) INDEX) {

#define CARE_CHECKED_POLICY_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

#define CARE_CHECKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) \
   CARE_CHECKED_POLICY_LOOP_START(care::parallel,INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_PARALLEL_LOOP_END(CHECK) CARE_CHECKED_POLICY_LOOP_END(CHECK)

#define CARE_CHECKED_REDUCE_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) \
   CARE_CHECKED_POLICY_LOOP_START(care::gpu_reduce,INDEX, START_INDEX, END_INDEX, CHECK)

#define CARE_CHECKED_REDUCE_LOOP_END(CHECK) CARE_CHECKED_POLICY_LOOP_END(CHECK)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////

#define CARE_CHECKED_CHUNKED_POLICY_LOOP_START(POLICY, INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) { \
   auto _care_checked_loop_end = END_INDEX; \
   decltype(_care_checked_loop_end) _care_checked_loop_begin = START_INDEX; \
   if (_care_checked_loop_end > _care_checked_loop_begin) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(POLICY{}, __FILE__, __LINE__, _care_checked_loop_begin, _care_checked_loop_end, CHUNK_SIZE, [=] CARE_DEVICE (decltype(_care_checked_loop_end) INDEX) {

#define CARE_CHECKED_CHUNKED_POLICY_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

#define CARE_CHECKED_CHUNKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) \
   CARE_CHECKED_CHUNKED_POLICY_LOOP_START(care::parallel,INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK)

#define CARE_CHECKED_CHUNKED_PARALLEL_LOOP_END(CHECK) CARE_CHECKED_CHUNKED_POLICY_LOOP_END(CHECK)

#define CARE_CHECKED_CHUNKED_REDUCE_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) \
   CARE_CHECKED_CHUNKED_POLICY_LOOP_START(care::gpu_reduce,INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK)

#define CARE_CHECKED_CHUNKED_REDUCE_LOOP_END(CHECK) CARE_CHECKED_CHUNKED_POLICY_LOOP_END(CHECK)



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
   care::forall(care::parallel{}, __FILE__, __LINE__, 0, 1, 0, [=] CARE_DEVICE (const int) {

#define CARE_CHECKED_PARALLEL_KERNEL_END(CHECK) }); \
   CARE_NEST_END(CHECK) }

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a RAJA loop that uses at least one
///        managed_ptr. If GPU is available, and managed_ptr is available
///        on the device, executes on the device. If GPU is not available
///        but OpenMP is, executes in parallel on the host. Otherwise,
///        executes sequentially on the host. The legacy version uses raw
///        OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_MANAGED_PTR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::managed_ptr_read{}, __FILE__, __LINE__, START_INDEX, END_INDEX, 0, [=] CARE_MANAGED_PTR_DEVICE (const int INDEX) {

#define CARE_CHECKED_MANAGED_PTR_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked RAJA loop that uses at least one
///        managed_ptr. If GPU is available, and managed_ptr is available
///        on the device, executes on the device. If GPU is not available
///        but OpenMP is, executes in parallel on the host. Otherwise,
///        executes sequentially on the host. The legacy version uses raw
///        OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_CHUNKED_MANAGED_PTR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, CHECK) { \
   if (END_INDEX > START_INDEX) { \
      CARE_NEST_BEGIN(CHECK) \
      care::forall(care::managed_ptr_read{}, __FILE__, __LINE__, START_INDEX, END_INDEX, CHUNK_SIZE, [=] CARE_MANAGED_PTR_DEVICE (const int INDEX) {

#define CARE_CHECKED_CHUNKED_MANAGED_PTR_LOOP_END(CHECK) }); \
   CARE_NEST_END(CHECK) }}

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros for updating/initializing managed_ptrs.
///        Will start and end a sequential loop of length one.
///        If GPU is available, also executes on the device to
///        keep both the host and device objects in sync.
///
/// @arg[in] CHECK The variable to check that the start and end macros match
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHECKED_MANAGED_PTR_UPDATE_KERNEL_START(CHECK) { \
   CARE_NEST_BEGIN(CHECK) \
   care::forall(care::managed_ptr_write{}, __FILE__, __LINE__, 0, 1, 0, [=] CARE_MANAGED_PTR_HOST_DEVICE (const int) {

#define CARE_CHECKED_MANAGED_PTR_UPDATE_KERNEL_END(CHECK) }); \
   CARE_NEST_END(CHECK) }

#endif  // CARE_LEGACY_COMPATIBILITY_MODE










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
/// @brief Macros that start and end a call to chunked forall.
///
/// @arg[in] POLICY The execution policy to use
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHUNKED_LOOP(POLICY, INDEX, START_INDEX, END_INDEX, CHUNK_SIZE) CARE_CHECKED_CHUNKED_LOOP_START(POLICY, INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, care_loop_chunked_check)

#define CARE_CHUNKED_LOOP_END CARE_CHECKED_CHUNKED_LOOP_END(care_loop_chunked_check)

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
/// @brief Macros that start and end a chunked OpenMP RAJA loop. If OpenMP is not
///        available, executes sequentially on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHUNKED_OPENMP_LOOP(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE) CARE_CHECKED_CHUNKED_OPENMP_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, care_openmp_loop_chunked_check)

#define CARE_CHUNKED_OPENMP_LOOP_END CARE_CHECKED_CHUNKED_OPENMP_LOOP_END(care_openmp_loop_chunked_check)

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
/// @brief Macros that start and end a RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available, executes
///        sequentially on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_GPU_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_GPU_LOOP_START(INDEX, START_INDEX, END_INDEX, care_gpu_loop_check)

#define CARE_GPU_LOOP_END CARE_CHECKED_GPU_LOOP_END(care_gpu_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available, executes
///        sequentially on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHUNKED_GPU_LOOP(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE) CARE_CHECKED_CHUNKED_GPU_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, care_gpu_loop_chunked_check)

#define CARE_CHUNKED_GPU_LOOP_END CARE_CHECKED_CHUNKED_GPU_LOOP_END(care_gpu_loop_chunked_check)

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
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_PARALLEL_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, care_parallel_loop_check)

#define CARE_PARALLEL_LOOP_END CARE_CHECKED_PARALLEL_LOOP_END(care_parallel_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHUNKED_PARALLEL_LOOP(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE) CARE_CHECKED_CHUNKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, care_parallel_loop_chunked_check)

#define CARE_CHUNKED_PARALLEL_LOOP_END CARE_CHECKED_CHUNKED_PARALLEL_LOOP_END(care_parallel_loop_chunked_check)


////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a RAJA loop that uses at least one
///        managed_ptr. If GPU is available, and managed_ptr is available
///        on the device, executes on the device. If GPU is not available
///        but OpenMP is, executes in parallel on the host. Otherwise,
///        executes sequentially on the host. The legacy version uses raw
///        OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_MANAGED_PTR_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_MANAGED_PTR_LOOP_START(INDEX, START_INDEX, END_INDEX, care_managed_ptr_read_loop_check)

#define CARE_MANAGED_PTR_LOOP_END CARE_CHECKED_MANAGED_PTR_LOOP_END(care_managed_ptr_read_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked RAJA loop that uses at least one
///        managed_ptr. If GPU is available, and managed_ptr is available
///        on the device, executes on the device. If GPU is not available
///        but OpenMP is, executes in parallel on the host. Otherwise,
///        executes sequentially on the host. The legacy version uses raw
///        OpenMP.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHUNKED_MANAGED_PTR_LOOP(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE) CARE_CHECKED_CHUNKED_MANAGED_PTR_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, care_managed_ptr_read_loop_chunked_check)

#define CARE_CHUNKED_MANAGED_PTR_LOOP_END CARE_CHECKED_CHUNKED_MANAGED_PTR_LOOP_END(care_managed_ptr_read_loop_chunked_check)

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
/// @brief Macros that start and end a chunked parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host.
///
///        WORK is an alias to PARALLEL that indicates a lot of work is taking place.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHUNKED_WORK_LOOP(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE) CARE_CHECKED_CHUNKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, care_work_loop_chunked_check)

#define CARE_CHUNKED_WORK_LOOP_END CARE_CHECKED_CHUNKED_PARALLEL_LOOP_END(care_work_loop_chunked_check)

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
/// @brief Macros that start and end a chunked parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host.
///
///        STREAM is an alias to PARALLEL that indicates not much work is taking place.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHUNKED_STREAM_LOOP(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE) CARE_CHECKED_CHUNKED_PARALLEL_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, care_stream_loop_chunked_check)

#define CARE_CHUNKED_STREAM_LOOP_END CARE_CHECKED_CHUNKED_PARALLEL_LOOP_END(care_stream_loop_chunked_check)

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
#define CARE_REDUCE_LOOP(INDEX, START_INDEX, END_INDEX) CARE_CHECKED_REDUCE_LOOP_START(INDEX, START_INDEX, END_INDEX, care_reduce_loop_check)

#define CARE_REDUCE_LOOP_END CARE_CHECKED_REDUCE_LOOP_END(care_reduce_loop_check)

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros that start and end a chunked parallel RAJA loop. If GPU is available,
///        executes on the device. If GPU is not available but OpenMP is,
///        executes in parallel on the host. Otherwise, executes sequentially
///        on the host.
///
///        REDUCE is an alias to PARALLEL that indicates a reduction is taking place.
///
/// @arg[in] INDEX The index variable
/// @arg[in] START_INDEX The starting index (inclusive)
/// @arg[in] END_INDEX The ending index (exclusive)
/// @arg[in] CHUNK_SIZE Maximum kernel size
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_CHUNKED_REDUCE_LOOP(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE) CARE_CHECKED_CHUNKED_REDUCE_LOOP_START(INDEX, START_INDEX, END_INDEX, CHUNK_SIZE, care_reduce_loop_chunked_check)

#define CARE_CHUNKED_REDUCE_LOOP_END CARE_CHECKED_CHUNKED_REDUCE_LOOP_END(care_reduce_loop_chunked_check)

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
/// @brief Macros for updating/initializing managed_ptrs.
///        Will start and end a sequential loop of length one.
///        If GPU is available, also executes on the device.
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_MANAGED_PTR_UPDATE_KERNEL { CARE_CHECKED_MANAGED_PTR_UPDATE_KERNEL_START(care_managed_ptr_write_kernel_check)

#define CARE_MANAGED_PTR_UPDATE_KERNEL_END CARE_CHECKED_MANAGED_PTR_UPDATE_KERNEL_END(care_managed_ptr_write_kernel_check) }

////////////////////////////////////////////////////////////////////////////////
///
/// @brief Macros for launching a 2D kernel with fixed y dimension and varying x dimension
///        If GPU is available, executes on the device.
///
////////////////////////////////////////////////////////////////////////////////
#define CARE_LOOP_2D_STREAM_JAGGED(XINDEX, XSTART, XEND, XLENGTHS, YINDEX, YSTART, YLENGTH, FLAT_INDEX)  \
   launch_2D_jagged(care::gpu{}, XSTART, XEND, XLENGTHS.data(chai::DEFAULT, true), YSTART, YLENGTH, __FILE__, __LINE__, [=] CARE_DEVICE (int XINDEX, int YINDEX)->void  {
#define CARE_LOOP_2D_STREAM_JAGGED_END });

#define CARE_LOOP_2D_REDUCE_JAGGED(XINDEX, XSTART, XEND, XLENGTHS, YINDEX, YSTART, YLENGTH, FLAT_INDEX)  \
   launch_2D_jagged(care::gpu_reduce{}, XSTART, XEND, XLENGTHS.data(chai::DEFAULT, true), YSTART, YLENGTH, __FILE__, __LINE__, [=] CARE_DEVICE (int XINDEX, int YINDEX)->void  {
#define CARE_LOOP_2D_REDUCE_JAGGED_END });

#endif // !defined(_CARE_DEFAULT_MACROS_H_)

