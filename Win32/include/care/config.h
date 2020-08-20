//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#if !defined(_CARE_CONFIG_H_)
#define _CARE_CONFIG_H_

/* #undef CARE_LOOP_VVERBOSE_ENABLED */
#define CARE_HAVE_LOOP_FUSER 1
/* #undef CARE_DEBUG */
#define CARE_ENABLE_GPU_SIMULATION_MODE 0
#define CARE_ENABLE_IMPLICIT_CONVERSIONS

#define CARE_HAVE_LLNL_GLOBALID 1

#if defined(_WIN32) && !defined(CARESTATICLIB)
#if defined(CARE_EXPORTS)
#define CARE_DLL_API __declspec(dllexport)
#else // defined(CARE_EXPORTS)
#define CARE_DLL_API __declspec(dllimport)
#endif // defined(CARE_EXPORTS)
#else // defined(_WIN32) && !defined(CARESTATICLIB)
#define CARE_DLL_API
#endif // defined(_WIN32) && !defined(CARESTATICLIB)

#if defined(__CUDACC__)
#define CARE_HOST_DEVICE __host__ __device__
#define CARE_DEVICE __device__
#define CARE_HOST __host__
#define CARE_GLOBAL __global__
#else // defined(__CUDACC__)
#define CARE_HOST_DEVICE
#define CARE_DEVICE
#define CARE_HOST
#define CARE_GLOBAL
#endif // defined(__CUDACC__)

#endif // !defined(_CARE_CONFIG_H_)

