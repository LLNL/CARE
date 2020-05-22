//////////////////////////////////////////////////////////////////////////////////////
// Copyright 2020 Lawrence Livermore National Security, LLC and other CARE developers.
// See the top-level LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_POLICIES_H_
#define _CARE_POLICIES_H_

namespace care {
   struct sequential {};
   struct openmp {};
   struct gpu {};
   struct parallel {};
   struct raja_fusible {};
   struct raja_fusible_seq {};
   struct raja_chai_everywhere {};
   struct gpu_simulation {};
} // namespace care

#endif // !defined(_CARE_POLICIES_H_)

