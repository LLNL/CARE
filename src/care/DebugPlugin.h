//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_DebugPlugin_H_
#define _CARE_DebugPlugin_H_

#include "care/config.h"
#include "RAJA/util/PluginStrategy.hpp"
#include "chai/ExecutionSpaces.hpp"

namespace care{

   class DebugPlugin : public RAJA::util::PluginStrategy
   {
      public:
         DebugPlugin() = default;

         CARE_DLL_API static void registerPlugin();
						
         void preLaunch(const RAJA::util::PluginContext& p) override;

         void postLaunch(const RAJA::util::PluginContext& p) override;
         
         /////////////////////////////////////////////////////////////////////////////////
         ///
         /// @brief Writes out debugging information after a loop is executed.
         ///
         /// @arg[in] space The execution space
         /// @arg[in] fileName The file where the loop macro was called
         /// @arg[in] lineNumber The line number where the loop macro was called
         ///
         /////////////////////////////////////////////////////////////////////////////////
         static void writeLoopData(chai::ExecutionSpace space, const char * fileName, int lineNumber);
   };
}

#endif
