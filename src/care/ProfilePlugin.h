//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
// project contributors. See the CARE LICENSE file for details.
//
// SPDX-License-Identifier: BSD-3-Clause
//////////////////////////////////////////////////////////////////////////////

#ifndef _CARE_ProfilePlugin_H_
#define _CARE_ProfilePlugin_H_

#include "care/config.h"
#include "RAJA/util/PluginStrategy.hpp"

namespace care{

   class ProfilePlugin : public RAJA::util::PluginStrategy
   {
      public:
         ProfilePlugin() = default;

         CARE_DLL_API static void registerPlugin();
						
         void preLaunch(const RAJA::util::PluginContext& p) override;

         void postLaunch(const RAJA::util::PluginContext& p) override;

      private:
         static unsigned int s_current_color;

         static uint32_t s_colors[7];
         
         static int s_num_colors;
         
         static bool s_profile_host_loops;     
   };
}

#endif
