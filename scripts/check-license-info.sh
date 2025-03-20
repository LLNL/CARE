#!/usr/bin/env zsh
##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

# This is used for the ~*tpl* line to ignore files in bundled tpls
setopt extended_glob

autoload colors

RED="\033[1;31m"
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

files_no_license=$(grep -rL "Copyright.*2020-25, Lawrence Livermore National Security, LLC and CARE" . \
   --exclude-dir=.git \
   --exclude-dir=blt \
   --exclude-dir=cub \
   --exclude-dir=umpire \
   --exclude-dir=raja \
   --exclude-dir=chai \
   --exclude-dir=radiuss-spack-configs \
   --exclude-dir=uberenv \
   --exclude-dir=build \
   --exclude=\*.swp \
   --exclude=ci.yml \
   --exclude=.gitignore \
   --exclude=.gitmodules \
   --exclude=LICENSE \
   --exclude=COPYRIGHT \
   --exclude=NOTICE \
   --exclude=\*.natvis \
   --exclude=repo.yaml \
   --exclude=package.py \
   --exclude=.uberenv_config.json)

files_no_license+=$(grep -rL "project contributors. See the CARE LICENSE file for details." . \
   --exclude-dir=.git \
   --exclude-dir=blt \
   --exclude-dir=cub \
   --exclude-dir=umpire \
   --exclude-dir=raja \
   --exclude-dir=chai \
   --exclude-dir=radiuss-spack-configs \
   --exclude-dir=uberenv \
   --exclude-dir=build \
   --exclude=\*.swp \
   --exclude=ci.yml \
   --exclude=.gitignore \
   --exclude=.gitmodules \
   --exclude=LICENSE \
   --exclude=COPYRIGHT \
   --exclude=NOTICE \
   --exclude=\*.natvis \
   --exclude=repo.yaml \
   --exclude=package.py \
   --exclude=.uberenv_config.json)

files_no_license+=$(grep -rL "SPDX-License-Identifier: BSD-3-Clause" . \
   --exclude-dir=.git \
   --exclude-dir=blt \
   --exclude-dir=cub \
   --exclude-dir=umpire \
   --exclude-dir=raja \
   --exclude-dir=chai \
   --exclude-dir=radiuss-spack-configs \
   --exclude-dir=uberenv \
   --exclude-dir=build \
   --exclude=\*.swp \
   --exclude=ci.yml \
   --exclude=.gitignore \
   --exclude=.gitmodules \
   --exclude=LICENSE \
   --exclude=COPYRIGHT \
   --exclude=NOTICE \
   --exclude=\*.natvis \
   --exclude=repo.yaml \
   --exclude=package.py \
   --exclude=.uberenv_config.json)

if [ $files_no_license ]; then
  print "${RED} [!] Some files are missing license text:${NOCOLOR}"
  echo "${files_no_license}"
  exit 255
else
  print "${GREEN} [Ok] All files have required license info.${NOCOLOR}"
  exit 0
fi
