#!/usr/bin/env bash

##############################################################################
# Copyright (c) 2020-25, Lawrence Livermore National Security, LLC and CARE
# project contributors. See the CARE LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
##############################################################################

#=============================================================================
# Change the copyright date in all files that contain the text
# "the CARE LICENSE file", which is part of the copyright statement at the
# top of each CARE file. We use this to distinguish CARE files from ones
# that we do not own (e.g., other repos included as submodules), which we
# do not want to modify. Note that this file and *.git files are omitted
# as well.
#
# IMPORTANT: Since this file is not modified (it is running the shell 
# script commands), you must EDIT THE COPYRIGHT DATES IN THE HEADER ABOVE 
# MANUALLY.
#
# Edit the 'grep' command below to change the set of files that will be
# modified.
#
# Change the 'sed' command below to change the content that is changed
# in each file and what it is changed to. Typically, this would only be
# modifying the end year in the copyright date.
#
#=============================================================================
#
# If you need to modify this script, you may want to run each of these 
# commands individually from the command line to make sure things are doing 
# what you think they should be doing. This is why they are separated into 
# steps here.
# 
#=============================================================================

#=============================================================================
# First find all the files we want to modify
#=============================================================================
grep -rl "the CARE LICENSE file" . --exclude-dir=.git --exclude-dir=blt --exclude-dir=cub --exclude-dir=umpire --exclude-dir=raja --exclude-dir=chai --exclude-dir=radiuss-spack-configs --exclude-dir=uberenv --exclude-dir=build --exclude=update-copyright.sh > files2change

#=============================================================================
# Replace the old copyright dates with new dates
#=============================================================================
for i in `cat files2change`
do
    echo $i
    cp $i $i.sed.bak
    # For when there is only a single year in the copyright
    # sed "s/Copyright (c) \([0-9]\{4\}\),/Copyright (c) 2020-25,/" $i.sed.bak > $i
    sed "s/Copyright (c) \([0-9]\{4\}\)-[0-9]\{2\},/Copyright (c) 2020-25,/" $i.sed.bak > $i
done

echo LICENSE
cp LICENSE LICENSE.sed.bak
sed "s/Copyright (c) \([0-9]\{4\}\)-[0-9]\{4\}/Copyright (c) 2020-2025/" LICENSE.sed.bak > LICENSE

for i in README.md
do 
    echo $i
    cp $i $i.sed.bak
    sed "s/\([0-9]\{4\}\)-[0-9]\{2\}/2020-25/" $i.sed.bak > $i
done

#=============================================================================
# Remove temporary files created in the process
#=============================================================================
find . -name \*.sed.bak -exec rm {} \;
rm files2change
