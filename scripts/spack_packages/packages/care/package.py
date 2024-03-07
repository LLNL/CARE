# Copyright 2013-2023 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

import socket

from spack.package import *

from .camp import blt_link_helpers
from .camp import cuda_for_radiuss_projects
from .camp import hip_for_radiuss_projects


class Care(CachedCMakePackage, CudaPackage, ROCmPackage):
    """
    CHAI and RAJA extensions (includes data structures and algorithms).
    """

    homepage = "https://github.com/LLNL/CARE"
    git      = "https://github.com/LLNL/CARE.git"

    version('develop', branch='develop', submodules='True')
    version('master', branch='master', submodules='True')
    version('0.10.0', tag='v0.10.0', submodules='True')
    version('0.3.0', tag='v0.3.0', submodules='True')
    version('0.2.0', tag='v0.2.0', submodules='True')

    variant('openmp', default=False, description='Enable openmp')
    variant("mpi", default=False, description="Enable MPI support")

    variant('tests', default=False, description='Build tests')
    variant('benchmarks', default=False, description='Build benchmarks.')
    variant('examples', default=False, description='Build examples.')
    variant('docs', default=False, description='Build documentation')

    variant('implicit_conversions', default=False, description='Enable implicit conversions to/from raw pointers')
    variant('loop_fuser', default=False, description='Enable loop fusion capability')

    depends_on('cmake@3.14.5:', type='build')
    depends_on('cmake@3.21:', when='+rocm', type='build')

    depends_on('blt@0.5.2:', type='build', when='@0.10.0:')
    depends_on('blt@0.4.1:', type='build', when='@0.3.1:')
    depends_on('blt@:0.3.6', type='build', when='@:0.3.0')

    depends_on('umpire~c~shared~werror@2022.10.0:', when='@0.10.0:')
    depends_on('raja~shared~vectorization~examples~exercises@2022.10.5:', when='@0.10.0:')
    depends_on('chai~shared+raja~examples+enable_pick@2022.10.0:', when='@0.10.0:')

    depends_on('umpire+mpi', when='+mpi')

    with when('+openmp'):
        depends_on('umpire+openmp')
        depends_on('raja+openmp')
        depends_on('chai+openmp')

    with when('+cuda'):
        # WARNING: this package currently only supports an internal cub
        # package. This will cause a race condition if compiled with another
        # package that uses cub. TODO: have all packages point to the same external
        # cub package.
        depends_on('cub')

        depends_on('umpire+cuda')
        depends_on('raja+cuda')
        depends_on('chai+cuda')

        for sm_ in CudaPackage.cuda_arch_values:
            depends_on('umpire+cuda cuda_arch={0}'.format(sm_), when='cuda_arch={0}'.format(sm_))
            depends_on('raja+cuda cuda_arch={0}'.format(sm_), when='cuda_arch={0}'.format(sm_))
            depends_on('chai+cuda cuda_arch={0}'.format(sm_), when='cuda_arch={0}'.format(sm_))

    with when('+rocm'):
        depends_on('umpire+rocm')
        depends_on('raja+rocm')
        depends_on('chai+rocm')

        for arch_ in ROCmPackage.amdgpu_targets:
            depends_on('umpire+rocm amdgpu_target={0}'.format(arch_), when='amdgpu_target={0}'.format(arch_))
            depends_on('raja+rocm amdgpu_target={0}'.format(arch_), when='amdgpu_target={0}'.format(arch_))
            depends_on('chai+rocm amdgpu_target={0}'.format(arch_), when='amdgpu_target={0}'.format(arch_))


    def _get_sys_type(self, spec):
        sys_type = spec.architecture

        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type


    @property
    def cache_name(self):
        hostname = socket.gethostname()

        if "SYS_TYPE" in env:
            hostname = hostname.rstrip("1234567890")
        return "{0}-{1}-{2}@{3}-{4}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version,
            self.spec.dag_hash(8)
        )


    def initconfig_compiler_entries(self):
        spec = self.spec
        compiler = self.compiler
        # Default entries are already defined in CachedCMakePackage, inherit them:
        entries = super(Care, self).initconfig_compiler_entries()

        #### BEGIN: Override CachedCMakePackage CMAKE_C_FLAGS and CMAKE_CXX_FLAGS
        flags = spec.compiler_flags

        # use global spack compiler flags
        cppflags = " ".join(flags["cppflags"])

        if cppflags:
            # avoid always ending up with " " with no flags defined
            cppflags += " "

        cflags = cppflags + " ".join(flags["cflags"])

        if cflags:
            entries.append(cmake_cache_string("CMAKE_C_FLAGS", cflags))

        cxxflags = cppflags + " ".join(flags["cxxflags"])

        if cxxflags:
            entries.append(cmake_cache_string("CMAKE_CXX_FLAGS", cxxflags))

        fflags = " ".join(flags["fflags"])

        if fflags:
            entries.append(cmake_cache_string("CMAKE_Fortran_FLAGS", fflags))

        #### END: Override CachedCMakePackage CMAKE_C_FLAGS and CMAKE_CXX_FLAGS

        blt_link_helpers(entries, spec, compiler)

        return entries


    def initconfig_hardware_entries(self):
        spec = self.spec
        compiler = self.compiler
        entries = super(Care, self).initconfig_hardware_entries()

        entries.append(cmake_cache_option("ENABLE_OPENMP", '+openmp' in spec))

        if '+cuda' in spec:
            entries.append(cmake_cache_option("ENABLE_CUDA", True))
            entries.append(cmake_cache_option("CUDA_SEPARABLE_COMPILATION", True))
            entries.append(cmake_cache_string("CUDA_TOOLKIT_ROOT_DIR", spec['cuda'].prefix))
            entries.append(cmake_cache_string("CUB_DIR", spec['cub'].prefix))

            cuda_for_radiuss_projects(entries, spec)
        else:
            entries.append(cmake_cache_option("ENABLE_CUDA", False))

        if '+rocm' in spec:
            entries.append(cmake_cache_option("ENABLE_HIP", True))
            entries.append(cmake_cache_string("HIP_ROOT_DIR", spec['hip'].prefix))

            hip_for_radiuss_projects(entries, spec, compiler)
        else:
            entries.append(cmake_cache_option("ENABLE_HIP", False))

        return entries


    def initconfig_mpi_entries(self):
        spec = self.spec

        entries = super(Care, self).initconfig_mpi_entries()
        entries.append(cmake_cache_option("ENABLE_MPI", "+mpi" in spec))

        return entries


    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))
        
        entries.append(cmake_cache_path('BLT_SOURCE_DIR', spec['blt'].prefix))
        entries.append(cmake_cache_path('UMPIRE_DIR', spec['umpire'].prefix))
        entries.append(cmake_cache_path('RAJA_DIR', spec['raja'].prefix))
        entries.append(cmake_cache_path('CHAI_DIR', spec['chai'].prefix))

        # Build options
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# Build Options")
        entries.append("#------------------{0}\n".format("-" * 60))

        entries.append(cmake_cache_string("CMAKE_BUILD_TYPE", spec.variants["build_type"].value))

        entries.append(cmake_cache_option('ENABLE_TESTS', '+tests' in spec))
        entries.append(cmake_cache_option('CARE_ENABLE_TESTS', '+tests' in spec))

        entries.append(cmake_cache_option('ENABLE_BENCHMARKS', '+benchmarks' in spec))
        entries.append(cmake_cache_option('CARE_ENABLE_BENCHMARKS', '+benchmarks' in spec))

        entries.append(cmake_cache_option('ENABLE_EXAMPLES', '+examples' in spec))
        entries.append(cmake_cache_option('CARE_ENABLE_EXAMPLES', '+examples' in spec))

        entries.append(cmake_cache_option('ENABLE_DOCS', '+docs' in spec))
        entries.append(cmake_cache_option('CARE_ENABLE_DOCS', '+docs' in spec))

        entries.append(cmake_cache_option('CARE_ENABLE_IMPLICIT_CONVERSIONS', '+implicit_conversions' in spec))
        entries.append(cmake_cache_option('CARE_ENABLE_LOOP_FUSER', '+loop_fuser' in spec))

        return entries
    

    def cmake_args(self):
        options = []
        return options

