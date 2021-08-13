
# Copyright 2013-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack import *

import socket


class Care(CachedCMakePackage, CudaPackage, ROCmPackage):
    """
    Algorithms for chai managed arrays.
    """

    homepage = "https://github.com/LLNL/CARE"
    git      = "https://github.com/LLNL/CARE.git"

    version('develop', branch='develop', submodules='True')
    version('main', branch='main', submodules='True')
    version('0.3.0', tag='v0.3.0', submodules='True')
    version('0.2.0', tag='v0.2.0', submodules='True')

    variant('openmp', default=False, description='Build Shared Libs')
    variant('implicit_conversions', default=True, description='Enable implicit'
            'conversions to/from raw pointers')
    variant('benchmarks', default=True, description='Build benchmarks.')
    variant('examples', default=True, description='Build examples.')
    variant('docs', default=False, description='Build documentation')
    variant('tests', default=False, description='Build tests')
    variant('loop_fuser', default=False, description='Enable loop fusion capability')
    variant('allow-unsupported-compilers', default=False, description="Allow untested combinations of cuda and host compilers.")

    depends_on('blt@0.4.0:', type='build', when='@0.3.1:')
    depends_on('blt@:0.3.6', type='build', when='@:0.3.0')

    depends_on('cmake@3.14.5', when="+cuda")

    depends_on('camp@0.1.0')
    depends_on('umpire@5.0.0')
    depends_on('raja@main')
    depends_on('chai+enable_pick@2.3.0')

    # WARNING: this package currently only supports an internal cub
    # package. This will cause a race condition if compiled with another
    # package that uses cub. TODO: have all packages point to the same external
    # cub package.
    depends_on('cub', when='+cuda')

    depends_on('camp+cuda', when='+cuda')
    depends_on('umpire+cuda~shared', when='+cuda')
    depends_on('raja+cuda~openmp', when='+cuda')
    depends_on('chai+cuda~shared~disable_rm', when='+cuda')
    depends_on('camp+allow-unsupported-compilers', when='+allow-unsupported-compilers')
    depends_on('umpire+allow-unsupported-compilers', when='+allow-unsupported-compilers')
    depends_on('raja+allow-unsupported-compilers', when='+allow-unsupported-compilers')
    depends_on('chai+allow-unsupported-compilers', when='+allow-unsupported-compilers')

    # variants +hip and amdgpu_targets are not automatically passed to
    # dependencies, so do it manually.
    depends_on('camp+hip', when='+hip')
    depends_on('umpire+hip', when='+hip')
    depends_on('raja+hip~openmp', when='+hip')
    depends_on('chai+hip~disable_rm', when='+hip')
    for val in ROCmPackage.amdgpu_targets:
        depends_on('camp amdgpu_target=%s' % val, when='amdgpu_target=%s' % val)
        depends_on('umpire amdgpu_target=%s' % val, when='amdgpu_target=%s' % val)
        depends_on('raja amdgpu_target=%s' % val, when='amdgpu_target=%s' % val)
        depends_on('chai amdgpu_target=%s' % val, when='amdgpu_target=%s' % val)

    conflicts('+openmp', when='+hip')
    conflicts('+openmp', when='+cuda')

    def flag_handler(self, name, flags):
        if self.spec.satisfies('%cce') and name == 'fflags':
            flags.append('-ef')

        if name in ('cflags', 'cxxflags', 'cppflags', 'fflags'):
            return (None, None, None)  # handled in the cmake cache
        return (flags, None, None)

    def _get_sys_type(self, spec):
        sys_type = spec.architecture
        # if on llnl systems, we can use the SYS_TYPE
        if "SYS_TYPE" in env:
            sys_type = env["SYS_TYPE"]
        return sys_type

    @property
    def cache_name(self):
        hostname = socket.gethostname()
        if "SYS_TYPE" in env:
            # Are we on a LLNL system then strip node number
            hostname = hostname.rstrip('1234567890')
        return "hc-{0}-{1}-{2}@{3}.cmake".format(
            hostname,
            self._get_sys_type(self.spec),
            self.spec.compiler.name,
            self.spec.compiler.version
        )

    def initconfig_compiler_entries(self):
        entries = super(Care, self).initconfig_compiler_entries()
        return entries

    def initconfig_hardware_entries(self):
        spec = self.spec
        entries = super(Care, self).initconfig_hardware_entries()

        if "+cuda" in spec:
            entries.append(cmake_cache_option("ENABLE_CUDA", True))
            entries.append(cmake_cache_option("CUDA_SEPARABLE_COMPILATION",
                                              True))
            entries.append(cmake_cache_string("CUDA_TOOLKIT_ROOT_DIR",spec['cuda'].prefix))
            entries.append(cmake_cache_string("NVTOOLSEXT_DIR",spec['cuda'].prefix))
            entries.append(cmake_cache_string("CUB_DIR",spec['cub'].prefix))

            # CUDA_FLAGS
            cudaflags  = "-restrict --expt-extended-lambda "

            if not spec.satisfies('cuda_arch=none'):
                cuda_arch = spec.variants['cuda_arch'].value[0]
                entries.append(cmake_cache_string(
                    "CMAKE_CUDA_ARCHITECTURES",
                    cuda_arch))
                cudaflags += '-arch compute_${CMAKE_CUDA_ARCHITECTURES} -code sm_${CMAKE_CUDA_ARCHITECTURES} '
            else:
                entries.append(
                    "# cuda_arch could not be determined\n\n")

            if "+cpp14" in spec:
                cudaflags += " -std=c++14"
            else:
                cudaflags += " -std=c++11"
            entries.append(
                cmake_cache_string("CMAKE_CUDA_FLAGS", cudaflags))

            entries.append(
                "# nvcc does not like gtest's 'pthreads' flag\n")
            entries.append(
                cmake_cache_option("gtest_disable_pthreads", True))

        if '+hip' in spec:
            entries.append(cmake_cache_option("ENABLE_HIP", True))
            entries.append(cmake_cache_string("HIP_ROOT_DIR", spec['hip'].prefix))
            
            archs = self.spec.variants['amdgpu_target'].value
            if archs != 'none':
                arch_str = ",".join(archs)
                entries.append(cmake_cache_string("HIP_HIPCC_FLAGS","--amdgpu-target={0}".format(arch_str)))

        entries.append("#------------------{0}".format("-" * 30))
        entries.append("# Hardware Specifics")
        entries.append("#------------------{0}\n".format("-" * 30))

        # OpenMP
        entries.append(cmake_cache_option("ENABLE_OPENMP", spec.satisfies('+openmp')))

        # Enable death tests
        entries.append(cmake_cache_option(
            "ENABLE_GTEST_DEATH_TESTS",
            not spec.satisfies('+cuda target=ppc64le:')
        ))

        return entries

    def initconfig_mpi_entries(self):
        entries = super(Care, self).initconfig_mpi_entries()
        return entries

    def initconfig_package_entries(self):
        spec = self.spec
        entries = []

        # TPL locations
        entries.append("#------------------{0}".format("-" * 60))
        entries.append("# TPLs")
        entries.append("#------------------{0}\n".format("-" * 60))
        
        entries.append(cmake_cache_path('BLT_SOURCE_DIR', spec['blt'].prefix))

        entries.append(cmake_cache_option('CARE_ENABLE_IMPLICIT_CONVERSIONS', spec.satisfies('+implicit_conversions')))
        entries.append(cmake_cache_option('CARE_ENABLE_LOOP_FUSER', spec.satisfies('+loop_fuser')))
        entries.append(cmake_cache_path('CAMP_DIR', spec['camp'].prefix))
        entries.append(cmake_cache_path('umpire_DIR', spec['umpire'].prefix.share.umpire.cmake))
        entries.append(cmake_cache_path('raja_DIR', spec['raja'].prefix.share.raja.cmake))
        entries.append(cmake_cache_path('chai_DIR', spec['chai'].prefix.share.chai.cmake))
        entries.append(cmake_cache_option('CARE_ENABLE_TESTS', spec.satisfies('+tests')))
        entries.append(cmake_cache_option('BLT_ENABLE_TESTS', spec.satisfies('+tests')))

        return entries
    

    def cmake_args(self):
        spec = self.spec
        from_variant = self.define_from_variant

        options = []

        # There are both CARE_ENABLE_* and ENABLE_* variables in here because
        # one controls the BLT infrastructure and the other controls the CARE
        # infrastructure. The goal is to just be able to use the CARE_ENABLE_*
        # variables, but CARE isn't set up correctly for that yet.
        options.append(from_variant('CARE_ENABLE_TESTS', 'tests'))
        options.append(from_variant('BLT_ENABLE_TESTS', 'tests'))

        options.append(from_variant('ENABLE_BENCHMARKS', 'benchmarks'))
        options.append(from_variant('CARE_ENABLE_BENCHMARKS', 'benchmarks'))

        options.append(from_variant('ENABLE_EXAMPLES', 'examples'))
        options.append(from_variant('CARE_ENABLE_EXAMPLES', 'examples'))

        options.append(from_variant('ENABLE_DOCS', 'docs'))
        options.append(from_variant('CARE_ENABLE_DOCS', 'docs'))

        return options
