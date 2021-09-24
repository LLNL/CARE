# module load cmake/3.21.1
# module load rocm/4.4.0
# module load gcc/10.3.0

set(ENABLE_HIP ON CACHE BOOL "Enable HIP build")
set(HIP_ROOT_DIR "/opt/rocm-4.4.0/hip" CACHE PATH "Path to HIP root directory")
set(BLT_ROCM_ARCH "gfx908" CACHE STRING "gfx architecture to use when generating ROCm code")
set(CMAKE_HIP_ARCHITECTURES "gfx908" CACHE STRING "HIP architectures")

set(ENABLE_RAJA_PLUGIN ON CACHE BOOL "Enable the RAJA plugin in CHAI")
set(RAJA_ENABLE_HIP_INDIRECT_FUNCTION_CALL ON CACHE BOOL "Enable use of device function pointers in hip backend")
set(CARE_ENABLE_LOOP_FUSER ON CACHE BOOL "Enable the loop fuser")

set(CHAI_ENABLE_MANAGED_PTR ON CACHE BOOL "Enable aliases, tests, and reproducer for managed_ptr")
set(CARE_ENABLE_MANAGED_PTR ON CACHE BOOL "Enable aliases, tests, and reproducer for managed_ptr")

set(COMPILER_BASE "/opt/cray/pe/gcc/10.3.0/bin" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/g++" CACHE PATH "")
