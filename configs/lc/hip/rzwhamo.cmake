set(ENABLE_HIP ON CACHE BOOL "Enable HIP build")
set(HIP_ROOT_DIR "/opt/rocm-4.4.0/hip" CACHE PATH "Path to HIP root directory")

set(ENABLE_RAJA_PLUGIN ON CACHE BOOL "Enable the RAJA plugin in CHAI")
set(CARE_ENABLE_LOOP_FUSER ON CACHE BOOL "Enable the loop fuser")

set(CHAI_ENABLE_MANAGED_PTR ON CACHE BOOL "Enable aliases, tests, and reproducer for managed_ptr")
set(CARE_ENABLE_MANAGED_PTR ON CACHE BOOL "Enable aliases, tests, and reproducer for managed_ptr")

set(COMPILER_BASE "/opt/cray/pe/gcc/10.3.0/bin" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/g++" CACHE PATH "")
