set(ENABLE_HIP ON CACHE BOOL "Enable HIP build")
set(HIP_ROOT_DIR "/opt/rocm-4.2.0/hip" CACHE PATH "Path to HIP root directory")

# Function pointers are not supported in HIP device code
set(CARE_ENABLE_LOOP_FUSER OFF CACHE STRING "Enable the loop fuser")

# Virtual functions are not supported in HIP device code
set(CHAI_ENABLE_MANAGED_PTR OFF CACHE BOOL "Enable aliases, tests, and reproducer for managed_ptr")
set(CARE_ENABLE_MANAGED_PTR OFF CACHE BOOL "Enable aliases, tests, and reproducer for managed_ptr")

set(COMPILER_BASE "/opt/gcc/10.3.0/bin" CACHE PATH "")
set(CMAKE_C_COMPILER "${COMPILER_BASE}/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "${COMPILER_BASE}/g++" CACHE PATH "")
