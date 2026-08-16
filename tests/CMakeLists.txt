# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(self_amdgpu_runtime_api_test runtime_test.c)
target_link_libraries(self_amdgpu_runtime_api_test PRIVATE SelfAmdgpuRuntime::runtime)
add_test(NAME self_amdgpu_runtime.api COMMAND self_amdgpu_runtime_api_test)

add_executable(
  self_amdgpu_runtime_smi_registry_test
  smi_registry_test.c
  ../src/smi_registry.c
  ../src/sha256.c)
target_compile_features(self_amdgpu_runtime_smi_registry_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_smi_registry_test PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_smi_registry_test PRIVATE Threads::Threads)
if(MSVC)
  target_compile_options(self_amdgpu_runtime_smi_registry_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_smi_registry_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.smi_registry
  COMMAND self_amdgpu_runtime_smi_registry_test)

add_executable(self_amdgpu_runtime_ccl_v1_test ccl_v1_test.c)
target_compile_features(self_amdgpu_runtime_ccl_v1_test PRIVATE c_std_11)
target_link_libraries(
  self_amdgpu_runtime_ccl_v1_test PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(self_amdgpu_runtime_ccl_v1_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_ccl_v1_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.ccl_v1
  COMMAND self_amdgpu_runtime_ccl_v1_test)

add_executable(
  self_amdgpu_runtime_ccl_carrier_v1_test ccl_carrier_v1_test.c)
target_compile_features(
  self_amdgpu_runtime_ccl_carrier_v1_test PRIVATE c_std_11)
target_link_libraries(
  self_amdgpu_runtime_ccl_carrier_v1_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(
    self_amdgpu_runtime_ccl_carrier_v1_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_ccl_carrier_v1_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.ccl_carrier_v1
  COMMAND self_amdgpu_runtime_ccl_carrier_v1_test)

add_executable(
  self_amdgpu_runtime_ccl_live_v1_test ccl_live_v1_test.c)
target_compile_features(
  self_amdgpu_runtime_ccl_live_v1_test PRIVATE c_std_11)
target_link_libraries(
  self_amdgpu_runtime_ccl_live_v1_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(
    self_amdgpu_runtime_ccl_live_v1_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_ccl_live_v1_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.ccl_live_v1
  COMMAND self_amdgpu_runtime_ccl_live_v1_test)

# Standalone multi-process integration gate for the live broker FD table, the
# CCL v1 ring planner, and carrier sessions. Its FP32 descriptor carries
# rank-coded uint32 bytes through an explicitly host-only mock reduction; this
# validates ordering/transport/lifecycle and is not device or gem5 acceptance.
add_executable(
  self_amdgpu_runtime_ccl_live_collective_v1_test
  ccl_live_collective_v1_test.c
  ../src/sha256.c)
target_compile_features(
  self_amdgpu_runtime_ccl_live_collective_v1_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_ccl_live_collective_v1_test PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_ccl_live_collective_v1_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(
    self_amdgpu_runtime_ccl_live_collective_v1_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_ccl_live_collective_v1_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.ccl_live_collective_v1
  COMMAND self_amdgpu_runtime_ccl_live_collective_v1_test)
set_tests_properties(
  self_amdgpu_runtime.ccl_live_collective_v1 PROPERTIES TIMEOUT 60)

# Keep a tiny public-header-only consumer as the extraction gate for CP-0014.
# This target deliberately has no simulator, ROCr, or device-library linkage;
# the companion CMake audit checks that property on the produced executable.
add_executable(
  self_amdgpu_runtime_host_native_boundary_test
  host_native_boundary_test.c)
target_compile_features(
  self_amdgpu_runtime_host_native_boundary_test
  PRIVATE c_std_11)
target_link_libraries(
  self_amdgpu_runtime_host_native_boundary_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(
    self_amdgpu_runtime_host_native_boundary_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_host_native_boundary_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.host_native.boundary
  COMMAND
    "${CMAKE_COMMAND}"
    "-DSAGR_HOST_NATIVE_BINARY=$<TARGET_FILE:self_amdgpu_runtime_host_native_boundary_test>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/host_native_boundary_audit.cmake")

add_executable(self_amdgpu_runtime_provider_test provider_test.c)
target_link_libraries(
  self_amdgpu_runtime_provider_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(self_amdgpu_runtime_provider_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_provider_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.provider
  COMMAND self_amdgpu_runtime_provider_test)
add_test(
  NAME self_amdgpu_runtime.provider.audit
  COMMAND
    "${CMAKE_COMMAND}"
    "-DSAGR_PROVIDER_SOURCE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/../src/provider.c"
    "-DSAGR_PROVIDER_HEADER_FILE=${CMAKE_CURRENT_SOURCE_DIR}/../include/self_amdgpu_runtime/provider.h"
    "-DSAGR_KMT_SOURCE_FILE=${CMAKE_CURRENT_SOURCE_DIR}/../src/kmt_shim.c"
    "-DSAGR_KMT_HEADER_FILE=${CMAKE_CURRENT_SOURCE_DIR}/../include/self_amdgpu_runtime/kmt_shim.h"
    "-DSAGR_PROVIDER_BINARY=$<TARGET_FILE:self_amdgpu_runtime_provider_test>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/provider_audit.cmake")

if(SELF_AMDGPU_RUNTIME_BUILD_HSAKMT_MODEL)
  add_executable(
    self_amdgpu_runtime_hsakmt_model_test
    hsakmt_model_test.c)
  add_dependencies(
    self_amdgpu_runtime_hsakmt_model_test
    self_amdgpu_hsakmt_model)
  target_compile_features(
    self_amdgpu_runtime_hsakmt_model_test PRIVATE c_std_11)
  target_include_directories(
    self_amdgpu_runtime_hsakmt_model_test
    SYSTEM PRIVATE "${SELF_AMDGPU_RUNTIME_HSAKMT_INCLUDE_DIR}")
  target_compile_definitions(
    self_amdgpu_runtime_hsakmt_model_test
    PRIVATE
      SAGR_HSAKMT_MODEL_PATH="$<TARGET_FILE:self_amdgpu_hsakmt_model>")
  target_link_libraries(
    self_amdgpu_runtime_hsakmt_model_test
    PRIVATE ${CMAKE_DL_LIBS})
  if(MSVC)
    target_compile_options(
      self_amdgpu_runtime_hsakmt_model_test PRIVATE /W4 /WX)
  else()
    target_compile_options(
      self_amdgpu_runtime_hsakmt_model_test
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
  add_test(
    NAME self_amdgpu_runtime.hsakmt_model
    COMMAND self_amdgpu_runtime_hsakmt_model_test)
  add_test(
    NAME self_amdgpu_runtime.hsakmt_model.audit
    COMMAND
      "${CMAKE_COMMAND}"
      "-DSAGR_HSAKMT_MODEL=$<TARGET_FILE:self_amdgpu_hsakmt_model>"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/hsakmt_model_audit.cmake")
endif()

find_package(Threads REQUIRED)
add_executable(
  self_amdgpu_runtime_kmt_shim_test
  kmt_shim_test.c
  ../src/transport_codec.c)
target_compile_features(self_amdgpu_runtime_kmt_shim_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_kmt_shim_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_kmt_shim_test
  PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
if(SELF_AMDGPU_RUNTIME_BUILD_HSAKMT_MODEL)
  add_dependencies(
    self_amdgpu_runtime_kmt_shim_test
    self_amdgpu_hsakmt_model)
  target_include_directories(
    self_amdgpu_runtime_kmt_shim_test
    SYSTEM PRIVATE "${SELF_AMDGPU_RUNTIME_HSAKMT_INCLUDE_DIR}")
  target_compile_definitions(
    self_amdgpu_runtime_kmt_shim_test
    PRIVATE
      SAGR_HSAKMT_MODEL_PATH="$<TARGET_FILE:self_amdgpu_hsakmt_model>")
  target_link_libraries(
    self_amdgpu_runtime_kmt_shim_test
    PRIVATE ${CMAKE_DL_LIBS})

  if(SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_LIBRARY OR
     SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_DEPENDENCY_PREFIX)
    if(NOT EXISTS "${SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_LIBRARY}" OR
       NOT IS_DIRECTORY
           "${SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_DEPENDENCY_PREFIX}")
      message(
        FATAL_ERROR
          "Both upstream libhsakmt archive and dependency prefix are required")
    endif()
    set(
      _hsakmt_dependencies
      "${SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_DEPENDENCY_PREFIX}/lib/libdrm_amdgpu.so"
      "${SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_DEPENDENCY_PREFIX}/lib/libdrm.so"
      "${SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_DEPENDENCY_PREFIX}/lib/libnuma.so"
      "${SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_DEPENDENCY_PREFIX}/lib/libpciaccess.so")
    foreach(_dependency IN LISTS _hsakmt_dependencies)
      if(NOT EXISTS "${_dependency}")
        message(FATAL_ERROR "Upstream libhsakmt dependency is absent: ${_dependency}")
      endif()
    endforeach()
    add_library(self_amdgpu_upstream_hsakmt STATIC IMPORTED)
    set_target_properties(
      self_amdgpu_upstream_hsakmt
      PROPERTIES IMPORTED_LOCATION
                 "${SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_LIBRARY}")
    set(_topology_tool
        "${CMAKE_CURRENT_SOURCE_DIR}/../tools/hsakmt-model-topology.py")
    file(SHA256 "${_topology_tool}" _topology_tool_sha256)
    set(_topology_directory
        "${CMAKE_CURRENT_BINARY_DIR}/hsakmt-model-topology-${_topology_tool_sha256}")
    set(_topology_stamp
        "${CMAKE_CURRENT_BINARY_DIR}/hsakmt-model-topology-${_topology_tool_sha256}.stamp")
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_custom_command(
      OUTPUT "${_topology_stamp}"
      COMMAND
        "${Python3_EXECUTABLE}" "${_topology_tool}"
        --output-dir "${_topology_directory}" --gpu-count 1
      COMMAND "${CMAKE_COMMAND}" -E touch "${_topology_stamp}"
      DEPENDS "${_topology_tool}"
      VERBATIM)
    add_custom_target(
      self_amdgpu_runtime_hsakmt_topology
      DEPENDS "${_topology_stamp}")
    add_executable(
      self_amdgpu_runtime_upstream_hsakmt_model_test
      hsakmt_upstream_model_test.c)
    target_compile_features(
      self_amdgpu_runtime_upstream_hsakmt_model_test PRIVATE c_std_11)
    target_include_directories(
      self_amdgpu_runtime_upstream_hsakmt_model_test
      SYSTEM PRIVATE "${SELF_AMDGPU_RUNTIME_HSAKMT_INCLUDE_DIR}")
    target_link_libraries(
      self_amdgpu_runtime_upstream_hsakmt_model_test
      PRIVATE
        self_amdgpu_upstream_hsakmt
        ${_hsakmt_dependencies}
        ${CMAKE_DL_LIBS}
        Threads::Threads
        rt)
    if(MSVC)
      target_compile_options(
        self_amdgpu_runtime_upstream_hsakmt_model_test PRIVATE /W4 /WX)
    else()
      target_compile_options(
        self_amdgpu_runtime_upstream_hsakmt_model_test
        PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
    endif()
    add_dependencies(
      self_amdgpu_runtime_kmt_shim_test
      self_amdgpu_runtime_hsakmt_topology
      self_amdgpu_runtime_upstream_hsakmt_model_test)
    target_compile_definitions(
      self_amdgpu_runtime_kmt_shim_test
      PRIVATE
        SAGR_UPSTREAM_HSAKMT_TEST=1
        SAGR_UPSTREAM_HSAKMT_TEST_PATH="$<TARGET_FILE:self_amdgpu_runtime_upstream_hsakmt_model_test>"
        SAGR_HSAKMT_TOPOLOGY_PATH="${_topology_directory}")
  endif()
endif()

if(SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_LIBRARY OR
   SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_INCLUDE_DIR)
  if(NOT EXISTS "${SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_LIBRARY}" OR
     NOT IS_DIRECTORY "${SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_INCLUDE_DIR}")
    message(
      FATAL_ERROR
        "Both upstream ROCr library and include directory are required")
  endif()
  if(NOT SELF_AMDGPU_RUNTIME_BUILD_HSAKMT_MODEL OR
     NOT DEFINED _topology_directory)
    message(
      FATAL_ERROR
        "The upstream ROCr smoke requires the HSAKMT model and topology")
  endif()
  add_executable(
    self_amdgpu_runtime_upstream_rocr_model_test
    rocr_upstream_model_test.c)
  target_compile_features(
    self_amdgpu_runtime_upstream_rocr_model_test PRIVATE c_std_11)
  target_include_directories(
    self_amdgpu_runtime_upstream_rocr_model_test
    PRIVATE ../include
    SYSTEM PRIVATE "${SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_INCLUDE_DIR}")
  target_compile_definitions(
    self_amdgpu_runtime_upstream_rocr_model_test
    PRIVATE
      SAGR_HSAKMT_MODEL_PATH="$<TARGET_FILE:self_amdgpu_hsakmt_model>")
  target_link_libraries(
    self_amdgpu_runtime_upstream_rocr_model_test
    PRIVATE "${SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_LIBRARY}" ${CMAKE_DL_LIBS})
  get_filename_component(_upstream_rocr_library_dir
                         "${SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_LIBRARY}" DIRECTORY)
  set_target_properties(
    self_amdgpu_runtime_upstream_rocr_model_test
    PROPERTIES BUILD_RPATH
               "$<TARGET_FILE_DIR:self_amdgpu_hsakmt_model>;${_upstream_rocr_library_dir}"
               INSTALL_RPATH "\$ORIGIN")
  if(MSVC)
    target_compile_options(
      self_amdgpu_runtime_upstream_rocr_model_test PRIVATE /W4 /WX)
  else()
    target_compile_options(
      self_amdgpu_runtime_upstream_rocr_model_test
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
  add_dependencies(
    self_amdgpu_runtime_kmt_shim_test
    self_amdgpu_runtime_upstream_rocr_model_test)
  target_compile_definitions(
    self_amdgpu_runtime_kmt_shim_test
    PRIVATE
      SAGR_UPSTREAM_ROCR_TEST=1
      SAGR_UPSTREAM_ROCR_TEST_PATH="$<TARGET_FILE:self_amdgpu_runtime_upstream_rocr_model_test>")

  # Standalone runtime-API probe for the explicitly enabled host-accessible
  # copy path. Keep it out of the default CTest set so merely building tests
  # never starts a model-backed HSA runtime.
  add_executable(
    self_amdgpu_runtime_upstream_rocr_fastcopy_probe
    fastcopy_probe.c)
  target_compile_features(
    self_amdgpu_runtime_upstream_rocr_fastcopy_probe PRIVATE c_std_11)
  target_include_directories(
    self_amdgpu_runtime_upstream_rocr_fastcopy_probe
    SYSTEM PRIVATE "${SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_INCLUDE_DIR}")
  target_link_libraries(
    self_amdgpu_runtime_upstream_rocr_fastcopy_probe
    PRIVATE "${SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_LIBRARY}" ${CMAKE_DL_LIBS})
  set_target_properties(
    self_amdgpu_runtime_upstream_rocr_fastcopy_probe
    PROPERTIES BUILD_RPATH
               "$<TARGET_FILE_DIR:self_amdgpu_hsakmt_model>;${_upstream_rocr_library_dir}")
  if(MSVC)
    target_compile_options(
      self_amdgpu_runtime_upstream_rocr_fastcopy_probe PRIVATE /W4 /WX)
  else()
    target_compile_options(
      self_amdgpu_runtime_upstream_rocr_fastcopy_probe
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
endif()
if(MSVC)
  target_compile_options(self_amdgpu_runtime_kmt_shim_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_kmt_shim_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.kmt_shim
  COMMAND self_amdgpu_runtime_kmt_shim_test)
set_tests_properties(self_amdgpu_runtime.kmt_shim PROPERTIES TIMEOUT 30)

add_executable(
  self_amdgpu_runtime_code_object_test
  code_object_test.c)
target_compile_features(self_amdgpu_runtime_code_object_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_code_object_test
  PRIVATE ../include)
target_link_libraries(
  self_amdgpu_runtime_code_object_test
  PRIVATE SelfAmdgpuRuntime::runtime)
target_compile_definitions(
  self_amdgpu_runtime_code_object_test
  PRIVATE
    SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH="${SELF_AMDGPU_RUNTIME_REPO_ROOT}/projects/rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/gpuReadWrite_kernels.hsaco"
    SAGR_CODE_OBJECT_BINARY_SEARCH_PATH="${SELF_AMDGPU_RUNTIME_REPO_ROOT}/projects/rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/binary_search_kernels.hsaco")
if(MSVC)
  target_compile_options(self_amdgpu_runtime_code_object_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_code_object_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.code_object
  COMMAND self_amdgpu_runtime_code_object_test)
set_tests_properties(self_amdgpu_runtime.code_object PROPERTIES TIMEOUT 30)

# CP-0021 compile/provenance gate for the unmodified pinned Triton tutorial.
# This target intentionally has no launcher, transport, or gem5 dependency.
add_executable(
  self_amdgpu_runtime_triton_vecadd_compile_test
  triton_vecadd_compile_test.c
  ../src/sha256.c)
target_compile_features(
  self_amdgpu_runtime_triton_vecadd_compile_test
  PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_triton_vecadd_compile_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_triton_vecadd_compile_test
  PRIVATE SelfAmdgpuRuntime::runtime)
target_compile_definitions(
  self_amdgpu_runtime_triton_vecadd_compile_test
  PRIVATE
    SAGR_TRITON_TUTORIAL_PATH="${SELF_AMDGPU_RUNTIME_REPO_ROOT}/projects/triton/python/tutorials/01-vector-add.py"
    SAGR_TRITON_HSACO_PATH="${SELF_AMDGPU_RUNTIME_REPO_ROOT}/artifacts/evidence/CP-0017/triton-builds/vecadd-gfx950.hsaco")
if(MSVC)
  target_compile_options(
    self_amdgpu_runtime_triton_vecadd_compile_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_triton_vecadd_compile_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.triton_vecadd_compile
  COMMAND self_amdgpu_runtime_triton_vecadd_compile_test)
set_tests_properties(
  self_amdgpu_runtime.triton_vecadd_compile
  PROPERTIES TIMEOUT 30)

add_executable(
  self_amdgpu_runtime_code_object_transport_test
  code_object_transport_test.c
  ../src/transport_codec.c
  ../src/sha256.c)
target_compile_features(self_amdgpu_runtime_code_object_transport_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_code_object_transport_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_code_object_transport_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(self_amdgpu_runtime_code_object_transport_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_code_object_transport_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.code_object_transport
  COMMAND self_amdgpu_runtime_code_object_transport_test)
set_tests_properties(self_amdgpu_runtime.code_object_transport PROPERTIES TIMEOUT 30)

# CP-0022 generic payload-v2 codec/admission gate.  The test exercises only
# fixed-record validation and identity linkage; it does not open a daemon or
# claim mapping, launcher, or gem5 execution.
add_executable(
  self_amdgpu_runtime_generic_dispatch_v2_transport_test
  generic_dispatch_v2_transport_test.c
  ../src/transport_codec.c)
target_compile_features(
  self_amdgpu_runtime_generic_dispatch_v2_transport_test PRIVATE c_std_11)
target_compile_definitions(
  self_amdgpu_runtime_generic_dispatch_v2_transport_test
  PRIVATE
    AMDGPU_SIM_BRIDGE_VECTOR_DIR="${SELF_AMDGPU_RUNTIME_REPO_ROOT}/tests/fixtures/runtime-gem5-bridge/generic-v2")
target_include_directories(
  self_amdgpu_runtime_generic_dispatch_v2_transport_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_generic_dispatch_v2_transport_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(
    self_amdgpu_runtime_generic_dispatch_v2_transport_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_generic_dispatch_v2_transport_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.generic_dispatch_v2_transport
  COMMAND self_amdgpu_runtime_generic_dispatch_v2_transport_test)
set_tests_properties(
  self_amdgpu_runtime.generic_dispatch_v2_transport PROPERTIES TIMEOUT 30)

add_executable(
  self_amdgpu_runtime_code_object_upload_test
  code_object_upload_test.c
  ../src/transport_codec.c)
target_compile_features(self_amdgpu_runtime_code_object_upload_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_code_object_upload_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_code_object_upload_test
  PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
target_compile_definitions(
  self_amdgpu_runtime_code_object_upload_test
  PRIVATE
    SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH="${SELF_AMDGPU_RUNTIME_REPO_ROOT}/projects/rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/gpuReadWrite_kernels.hsaco")
if(MSVC)
  target_compile_options(self_amdgpu_runtime_code_object_upload_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_code_object_upload_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.code_object_upload
  COMMAND self_amdgpu_runtime_code_object_upload_test)
set_tests_properties(self_amdgpu_runtime.code_object_upload PROPERTIES TIMEOUT 30)

add_executable(
  self_amdgpu_runtime_protocol_test
  protocol_test.c
  ../src/transport_codec.c)
target_compile_features(self_amdgpu_runtime_protocol_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_protocol_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_protocol_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(self_amdgpu_runtime_protocol_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_protocol_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.protocol
  COMMAND self_amdgpu_runtime_protocol_test)

find_package(Threads REQUIRED)
add_executable(
  self_amdgpu_runtime_transport_test
  transport_test.c
  ../src/transport_codec.c)
target_compile_features(self_amdgpu_runtime_transport_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_transport_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_transport_test
  PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
if(MSVC)
  target_compile_options(self_amdgpu_runtime_transport_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_transport_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.transport
  COMMAND self_amdgpu_runtime_transport_test)
set_tests_properties(self_amdgpu_runtime.transport PROPERTIES TIMEOUT 30)

add_executable(
  self_amdgpu_runtime_queue_test
  queue_test.c
  ../src/transport_codec.c)
target_compile_features(self_amdgpu_runtime_queue_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_queue_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_queue_test
  PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
if(SELF_AMDGPU_RUNTIME_BUILD_TOOLS)
  target_compile_definitions(
    self_amdgpu_runtime_queue_test
    PRIVATE SAGR_QUEUE_CLI_PATH="$<TARGET_FILE:sagr-handshake>")
  add_dependencies(self_amdgpu_runtime_queue_test sagr-handshake)
endif()
if(MSVC)
  target_compile_options(self_amdgpu_runtime_queue_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_queue_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.queue
  COMMAND self_amdgpu_runtime_queue_test)
set_tests_properties(self_amdgpu_runtime.queue PROPERTIES TIMEOUT 30)

add_executable(
  self_amdgpu_runtime_memory_test
  memory_test.c
  ../src/transport_codec.c)
target_compile_features(self_amdgpu_runtime_memory_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_memory_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_memory_test
  PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
if(SELF_AMDGPU_RUNTIME_BUILD_TOOLS)
  target_compile_definitions(
    self_amdgpu_runtime_memory_test
    PRIVATE SAGR_MEMORY_CLI_PATH="$<TARGET_FILE:sagr-handshake>")
  add_dependencies(self_amdgpu_runtime_memory_test sagr-handshake)
endif()
if(MSVC)
  target_compile_options(self_amdgpu_runtime_memory_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_memory_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.memory
  COMMAND self_amdgpu_runtime_memory_test)
set_tests_properties(self_amdgpu_runtime.memory PROPERTIES TIMEOUT 30)

add_executable(
  self_amdgpu_runtime_signal_test
  signal_test.c
  ../src/transport_codec.c)
target_compile_features(self_amdgpu_runtime_signal_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_signal_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_signal_test
  PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
if(SELF_AMDGPU_RUNTIME_BUILD_TOOLS)
  target_compile_definitions(
    self_amdgpu_runtime_signal_test
    PRIVATE SAGR_SIGNAL_CLI_PATH="$<TARGET_FILE:sagr-handshake>")
  add_dependencies(self_amdgpu_runtime_signal_test sagr-handshake)
endif()
if(MSVC)
  target_compile_options(self_amdgpu_runtime_signal_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_signal_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.signal
  COMMAND self_amdgpu_runtime_signal_test)
set_tests_properties(self_amdgpu_runtime.signal PROPERTIES TIMEOUT 30)

add_executable(
  self_amdgpu_runtime_dispatch_test
  dispatch_test.c
  ../src/transport_codec.c)
target_compile_features(self_amdgpu_runtime_dispatch_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_dispatch_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_dispatch_test
  PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
if(SELF_AMDGPU_RUNTIME_BUILD_TOOLS)
  target_compile_definitions(
    self_amdgpu_runtime_dispatch_test
    PRIVATE SAGR_DISPATCH_CLI_PATH="$<TARGET_FILE:sagr-handshake>")
  add_dependencies(self_amdgpu_runtime_dispatch_test sagr-handshake)
endif()
if(MSVC)
  target_compile_options(self_amdgpu_runtime_dispatch_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_dispatch_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.dispatch
  COMMAND self_amdgpu_runtime_dispatch_test)
set_tests_properties(self_amdgpu_runtime.dispatch PROPERTIES TIMEOUT 30)

# CP-0023 client lease/lifecycle gate.  The mock daemon exercises the public
# v2 API and the existing v1 queue/memory/signal carrier, but never executes a
# kernel or claims a normal launcher result.
add_executable(
  self_amdgpu_runtime_generic_dispatch_v2_client_test
  generic_dispatch_v2_client_test.c
  ../src/transport_codec.c)
target_compile_features(
  self_amdgpu_runtime_generic_dispatch_v2_client_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_generic_dispatch_v2_client_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_generic_dispatch_v2_client_test
  PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
if(MSVC)
  target_compile_options(
    self_amdgpu_runtime_generic_dispatch_v2_client_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_generic_dispatch_v2_client_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.generic_dispatch_v2_client
  COMMAND self_amdgpu_runtime_generic_dispatch_v2_client_test)
set_tests_properties(
  self_amdgpu_runtime.generic_dispatch_v2_client PROPERTIES TIMEOUT 30)

# CP-0026 endpoint probe.  This is an opt-in real-daemon execution gate: the
# endpoint and gpuReadWrite HSACO come from SAGR_GENERIC_BRIDGE_ENDPOINT and
# SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH, no mock server is started, and return
# 77 is an intentional skip when the endpoint is not configured or available.
# The positive path proves the owner-bound protocol lifecycle, three-buffer
# H2D/SUBMIT/type-20/D2H oracle, and reconnect cleanup.  Compiler, JIT, and
# launcher paths remain explicitly outside this test's claims.
add_executable(
  self_amdgpu_runtime_generic_dispatch_v2_endpoint_test
  generic_dispatch_v2_endpoint_test.c
  ../src/transport_codec.c)
target_compile_features(
  self_amdgpu_runtime_generic_dispatch_v2_endpoint_test PRIVATE c_std_11)
target_include_directories(
  self_amdgpu_runtime_generic_dispatch_v2_endpoint_test
  PRIVATE ../src ../include)
target_link_libraries(
  self_amdgpu_runtime_generic_dispatch_v2_endpoint_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(
    self_amdgpu_runtime_generic_dispatch_v2_endpoint_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_generic_dispatch_v2_endpoint_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.generic_dispatch_v2_endpoint
  COMMAND self_amdgpu_runtime_generic_dispatch_v2_endpoint_test)
set_tests_properties(
  self_amdgpu_runtime.generic_dispatch_v2_endpoint
  PROPERTIES TIMEOUT 30 SKIP_RETURN_CODE 77)

add_executable(self_amdgpu_runtime_cli_test cli_test.c)
target_compile_features(self_amdgpu_runtime_cli_test PRIVATE c_std_11)
target_link_libraries(
  self_amdgpu_runtime_cli_test
  PRIVATE SelfAmdgpuRuntime::runtime)
if(MSVC)
  target_compile_options(self_amdgpu_runtime_cli_test PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime_cli_test
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()
add_test(
  NAME self_amdgpu_runtime.cli
  COMMAND self_amdgpu_runtime_cli_test)

if(SELF_AMDGPU_RUNTIME_BUILD_OPENCL)
  add_executable(
    self_amdgpu_runtime_opencl_api_test
    opencl_api_test.c
    ../src/opencl_geometry.c)
  target_compile_features(
    self_amdgpu_runtime_opencl_api_test
    PRIVATE c_std_11)
  target_compile_definitions(
    self_amdgpu_runtime_opencl_api_test
    PRIVATE
      SAGR_OPENCL_VECADD_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../examples/opencl/vecadd.cl")
  target_include_directories(
    self_amdgpu_runtime_opencl_api_test
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src")
  target_link_libraries(
    self_amdgpu_runtime_opencl_api_test
    PRIVATE SelfAmdgpuRuntime::OpenCL SelfAmdgpuRuntime::runtime)
  if(MSVC)
    target_compile_options(
      self_amdgpu_runtime_opencl_api_test PRIVATE /W4 /WX)
  else()
    target_compile_options(
      self_amdgpu_runtime_opencl_api_test
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
  add_test(
    NAME self_amdgpu_runtime.opencl_api
    COMMAND self_amdgpu_runtime_opencl_api_test)
  set_tests_properties(
    self_amdgpu_runtime.opencl_api
    PROPERTIES TIMEOUT 120)

  add_executable(
    self_amdgpu_runtime_opencl_vecadd_e2e
    opencl_vecadd_e2e.c)
  target_compile_features(
    self_amdgpu_runtime_opencl_vecadd_e2e
    PRIVATE c_std_11)
  target_compile_definitions(
    self_amdgpu_runtime_opencl_vecadd_e2e
    PRIVATE
      SAGR_OPENCL_VECADD_SOURCE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../examples/opencl/vecadd.cl")
  target_link_libraries(
    self_amdgpu_runtime_opencl_vecadd_e2e
    PRIVATE SelfAmdgpuRuntime::OpenCL)
  if(MSVC)
    target_compile_options(
      self_amdgpu_runtime_opencl_vecadd_e2e PRIVATE /W4 /WX)
  else()
    target_compile_options(
      self_amdgpu_runtime_opencl_vecadd_e2e
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
  add_test(
    NAME self_amdgpu_runtime.opencl_vecadd_e2e
    COMMAND self_amdgpu_runtime_opencl_vecadd_e2e)
  set_tests_properties(
    self_amdgpu_runtime.opencl_vecadd_e2e
    PROPERTIES TIMEOUT 180 SKIP_RETURN_CODE 77)
endif()

if(SELF_AMDGPU_RUNTIME_TEST_INSTALLED_PACKAGE)
  add_test(
    NAME self_amdgpu_runtime.package
    COMMAND
      "${CMAKE_COMMAND}"
      "-DSELF_AMDGPU_RUNTIME_BUILD_DIR=${PROJECT_BINARY_DIR}"
      "-DSELF_AMDGPU_RUNTIME_PACKAGE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/package"
      "-DSELF_AMDGPU_RUNTIME_PACKAGE_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-test"
      "-DCMAKE_INSTALL_BINDIR=${CMAKE_INSTALL_BINDIR}"
      "-DCMAKE_INSTALL_LIBDIR=${CMAKE_INSTALL_LIBDIR}"
      "-DSELF_AMDGPU_RUNTIME_BUILD_TOOLS=${SELF_AMDGPU_RUNTIME_BUILD_TOOLS}"
      -P "${CMAKE_CURRENT_SOURCE_DIR}/run_package_test.cmake")
else()
  message(
    STATUS
      "Installed-package consumer test disabled by "
      "SELF_AMDGPU_RUNTIME_TEST_INSTALLED_PACKAGE=OFF")
endif()
