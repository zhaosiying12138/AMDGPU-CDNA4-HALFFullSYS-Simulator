# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(self_amdgpu_runtime_api_test runtime_test.c)
target_link_libraries(self_amdgpu_runtime_api_test PRIVATE SelfAmdgpuRuntime::runtime)
add_test(NAME self_amdgpu_runtime.api COMMAND self_amdgpu_runtime_api_test)

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
    SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/gpuReadWrite_kernels.hsaco"
    SAGR_CODE_OBJECT_BINARY_SEARCH_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/binary_search_kernels.hsaco")
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
    SAGR_CODE_OBJECT_GPU_READ_WRITE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../rocm-systems/projects/rdc/rdc_libs/rdc_modules/kernels/hsaco/gfx950/gpuReadWrite_kernels.hsaco")
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
