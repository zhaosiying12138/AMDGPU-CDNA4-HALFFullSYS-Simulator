# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(self_amdgpu_runtime_api_test runtime_test.c)
target_link_libraries(self_amdgpu_runtime_api_test PRIVATE SelfAmdgpuRuntime::runtime)
add_test(NAME self_amdgpu_runtime.api COMMAND self_amdgpu_runtime_api_test)

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
