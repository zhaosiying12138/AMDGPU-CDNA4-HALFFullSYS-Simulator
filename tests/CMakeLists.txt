# SPDX-License-Identifier: GPL-3.0-or-later

add_executable(self_amdgpu_runtime_api_test runtime_test.c)
target_link_libraries(self_amdgpu_runtime_api_test PRIVATE SelfAmdgpuRuntime::runtime)
add_test(NAME self_amdgpu_runtime.api COMMAND self_amdgpu_runtime_api_test)

add_test(
  NAME self_amdgpu_runtime.package
  COMMAND
    "${CMAKE_COMMAND}"
    "-DSELF_AMDGPU_RUNTIME_BUILD_DIR=${PROJECT_BINARY_DIR}"
    "-DSELF_AMDGPU_RUNTIME_PACKAGE_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/package"
    "-DSELF_AMDGPU_RUNTIME_PACKAGE_TEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/package-test"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/run_package_test.cmake")
