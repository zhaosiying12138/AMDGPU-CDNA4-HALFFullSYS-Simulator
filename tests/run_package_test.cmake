# SPDX-License-Identifier: GPL-3.0-or-later

foreach(required_variable IN ITEMS SELF_AMDGPU_RUNTIME_BUILD_DIR
                                   SELF_AMDGPU_RUNTIME_PACKAGE_SOURCE_DIR
                                   SELF_AMDGPU_RUNTIME_PACKAGE_TEST_DIR)
  if(NOT DEFINED ${required_variable} OR "${${required_variable}}" STREQUAL "")
    message(FATAL_ERROR "${required_variable} is required")
  endif()
endforeach()

set(install_prefix "${SELF_AMDGPU_RUNTIME_PACKAGE_TEST_DIR}/install")
set(consumer_build_dir "${SELF_AMDGPU_RUNTIME_PACKAGE_TEST_DIR}/build")
file(REMOVE_RECURSE "${SELF_AMDGPU_RUNTIME_PACKAGE_TEST_DIR}")

execute_process(
  COMMAND "${CMAKE_COMMAND}" --install "${SELF_AMDGPU_RUNTIME_BUILD_DIR}"
          --prefix "${install_prefix}"
  RESULT_VARIABLE command_result
  COMMAND_ECHO STDOUT)
if(NOT command_result EQUAL 0)
  message(FATAL_ERROR "package install failed: ${command_result}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}"
          -S "${SELF_AMDGPU_RUNTIME_PACKAGE_SOURCE_DIR}"
          -B "${consumer_build_dir}"
          "-DCMAKE_PREFIX_PATH=${install_prefix}"
  RESULT_VARIABLE command_result
  COMMAND_ECHO STDOUT)
if(NOT command_result EQUAL 0)
  message(FATAL_ERROR "package consumer configure failed: ${command_result}")
endif()

execute_process(
  COMMAND "${CMAKE_COMMAND}" --build "${consumer_build_dir}"
  RESULT_VARIABLE command_result
  COMMAND_ECHO STDOUT)
if(NOT command_result EQUAL 0)
  message(FATAL_ERROR "package consumer build failed: ${command_result}")
endif()

execute_process(
  COMMAND "${CMAKE_CTEST_COMMAND}" --test-dir "${consumer_build_dir}"
          --output-on-failure
  RESULT_VARIABLE command_result
  COMMAND_ECHO STDOUT)
if(NOT command_result EQUAL 0)
  message(FATAL_ERROR "package consumer test failed: ${command_result}")
endif()
