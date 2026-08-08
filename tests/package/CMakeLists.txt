# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)
project(self_amdgpu_runtime_package_consumer LANGUAGES C)

find_package(SelfAmdgpuRuntime 0.6 CONFIG REQUIRED)

add_executable(self_amdgpu_runtime_package_consumer main.c)
target_link_libraries(
  self_amdgpu_runtime_package_consumer
  PRIVATE SelfAmdgpuRuntime::runtime)

enable_testing()
add_test(
  NAME self_amdgpu_runtime.package_consumer
  COMMAND self_amdgpu_runtime_package_consumer)
