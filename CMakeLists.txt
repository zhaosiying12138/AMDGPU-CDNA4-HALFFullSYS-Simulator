# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)

project(
  self_amdgpu_runtime
  VERSION 0.3.0
  DESCRIPTION "Host-side C runtime foundation for the self-amdgpu gem5 transport"
  LANGUAGES C)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

file(
  RELATIVE_PATH SELF_AMDGPU_RUNTIME_INSTALL_LIBDIR_FROM_BINDIR
  "${CMAKE_INSTALL_FULL_BINDIR}"
  "${CMAKE_INSTALL_FULL_LIBDIR}")

option(SELF_AMDGPU_RUNTIME_BUILD_TESTS "Build self-amdgpu-runtime tests" ON)
option(SELF_AMDGPU_RUNTIME_BUILD_TOOLS "Build self-amdgpu-runtime tools" ON)
option(
  SELF_AMDGPU_RUNTIME_TEST_INSTALLED_PACKAGE
  "Test the installed package with an independent default-toolchain consumer"
  ON)

add_library(
  self_amdgpu_runtime
  src/runtime.c
  src/transport.c
  src/transport_codec.c)
add_library(SelfAmdgpuRuntime::runtime ALIAS self_amdgpu_runtime)

set_target_properties(
  self_amdgpu_runtime
  PROPERTIES C_VISIBILITY_PRESET hidden
             EXPORT_NAME runtime
             VISIBILITY_INLINES_HIDDEN YES
             VERSION "${PROJECT_VERSION}"
             SOVERSION "1")

target_compile_features(self_amdgpu_runtime PUBLIC c_std_11)
target_compile_definitions(
  self_amdgpu_runtime
  PRIVATE SELF_AMDGPU_RUNTIME_BUILDING_LIBRARY=1)
if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(
    self_amdgpu_runtime
    PUBLIC SELF_AMDGPU_RUNTIME_STATIC_DEFINE=1)
endif()
target_include_directories(
  self_amdgpu_runtime
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(SELF_AMDGPU_RUNTIME_BUILD_TOOLS)
  add_executable(sagr-handshake tools/sagr-handshake.c)
  set_target_properties(
    sagr-handshake
    PROPERTIES INSTALL_RPATH
               "$ORIGIN/${SELF_AMDGPU_RUNTIME_INSTALL_LIBDIR_FROM_BINDIR}")
  target_compile_features(sagr-handshake PRIVATE c_std_11)
  target_link_libraries(sagr-handshake PRIVATE SelfAmdgpuRuntime::runtime)
  if(MSVC)
    target_compile_options(sagr-handshake PRIVATE /W4 /WX)
  else()
    target_compile_options(
      sagr-handshake
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
  install(TARGETS sagr-handshake RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif()

if(MSVC)
  target_compile_options(self_amdgpu_runtime PRIVATE /W4 /WX)
else()
  target_compile_options(
    self_amdgpu_runtime
    PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
endif()

install(
  TARGETS self_amdgpu_runtime
  EXPORT SelfAmdgpuRuntimeTargets
  LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
  RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
  INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(
  DIRECTORY include/
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
  FILES_MATCHING
  PATTERN "*.h")

set(SELF_AMDGPU_RUNTIME_CMAKE_INSTALL_DIR
    "${CMAKE_INSTALL_LIBDIR}/cmake/SelfAmdgpuRuntime")

configure_package_config_file(
  cmake/SelfAmdgpuRuntimeConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/SelfAmdgpuRuntimeConfig.cmake"
  INSTALL_DESTINATION "${SELF_AMDGPU_RUNTIME_CMAKE_INSTALL_DIR}")
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/SelfAmdgpuRuntimeConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion)

install(
  EXPORT SelfAmdgpuRuntimeTargets
  FILE SelfAmdgpuRuntimeTargets.cmake
  NAMESPACE SelfAmdgpuRuntime::
  DESTINATION "${SELF_AMDGPU_RUNTIME_CMAKE_INSTALL_DIR}")
install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/SelfAmdgpuRuntimeConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/SelfAmdgpuRuntimeConfigVersion.cmake"
  DESTINATION "${SELF_AMDGPU_RUNTIME_CMAKE_INSTALL_DIR}")

if(SELF_AMDGPU_RUNTIME_BUILD_TESTS)
  include(CTest)
  if(BUILD_TESTING)
    add_subdirectory(tests)
  endif()
endif()
