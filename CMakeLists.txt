# SPDX-License-Identifier: GPL-3.0-or-later

cmake_minimum_required(VERSION 3.20)

project(
  self_amdgpu_runtime
  VERSION 0.8.0
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
option(SELF_AMDGPU_RUNTIME_BUILD_OPENCL "Build the local OpenCL 1.2 subset" ON)
option(
  SELF_AMDGPU_RUNTIME_BUILD_HSAKMT_MODEL
  "Build the upstream libhsakmt model-interface provider"
  ON)
option(
  SELF_AMDGPU_RUNTIME_TEST_INSTALLED_PACKAGE
  "Test the installed package with an independent default-toolchain consumer"
  ON)
set(
  SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_LIBRARY
  ""
  CACHE FILEPATH "Optional unchanged upstream libhsakmt archive for the Model smoke")
set(
  SELF_AMDGPU_RUNTIME_UPSTREAM_HSAKMT_DEPENDENCY_PREFIX
  ""
  CACHE PATH "Private libdrm/NUMA prefix for the upstream libhsakmt smoke")
set(
  SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_LIBRARY
  ""
  CACHE FILEPATH "Optional unchanged upstream libhsa-runtime64 shared library for the ROCr smoke")
set(
  SELF_AMDGPU_RUNTIME_UPSTREAM_ROCR_INCLUDE_DIR
  ""
  CACHE PATH "Public headers matching the upstream libhsa-runtime64 ROCr smoke")

if(NOT UNIX)
  message(FATAL_ERROR "The managed gem5 runtime requires UNIX")
endif()
find_package(Threads REQUIRED)
set(
  SELF_AMDGPU_RUNTIME_REPO_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../.."
  CACHE PATH "Repository root containing the managed gem5 and test inputs")
get_filename_component(
  SELF_AMDGPU_RUNTIME_REPO_ROOT
  "${SELF_AMDGPU_RUNTIME_REPO_ROOT}"
  ABSOLUTE)
set(
  SELF_AMDGPU_RUNTIME_OPENCL_PREFIX
  "${CMAKE_INSTALL_PREFIX}"
  CACHE PATH "Versioned local ROCm prefix used by the managed runtime")
set(
  SELF_AMDGPU_RUNTIME_OPENCL_GEM5
  "${SELF_AMDGPU_RUNTIME_REPO_ROOT}/projects/gem5/build/VEGA_X86/gem5.opt"
  CACHE FILEPATH "Repository-local gem5 executable for the managed runtime")
set(
  SELF_AMDGPU_RUNTIME_OPENCL_GEM5_CONFIG
  "${SELF_AMDGPU_RUNTIME_REPO_ROOT}/projects/gem5/configs/example/gemsim/host_dispatch.py"
  CACHE FILEPATH "Repository-local gem5 dispatch configuration")

add_library(
  self_amdgpu_runtime
  src/runtime.c
  src/transport.c
  src/transport_codec.c
  src/provider.c
  src/kmt_shim.c
  src/code_object.c
  src/ccl_v1.c
  src/ccl_carrier_v1.c
  src/ccl_live_v1.c
  src/managed_session.c
  src/smi_registry.c
  src/sha256.c)
add_library(SelfAmdgpuRuntime::runtime ALIAS self_amdgpu_runtime)

set_target_properties(
  self_amdgpu_runtime
  PROPERTIES C_VISIBILITY_PRESET hidden
             EXPORT_NAME runtime
             POSITION_INDEPENDENT_CODE ON
             VISIBILITY_INLINES_HIDDEN YES
             VERSION "${PROJECT_VERSION}"
             SOVERSION "1")

target_compile_features(self_amdgpu_runtime PUBLIC c_std_11)
target_compile_definitions(
  self_amdgpu_runtime
  PRIVATE
    SELF_AMDGPU_RUNTIME_BUILDING_LIBRARY=1
    SAGR_MANAGED_DEFAULT_PREFIX="${SELF_AMDGPU_RUNTIME_OPENCL_PREFIX}"
    SAGR_MANAGED_DEFAULT_GEM5="${SELF_AMDGPU_RUNTIME_OPENCL_GEM5}"
    SAGR_MANAGED_DEFAULT_GEM5_CONFIG="${SELF_AMDGPU_RUNTIME_OPENCL_GEM5_CONFIG}"
    SAGR_MANAGED_DEFAULT_REPO_ROOT="${SELF_AMDGPU_RUNTIME_REPO_ROOT}")
if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(
    self_amdgpu_runtime
    PUBLIC SELF_AMDGPU_RUNTIME_STATIC_DEFINE=1)
endif()
target_include_directories(
  self_amdgpu_runtime
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(self_amdgpu_runtime PRIVATE Threads::Threads)

if(SELF_AMDGPU_RUNTIME_BUILD_HSAKMT_MODEL)
  set(
    SELF_AMDGPU_RUNTIME_HSAKMT_INCLUDE_DIR
    "${SELF_AMDGPU_RUNTIME_REPO_ROOT}/projects/rocm-systems/projects/rocr-runtime/libhsakmt/include"
    CACHE PATH "Pinned upstream libhsakmt public include directory")
  if(NOT EXISTS
     "${SELF_AMDGPU_RUNTIME_HSAKMT_INCLUDE_DIR}/hsakmt/hsakmtmodeliface.h")
    message(
      FATAL_ERROR
        "The pinned upstream hsakmt model-interface header is unavailable")
  endif()
  add_library(
    self_amdgpu_hsakmt_model SHARED
    src/hsakmt_model.c)
  add_library(
    SelfAmdgpuRuntime::HsakmtModel ALIAS self_amdgpu_hsakmt_model)
  set_target_properties(
    self_amdgpu_hsakmt_model
    PROPERTIES C_VISIBILITY_PRESET hidden
               EXPORT_NAME HsakmtModel
               INSTALL_RPATH "$ORIGIN"
               OUTPUT_NAME self_amdgpu_hsakmt_model
               VERSION "1.1.0"
               SOVERSION "1"
               VISIBILITY_INLINES_HIDDEN YES)
  target_compile_features(self_amdgpu_hsakmt_model PRIVATE c_std_11)
  target_include_directories(
    self_amdgpu_hsakmt_model
    SYSTEM PRIVATE "${SELF_AMDGPU_RUNTIME_HSAKMT_INCLUDE_DIR}")
  target_link_libraries(
    self_amdgpu_hsakmt_model
    PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
  if(MSVC)
    target_compile_options(self_amdgpu_hsakmt_model PRIVATE /W4 /WX)
  else()
    target_compile_options(
      self_amdgpu_hsakmt_model
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
    target_link_options(
      self_amdgpu_hsakmt_model
      PRIVATE
        "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/hsakmt_model.map")
  endif()

  # Simulator-aware AMD SMI device discovery. Unchanged upstream vLLM and
  # PyTorch decide whether they are on ROCm by asking AMD SMI to enumerate
  # processors; with the KMD replaced, the production library reports
  # DRIVER_NOT_LOADED and ROCm auto-selection fails before HIP is consulted.
  # The soname matches what the upstream amdsmi Python package searches for.
  add_library(self_amdgpu_amd_smi SHARED src/amd_smi_model.c
                                          src/amd_smi_unsupported.c)
  add_library(SelfAmdgpuRuntime::AmdSmi ALIAS self_amdgpu_amd_smi)
  set_target_properties(
    self_amdgpu_amd_smi
    PROPERTIES C_VISIBILITY_PRESET hidden
               EXPORT_NAME AmdSmi
               INSTALL_RPATH "$ORIGIN"
               OUTPUT_NAME amd_smi
               VERSION "26.0.0"
               SOVERSION "26"
               VISIBILITY_INLINES_HIDDEN YES)
  target_compile_features(self_amdgpu_amd_smi PRIVATE c_std_11)
  target_link_libraries(self_amdgpu_amd_smi PRIVATE Threads::Threads)
  if(MSVC)
    target_compile_options(self_amdgpu_amd_smi PRIVATE /W4 /WX)
  else()
    target_compile_options(
      self_amdgpu_amd_smi
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
  install(TARGETS self_amdgpu_amd_smi LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()

if(SELF_AMDGPU_RUNTIME_BUILD_OPENCL)
  add_library(
    self_amdgpu_opencl SHARED
    src/opencl_api.c
    src/opencl_compiler.c
    src/opencl_enqueue.c
    src/opencl_geometry.c
    src/opencl_objects.c
    src/simulator_process.c
    src/sha256.c)
  add_library(SelfAmdgpuRuntime::OpenCL ALIAS self_amdgpu_opencl)
  set_target_properties(
    self_amdgpu_opencl
    PROPERTIES C_VISIBILITY_PRESET hidden
               EXPORT_NAME OpenCL
               INSTALL_RPATH "$ORIGIN"
               OUTPUT_NAME OpenCL
               VERSION "1.2.0"
               SOVERSION "1"
               VISIBILITY_INLINES_HIDDEN YES)
  target_compile_features(self_amdgpu_opencl PUBLIC c_std_11)
  target_compile_definitions(
    self_amdgpu_opencl
    PRIVATE
      SAGR_OPENCL_DEFAULT_PREFIX="${SELF_AMDGPU_RUNTIME_OPENCL_PREFIX}"
      SAGR_OPENCL_DEFAULT_GEM5="${SELF_AMDGPU_RUNTIME_OPENCL_GEM5}"
      SAGR_OPENCL_DEFAULT_GEM5_CONFIG="${SELF_AMDGPU_RUNTIME_OPENCL_GEM5_CONFIG}"
      SAGR_OPENCL_DEFAULT_REPO_ROOT="${SELF_AMDGPU_RUNTIME_REPO_ROOT}")
  target_include_directories(
    self_amdgpu_opencl
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(
    self_amdgpu_opencl
    PRIVATE SelfAmdgpuRuntime::runtime Threads::Threads)
  if(MSVC)
    target_compile_options(self_amdgpu_opencl PRIVATE /W4 /WX)
  else()
    target_compile_options(
      self_amdgpu_opencl
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()

  add_executable(
    self_amdgpu_opencl_vecadd
    examples/opencl/vecadd_host.c)
  set_target_properties(
    self_amdgpu_opencl_vecadd
    PROPERTIES INSTALL_RPATH
               "$ORIGIN/${SELF_AMDGPU_RUNTIME_INSTALL_LIBDIR_FROM_BINDIR}"
               OUTPUT_NAME opencl-vecadd)
  target_compile_features(self_amdgpu_opencl_vecadd PRIVATE c_std_11)
  target_compile_definitions(
    self_amdgpu_opencl_vecadd
    PRIVATE
      SAGR_OPENCL_VECADD_SOURCE_PATH="${CMAKE_INSTALL_FULL_DATADIR}/self-amdgpu-runtime/opencl/vecadd.cl")
  target_link_libraries(
    self_amdgpu_opencl_vecadd PRIVATE SelfAmdgpuRuntime::OpenCL)
  if(MSVC)
    target_compile_options(self_amdgpu_opencl_vecadd PRIVATE /W4 /WX)
  else()
    target_compile_options(
      self_amdgpu_opencl_vecadd
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
endif()

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

  add_executable(sagr-triton-hsaco-probe tools/sagr-triton-hsaco-probe.c)
  set_target_properties(
    sagr-triton-hsaco-probe
    PROPERTIES INSTALL_RPATH
               "$ORIGIN/${SELF_AMDGPU_RUNTIME_INSTALL_LIBDIR_FROM_BINDIR}")
  target_compile_features(sagr-triton-hsaco-probe PRIVATE c_std_11)
  target_link_libraries(sagr-triton-hsaco-probe PRIVATE SelfAmdgpuRuntime::runtime)
  if(MSVC)
    target_compile_options(sagr-triton-hsaco-probe PRIVATE /W4 /WX)
  else()
    target_compile_options(
      sagr-triton-hsaco-probe
      PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Werror)
  endif()
  install(TARGETS sagr-triton-hsaco-probe RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}")
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
if(SELF_AMDGPU_RUNTIME_BUILD_HSAKMT_MODEL)
  install(
    TARGETS self_amdgpu_hsakmt_model
    EXPORT SelfAmdgpuRuntimeTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
endif()
if(SELF_AMDGPU_RUNTIME_BUILD_OPENCL)
  install(
    TARGETS self_amdgpu_opencl self_amdgpu_opencl_vecadd
    EXPORT SelfAmdgpuRuntimeTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
  install(
    FILES examples/opencl/vecadd.cl
    DESTINATION "${CMAKE_INSTALL_DATADIR}/self-amdgpu-runtime/opencl")
endif()
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
