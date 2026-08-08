# SPDX-License-Identifier: GPL-3.0-or-later

if(NOT DEFINED SAGR_PROVIDER_SOURCE_FILE OR
   NOT DEFINED SAGR_PROVIDER_HEADER_FILE)
  message(FATAL_ERROR "provider audit requires source and header paths")
endif()

set(_forbidden_source_tokens
    "/dev/kfd"
    "/dev/dri"
    "/dev/dxg"
    "/dev/udmabuf"
    "dlopen"
    "dlsym"
    "libhsakmt"
    "libdrm"
    "amdsmi")

foreach(_source_file IN ITEMS "${SAGR_PROVIDER_SOURCE_FILE}"
                              "${SAGR_PROVIDER_HEADER_FILE}"
                              "${SAGR_KMT_SOURCE_FILE}"
                              "${SAGR_KMT_HEADER_FILE}")
  if(_source_file STREQUAL "")
    continue()
  endif()
  if(NOT EXISTS "${_source_file}")
    message(FATAL_ERROR "provider audit input does not exist: ${_source_file}")
  endif()
  file(READ "${_source_file}" _source_text)
  foreach(_token IN LISTS _forbidden_source_tokens)
    string(FIND "${_source_text}" "${_token}" _token_offset)
    if(NOT _token_offset EQUAL -1)
      message(FATAL_ERROR
              "provider source contains forbidden token '${_token}': ${_source_file}")
    endif()
  endforeach()
endforeach()

# A dynamic test executable is the closest portable dependency check available
# in the child build. A static executable is accepted when ldd reports that it
# is not dynamic; either way production GPU DSOs must not appear.
if(DEFINED SAGR_PROVIDER_BINARY AND EXISTS "${SAGR_PROVIDER_BINARY}")
  find_program(SAGR_LDD_EXECUTABLE ldd)
  if(SAGR_LDD_EXECUTABLE)
    execute_process(
      COMMAND "${SAGR_LDD_EXECUTABLE}" "${SAGR_PROVIDER_BINARY}"
      RESULT_VARIABLE _ldd_result
      OUTPUT_VARIABLE _ldd_output
      ERROR_VARIABLE _ldd_error)
    string(CONCAT _dependency_text "${_ldd_output}" "${_ldd_error}")
    string(TOLOWER "${_dependency_text}" _dependency_text_lower)
    foreach(_token IN ITEMS "libhsakmt" "libdrm" "libdrm_amdgpu" "amdsmi")
      string(FIND "${_dependency_text_lower}" "${_token}" _token_offset)
      if(NOT _token_offset EQUAL -1)
        message(FATAL_ERROR
                "provider test executable has forbidden dependency '${_token}'")
      endif()
    endforeach()
    if(NOT _ldd_result EQUAL 0 AND
       NOT _dependency_text_lower MATCHES "not a dynamic executable")
      message(FATAL_ERROR
              "dependency audit could not inspect provider test executable: ${_dependency_text}")
    endif()
  else()
    message(STATUS "provider dependency audit skipped: ldd unavailable")
  endif()
endif()

message(STATUS "provider no-device/DSO audit passed")
