# SPDX-License-Identifier: GPL-3.0-or-later

if(NOT DEFINED SAGR_HOST_NATIVE_BINARY OR
   NOT EXISTS "${SAGR_HOST_NATIVE_BINARY}")
  message(FATAL_ERROR "host-native audit requires a built consumer binary")
endif()

# The runtime is a host-side C ABI.  It may speak the existing wire protocol,
# but it must not acquire a simulator front-end or a production GPU DSO merely
# by linking the reusable boundary library.
set(_forbidden_dependencies
    "gem5"
    "vega_x86"
    "libamdhip64"
    "libhsa-runtime"
    "libhsakmt"
    "libdrm"
    "amdsmi")

find_program(_sagr_ldd ldd)
if(_sagr_ldd)
  execute_process(
    COMMAND "${_sagr_ldd}" "${SAGR_HOST_NATIVE_BINARY}"
    RESULT_VARIABLE _ldd_result
    OUTPUT_VARIABLE _ldd_output
    ERROR_VARIABLE _ldd_error)
  string(CONCAT _dependency_text "${_ldd_output}" "${_ldd_error}")
  string(TOLOWER "${_dependency_text}" _dependency_text_lower)
  foreach(_token IN LISTS _forbidden_dependencies)
    string(FIND "${_dependency_text_lower}" "${_token}" _token_offset)
    if(NOT _token_offset EQUAL -1)
      message(FATAL_ERROR
              "host-native consumer has forbidden dependency '${_token}': ${_dependency_text}")
    endif()
  endforeach()
  if(NOT _ldd_result EQUAL 0 AND
     NOT _dependency_text_lower MATCHES "not a dynamic executable")
    message(FATAL_ERROR
            "host-native dependency audit could not inspect binary: ${_dependency_text}")
  endif()
else()
  message(STATUS "host-native dependency audit: ldd unavailable")
endif()

find_program(_sagr_nm nm)
if(_sagr_nm)
  execute_process(
    COMMAND "${_sagr_nm}" -D --undefined-only "${SAGR_HOST_NATIVE_BINARY}"
    RESULT_VARIABLE _nm_result
    OUTPUT_VARIABLE _nm_output
    ERROR_VARIABLE _nm_error)
  string(CONCAT _symbol_text "${_nm_output}" "${_nm_error}")
  string(TOLOWER "${_symbol_text}" _symbol_text_lower)
  foreach(_token IN ITEMS "gem5" "vega_x86" "x86isa" "amdgpudevice"
                          "threadcontext" "setranslatingportproxy")
    string(FIND "${_symbol_text_lower}" "${_token}" _token_offset)
    if(NOT _token_offset EQUAL -1)
      message(FATAL_ERROR
              "host-native consumer has forbidden undefined symbol '${_token}'")
    endif()
  endforeach()
  if(NOT _nm_result EQUAL 0)
    message(FATAL_ERROR
            "host-native symbol audit could not inspect binary: ${_symbol_text}")
  endif()
else()
  message(STATUS "host-native symbol audit: nm unavailable")
endif()

message(STATUS "host-native runtime boundary audit passed")
