if(NOT DEFINED SAGR_HSAKMT_MODEL OR NOT EXISTS "${SAGR_HSAKMT_MODEL}")
  message(FATAL_ERROR "hsakmt model DSO is unavailable")
endif()

find_program(SAGR_READELF readelf REQUIRED)
execute_process(
  COMMAND "${SAGR_READELF}" --dynamic --wide "${SAGR_HSAKMT_MODEL}"
  RESULT_VARIABLE dynamic_status
  OUTPUT_VARIABLE dynamic_output
  ERROR_VARIABLE dynamic_error)
if(NOT dynamic_status EQUAL 0)
  message(FATAL_ERROR "readelf failed: ${dynamic_error}")
endif()
foreach(forbidden IN ITEMS libhsakmt libdrm libdrm_amdgpu libamd_smi)
  if(dynamic_output MATCHES "${forbidden}")
    message(FATAL_ERROR "model DSO links forbidden host library ${forbidden}")
  endif()
endforeach()

execute_process(
  COMMAND "${SAGR_READELF}" --dyn-syms --wide "${SAGR_HSAKMT_MODEL}"
  RESULT_VARIABLE symbols_status
  OUTPUT_VARIABLE symbols_output
  ERROR_VARIABLE symbols_error)
if(NOT symbols_status EQUAL 0)
  message(FATAL_ERROR "readelf symbols failed: ${symbols_error}")
endif()
if(NOT symbols_output MATCHES "get_hsakmt_model_functions@@SAGR_HSAKMT_MODEL_1.1")
  message(FATAL_ERROR "official hsakmt model getter is not versioned and exported")
endif()
foreach(forbidden IN ITEMS hsaKmtOpenKFD amdgpu_device_initialize drmCommandWriteRead)
  if(symbols_output MATCHES "${forbidden}")
    message(FATAL_ERROR "model DSO leaks upper-runtime symbol ${forbidden}")
  endif()
endforeach()
