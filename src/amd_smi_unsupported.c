/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Explicit refusals for the rest of the AMD SMI surface.
 *
 * The upstream amdsmi Python wrapper binds its entire symbol table at import
 * time, so a provider that exported only the discovery entry points would fail
 * to import at all. Every entry point here therefore exists, and every one of
 * them refuses: a simulated device has no fan, no power rail, and no thermal
 * sensor, so returning AMDSMI_STATUS_NOT_SUPPORTED is the honest answer and
 * keeps a caller from mistaking a fabricated reading for a measurement.
 *
 * Generated against the pinned amdsmi wrapper symbol set; discovery lives in
 * amd_smi_model.c.
 */

#include <stdint.h>

#define SAGR_SMI_EXPORT __attribute__((visibility("default")))
#define AMDSMI_STATUS_NOT_SUPPORTED 2u

/* 188 unsupported entry points. */
SAGR_SMI_EXPORT uint32_t amdsmi_clean_gpu_local_data(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_cpu_apb_disable(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_cpu_apb_enable(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_first_online_core_on_cpu_socket(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_free_name_value_pairs(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_afids_from_cper(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_clk_freq(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_clock_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_affinity_with_scope(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_cclk_limit(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_core_boostlimit(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_core_current_freq_limit(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_core_energy(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_cores_per_socket(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_current_io_bandwidth(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_current_xgmi_bw(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_ddr_bw(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_dimm_power_consumption(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_dimm_temp_range_and_refresh_rate(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_dimm_thermal_sensor(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_family(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_fclk_mclk(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_handles(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_hsmp_driver_version(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_hsmp_proto_ver(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_model(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_model_name(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_prochot_status(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_pwr_svi_telemetry_all_rails(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_smu_fw_version(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_c0_residency(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_count(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_current_active_freq_limit(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_energy(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_freq_range(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_lclk_dpm_level(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_power(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_power_cap(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_power_cap_max(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpu_socket_temperature(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_cpucore_handles(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_energy_count(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_esmi_err_msg(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_fw_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_accelerator_partition_profile(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_accelerator_partition_profile_config(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_activity(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_asic_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_available_counters(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_bad_page_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_bad_page_threshold(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_bdf_id(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_board_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_busy_percent(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_cache_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_compute_partition(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_compute_process_gpus(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_compute_process_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_compute_process_info_by_pid(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_cper_entries(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_device_bdf(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_device_uuid(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_driver_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_ecc_count(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_ecc_enabled(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_ecc_status(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_enumeration_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_event_notification(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_fan_rpms(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_fan_speed(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_fan_speed_max(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_id(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_kfd_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_mem_overdrive_level(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_memory_partition(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_memory_partition_config(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_memory_reserved_pages(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_memory_total(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_memory_usage(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_metrics_header_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_metrics_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_od_volt_curve_regions(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_od_volt_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_overdrive_level(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_partition_metrics_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_pci_bandwidth(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_pci_replay_counter(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_pci_throughput(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_perf_level(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_pm_metrics_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_power_profile_presets(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_process_isolation(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_process_list(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_ptl_formats(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_ptl_state(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_ras_block_features_enabled(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_ras_feature_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_reg_table_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_revision(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_subsystem_id(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_subsystem_name(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_topo_numa_affinity(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_total_ecc_count(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_vbios_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_vendor_name(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_virtualization_mode(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_volt_metric(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_vram_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_vram_usage(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_vram_vendor(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_xcd_counter(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_gpu_xgmi_link_status(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_hsmp_metrics_table(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_hsmp_metrics_table_version(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_lib_version(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_link_metrics(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_link_topology_nearest(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_minmax_bandwidth_between_processors(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_node_handle(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_npm_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_pcie_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_power_cap_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_power_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_processor_count_from_handles(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_processor_handle_from_bdf(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_processor_handles_by_type(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_processor_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_processor_type(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_soc_pstate(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_socket_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_supported_power_cap(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_temp_metric(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_threads_per_core(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_utilization_count(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_violation_status(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_xgmi_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_get_xgmi_plpd(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_gpu_control_counter(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_gpu_counter_group_supported(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_gpu_create_counter(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_gpu_destroy_counter(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_gpu_driver_reload(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_gpu_read_counter(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_gpu_validate_ras_eeprom(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_gpu_xgmi_error_status(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_init_gpu_event_notification(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_is_P2P_accessible(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_is_gpu_power_management_enabled(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_reset_gpu(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_reset_gpu_fan(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_reset_gpu_xgmi_error(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_clk_freq(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_cpu_core_boostlimit(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_cpu_df_pstate_range(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_cpu_gmi3_link_width_range(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_cpu_pcie_link_rate(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_cpu_pwr_efficiency_mode(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_cpu_socket_boostlimit(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_cpu_socket_lclk_dpm_level(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_cpu_socket_power_cap(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_cpu_xgmi_width(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_accelerator_partition_profile(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_clk_limit(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_clk_range(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_compute_partition(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_event_notification_mask(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_fan_speed(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_memory_partition(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_memory_partition_mode(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_od_clk_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_od_volt_info(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_overdrive_level(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_pci_bandwidth(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_perf_determinism_mode(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_perf_level(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_power_profile(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_process_isolation(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_ptl_formats(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_gpu_ptl_state(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_power_cap(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_soc_pstate(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_set_xgmi_plpd(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_status_code_to_string(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_stop_gpu_event_notification(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_topo_get_link_type(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_topo_get_link_weight(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_topo_get_numa_node_number(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
SAGR_SMI_EXPORT uint32_t amdsmi_topo_get_p2p_status(void) { return AMDSMI_STATUS_NOT_SUPPORTED; }
