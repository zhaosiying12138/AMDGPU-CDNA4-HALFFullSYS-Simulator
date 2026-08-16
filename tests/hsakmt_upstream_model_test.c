/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <stdbool.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <hsakmt/hsakmt.h>

int main(void) {
  HsaVersionInfo version;
  HsaSystemProperties properties;
  HsaClockCounters first_clock;
  HsaClockCounters second_clock;
  bool model_enabled = false;
  HSAKMT_STATUS status;
  memset(&version, 0, sizeof(version));
  memset(&properties, 0, sizeof(properties));
  memset(&first_clock, 0, sizeof(first_clock));
  memset(&second_clock, 0, sizeof(second_clock));
  errno = 0;
  status = hsaKmtOpenKFD();
  if (status != HSAKMT_STATUS_SUCCESS) {
    fprintf(stderr, "hsaKmtOpenKFD failed: status=%d errno=%d\n",
            (int)status, errno);
    return 2;
  }
  status = hsaKmtModelEnabled(&model_enabled);
  if (status != HSAKMT_STATUS_SUCCESS || !model_enabled) {
    fprintf(stderr, "hsaKmtModelEnabled failed: status=%d enabled=%d\n",
            (int)status, model_enabled ? 1 : 0);
    (void)hsaKmtCloseKFD();
    return 3;
  }
  status = hsaKmtGetVersion(&version);
  if (status != HSAKMT_STATUS_SUCCESS ||
      version.KernelInterfaceMajorVersion != 1U ||
      version.KernelInterfaceMinorVersion != 9U) {
    fprintf(stderr,
            "hsaKmtGetVersion failed: status=%d version=%u.%u errno=%d\n",
            (int)status, version.KernelInterfaceMajorVersion,
            version.KernelInterfaceMinorVersion, errno);
    (void)hsaKmtCloseKFD();
    return 3;
  }
  status = hsaKmtAcquireSystemProperties(&properties);
  if (status != HSAKMT_STATUS_SUCCESS || properties.NumNodes != 2U) {
    fprintf(stderr,
            "hsaKmtAcquireSystemProperties failed: status=%d nodes=%u "
            "errno=%d\n",
            (int)status, properties.NumNodes, errno);
    (void)hsaKmtCloseKFD();
    return 4;
  }
  status = hsaKmtGetClockCounters(1U, &first_clock);
  if (status != HSAKMT_STATUS_SUCCESS) {
    fprintf(stderr, "first hsaKmtGetClockCounters failed: status=%d errno=%d\n",
            (int)status, errno);
    (void)hsaKmtReleaseSystemProperties();
    (void)hsaKmtCloseKFD();
    return 5;
  }
  status = hsaKmtGetClockCounters(1U, &second_clock);
  if (status != HSAKMT_STATUS_SUCCESS ||
      first_clock.GPUClockCounter == 0U ||
      second_clock.GPUClockCounter <= first_clock.GPUClockCounter ||
      first_clock.GPUClockCounter != first_clock.CPUClockCounter ||
      first_clock.GPUClockCounter != first_clock.SystemClockCounter ||
      first_clock.SystemClockFrequencyHz != UINT64_C(1000000000) ||
      second_clock.SystemClockFrequencyHz != UINT64_C(1000000000)) {
    fprintf(stderr,
            "clock correlation failed: status=%d first=%llu second=%llu "
            "frequency=%llu errno=%d\n",
            (int)status, (unsigned long long)first_clock.GPUClockCounter,
            (unsigned long long)second_clock.GPUClockCounter,
            (unsigned long long)second_clock.SystemClockFrequencyHz, errno);
    (void)hsaKmtReleaseSystemProperties();
    (void)hsaKmtCloseKFD();
    return 5;
  }
  status = hsaKmtReleaseSystemProperties();
  if (status != HSAKMT_STATUS_SUCCESS) {
    fprintf(stderr, "hsaKmtReleaseSystemProperties failed: status=%d\n",
            (int)status);
    (void)hsaKmtCloseKFD();
    return 5;
  }
  status = hsaKmtCloseKFD();
  if (status != HSAKMT_STATUS_SUCCESS) {
    fprintf(stderr, "hsaKmtCloseKFD failed: status=%d\n", (int)status);
    return 6;
  }
  return 0;
}
