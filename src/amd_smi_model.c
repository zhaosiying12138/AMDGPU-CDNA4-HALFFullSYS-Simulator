/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Simulator-aware AMD SMI device-discovery provider.
 *
 * Unchanged upstream frameworks decide whether they are on ROCm by asking AMD
 * SMI to enumerate processors, not by asking HIP.  vLLM's rocm_platform_plugin
 * calls amdsmi_init() plus amdsmi_get_processor_handles() and silently falls
 * back to an unspecified platform when that returns nothing, and PyTorch's
 * torch.cuda emits "Can't initialize amdsmi" for the same reason.  On a host
 * whose KMD has been replaced by the gem5 model there is no /dev/kfd, so the
 * production library reports AMDSMI_STATUS_DRIVER_NOT_LOADED and every
 * framework's ROCm auto-selection fails before it ever reaches HIP.
 *
 * This provider answers only the discovery questions, from the same simulated
 * topology the model DSO already publishes.  It never opens /dev/kfd,
 * /dev/dri, or any production management library, and it fails closed: with no
 * readable topology it reports an initialisation error rather than inventing a
 * device.  Everything beyond enumeration remains NOT_SUPPORTED so a caller
 * that needs real telemetry gets an explicit refusal instead of a fake
 * reading.
 */

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAGR_SMI_EXPORT __attribute__((visibility("default")))

typedef uint32_t amdsmi_status_t;

#define AMDSMI_STATUS_SUCCESS 0u
#define AMDSMI_STATUS_INVAL 1u
#define AMDSMI_STATUS_NOT_SUPPORTED 2u
#define AMDSMI_STATUS_INIT_ERROR 18u
#define AMDSMI_STATUS_NOT_FOUND 31u

typedef void *amdsmi_socket_handle;
typedef void *amdsmi_processor_handle;

/* One simulated device per socket keeps the socket/processor relation trivial
 * and matches how a discrete GPU is reported on real hardware. */
#define SAGR_SMI_MAX_DEVICES 16

struct sagr_smi_device {
    uint32_t node_id;
    uint32_t gfx_target_version;
    uint32_t device_id;
};

static pthread_mutex_t sagr_smi_lock = PTHREAD_MUTEX_INITIALIZER;
static int sagr_smi_initialised;
static uint32_t sagr_smi_device_count;
static struct sagr_smi_device sagr_smi_devices[SAGR_SMI_MAX_DEVICES];

/* Handles are opaque to callers; hand out stable distinct addresses. */
static amdsmi_socket_handle sagr_smi_socket_handle(uint32_t index)
{
    return (amdsmi_socket_handle)&sagr_smi_devices[index];
}

static int sagr_smi_socket_index(amdsmi_socket_handle handle, uint32_t *index)
{
    for (uint32_t i = 0; i < sagr_smi_device_count; ++i) {
        if (handle == sagr_smi_socket_handle(i)) {
            *index = i;
            return 1;
        }
    }
    return 0;
}

/* Read one "key value" line out of a KFD-style properties file. */
static int sagr_smi_read_property(const char *path, const char *key,
                                  unsigned long long *value)
{
    FILE *stream = fopen(path, "re");
    if (stream == NULL) {
        return 0;
    }

    char line[512];
    const size_t key_length = strlen(key);
    int found = 0;
    while (fgets(line, (int)sizeof(line), stream) != NULL) {
        if (strncmp(line, key, key_length) != 0 || line[key_length] != ' ') {
            continue;
        }
        errno = 0;
        char *end = NULL;
        const unsigned long long parsed = strtoull(line + key_length + 1, &end, 10);
        if (end != line + key_length + 1 && errno == 0) {
            *value = parsed;
            found = 1;
        }
        break;
    }

    fclose(stream);
    return found;
}

/* A KFD node is a GPU when it owns SIMDs. A CPU node reports simd_count 0. */
static void sagr_smi_scan_node(const char *topology, const char *entry)
{
    char *end = NULL;
    errno = 0;
    const unsigned long node_id = strtoul(entry, &end, 10);
    if (end == entry || *end != '\0' || errno != 0 || node_id > UINT32_MAX) {
        return;
    }

    char path[PATH_MAX];
    if (snprintf(path, sizeof(path), "%s/nodes/%s/properties", topology, entry) >=
        (int)sizeof(path)) {
        return;
    }

    unsigned long long simd_count = 0;
    if (!sagr_smi_read_property(path, "simd_count", &simd_count) || simd_count == 0) {
        return;
    }

    if (sagr_smi_device_count >= SAGR_SMI_MAX_DEVICES) {
        return;
    }

    unsigned long long gfx_target_version = 0;
    unsigned long long device_id = 0;
    (void)sagr_smi_read_property(path, "gfx_target_version", &gfx_target_version);
    (void)sagr_smi_read_property(path, "device_id", &device_id);

    struct sagr_smi_device *device = &sagr_smi_devices[sagr_smi_device_count++];
    device->node_id = (uint32_t)node_id;
    device->gfx_target_version = (uint32_t)gfx_target_version;
    device->device_id = (uint32_t)device_id;
}

static amdsmi_status_t sagr_smi_discover(void)
{
    const char *topology = getenv("HSA_MODEL_TOPOLOGY");
    if (topology == NULL || topology[0] != '/') {
        return AMDSMI_STATUS_INIT_ERROR;
    }

    char nodes[PATH_MAX];
    if (snprintf(nodes, sizeof(nodes), "%s/nodes", topology) >= (int)sizeof(nodes)) {
        return AMDSMI_STATUS_INIT_ERROR;
    }

    DIR *directory = opendir(nodes);
    if (directory == NULL) {
        return AMDSMI_STATUS_INIT_ERROR;
    }

    sagr_smi_device_count = 0;
    const struct dirent *entry = NULL;
    while ((entry = readdir(directory)) != NULL) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        sagr_smi_scan_node(topology, entry->d_name);
    }
    closedir(directory);

    /* Report ascending node order so repeated enumeration is stable. */
    for (uint32_t i = 1; i < sagr_smi_device_count; ++i) {
        const struct sagr_smi_device key = sagr_smi_devices[i];
        uint32_t j = i;
        while (j > 0 && sagr_smi_devices[j - 1].node_id > key.node_id) {
            sagr_smi_devices[j] = sagr_smi_devices[j - 1];
            --j;
        }
        sagr_smi_devices[j] = key;
    }

    return sagr_smi_device_count > 0 ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_NOT_FOUND;
}

SAGR_SMI_EXPORT amdsmi_status_t amdsmi_init(uint64_t flags)
{
    (void)flags;
    pthread_mutex_lock(&sagr_smi_lock);
    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    if (!sagr_smi_initialised) {
        status = sagr_smi_discover();
        sagr_smi_initialised = (status == AMDSMI_STATUS_SUCCESS);
    }
    pthread_mutex_unlock(&sagr_smi_lock);
    return status;
}

SAGR_SMI_EXPORT amdsmi_status_t amdsmi_shut_down(void)
{
    pthread_mutex_lock(&sagr_smi_lock);
    sagr_smi_initialised = 0;
    sagr_smi_device_count = 0;
    pthread_mutex_unlock(&sagr_smi_lock);
    return AMDSMI_STATUS_SUCCESS;
}

/*
 * Both enumeration entry points use the caller-counts-first convention: a null
 * output array reports how many entries exist, and a non-null array is filled
 * with at most the count the caller supplied.
 */
SAGR_SMI_EXPORT amdsmi_status_t amdsmi_get_socket_handles(
    uint32_t *socket_count, amdsmi_socket_handle *socket_handles)
{
    if (socket_count == NULL) {
        return AMDSMI_STATUS_INVAL;
    }

    pthread_mutex_lock(&sagr_smi_lock);
    if (!sagr_smi_initialised) {
        pthread_mutex_unlock(&sagr_smi_lock);
        return AMDSMI_STATUS_INIT_ERROR;
    }

    if (socket_handles == NULL) {
        *socket_count = sagr_smi_device_count;
    } else {
        const uint32_t limit = *socket_count < sagr_smi_device_count
                                   ? *socket_count
                                   : sagr_smi_device_count;
        for (uint32_t i = 0; i < limit; ++i) {
            socket_handles[i] = sagr_smi_socket_handle(i);
        }
        *socket_count = limit;
    }
    pthread_mutex_unlock(&sagr_smi_lock);
    return AMDSMI_STATUS_SUCCESS;
}

SAGR_SMI_EXPORT amdsmi_status_t amdsmi_get_processor_handles(
    amdsmi_socket_handle socket_handle, uint32_t *processor_count,
    amdsmi_processor_handle *processor_handles)
{
    if (processor_count == NULL) {
        return AMDSMI_STATUS_INVAL;
    }

    pthread_mutex_lock(&sagr_smi_lock);
    if (!sagr_smi_initialised) {
        pthread_mutex_unlock(&sagr_smi_lock);
        return AMDSMI_STATUS_INIT_ERROR;
    }

    uint32_t index = 0;
    if (!sagr_smi_socket_index(socket_handle, &index)) {
        pthread_mutex_unlock(&sagr_smi_lock);
        return AMDSMI_STATUS_INVAL;
    }

    if (processor_handles == NULL) {
        *processor_count = 1u;
    } else {
        if (*processor_count >= 1u) {
            processor_handles[0] = (amdsmi_processor_handle)&sagr_smi_devices[index];
            *processor_count = 1u;
        } else {
            *processor_count = 0u;
        }
    }
    pthread_mutex_unlock(&sagr_smi_lock);
    return AMDSMI_STATUS_SUCCESS;
}
