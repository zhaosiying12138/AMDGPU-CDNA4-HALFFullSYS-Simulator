/* SPDX-License-Identifier: GPL-3.0-or-later */
/*
 * Simulator-aware rocminfo.
 *
 * Frameworks do not only ask libraries what hardware exists -- some shell out
 * to `rocminfo` and parse its output. aiter's chip_info.get_gfx_runtime() does
 * exactly that, and it documents that it deliberately ignores the GPU_ARCHS
 * override because it wants "the arch of the live GPU". Upstream rocminfo
 * refuses before it ever reaches ROCr, printing "ROCk module is NOT loaded",
 * because it looks for the amdgpu kernel module. On a host whose KMD has been
 * replaced by the gem5 model that check can never pass.
 *
 * This reports the simulated agents from the same topology the model DSO
 * publishes, in the subset of rocminfo's format that callers actually parse:
 * one "Name:" and one "Gfx Version:"/"Marketing Name" block per agent. It
 * never opens /dev/kfd, /dev/dri or /sys/class/kfd, and it fails closed --
 * with no readable topology it exits non-zero rather than naming a device
 * that does not exist, so a caller cannot mistake it for a live GPU.
 *
 * It is deliberately not a full rocminfo: everything it does not know, it does
 * not print.
 */

#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAGR_ROCMINFO_MAX_NODES 64

struct sagr_node {
    unsigned long node_id;
    unsigned long long simd_count;
    unsigned long long gfx_target_version;
    unsigned long long cpu_cores_count;
};

static int read_property(const char *path, const char *key,
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

/* gfx_target_version encodes major/minor/step as MMmmss, e.g. 90500 -> gfx950.
 * Minor and step are printed without padding, which is how the target names
 * are spelled (gfx950, gfx1201). */
static void format_gfx_name(unsigned long long version, char *out, size_t size)
{
    const unsigned long long major = version / 10000ull;
    const unsigned long long minor = (version / 100ull) % 100ull;
    const unsigned long long step = version % 100ull;
    snprintf(out, size, "gfx%llu%llu%llu", major, minor, step);
}

int main(void)
{
    const char *topology = getenv("HSA_MODEL_TOPOLOGY");
    if (topology == NULL || topology[0] != '/') {
        fprintf(stderr,
                "rocminfo(model): HSA_MODEL_TOPOLOGY is unset or not absolute; "
                "refusing to report a device\n");
        return 1;
    }

    char nodes_path[PATH_MAX];
    if (snprintf(nodes_path, sizeof(nodes_path), "%s/nodes", topology) >=
        (int)sizeof(nodes_path)) {
        fprintf(stderr, "rocminfo(model): topology path is too long\n");
        return 1;
    }

    DIR *directory = opendir(nodes_path);
    if (directory == NULL) {
        fprintf(stderr, "rocminfo(model): cannot read %s: %s\n", nodes_path,
                strerror(errno));
        return 1;
    }

    struct sagr_node nodes[SAGR_ROCMINFO_MAX_NODES];
    size_t count = 0;
    const struct dirent *entry = NULL;
    while ((entry = readdir(directory)) != NULL && count < SAGR_ROCMINFO_MAX_NODES) {
        if (entry->d_name[0] == '.') {
            continue;
        }
        char *end = NULL;
        errno = 0;
        const unsigned long node_id = strtoul(entry->d_name, &end, 10);
        if (end == entry->d_name || *end != '\0' || errno != 0) {
            continue;
        }
        char properties[PATH_MAX];
        if (snprintf(properties, sizeof(properties), "%s/nodes/%s/properties",
                     topology, entry->d_name) >= (int)sizeof(properties)) {
            continue;
        }
        struct sagr_node node;
        node.node_id = node_id;
        node.simd_count = 0;
        node.gfx_target_version = 0;
        node.cpu_cores_count = 0;
        if (!read_property(properties, "simd_count", &node.simd_count)) {
            continue;
        }
        (void)read_property(properties, "gfx_target_version",
                            &node.gfx_target_version);
        (void)read_property(properties, "cpu_cores_count", &node.cpu_cores_count);
        nodes[count++] = node;
    }
    closedir(directory);

    /* Ascending node order keeps repeated reports stable. */
    for (size_t i = 1; i < count; ++i) {
        const struct sagr_node key = nodes[i];
        size_t j = i;
        while (j > 0 && nodes[j - 1].node_id > key.node_id) {
            nodes[j] = nodes[j - 1];
            --j;
        }
        nodes[j] = key;
    }

    size_t gpu_count = 0;
    for (size_t i = 0; i < count; ++i) {
        if (nodes[i].simd_count > 0) {
            ++gpu_count;
        }
    }
    if (gpu_count == 0) {
        fprintf(stderr,
                "rocminfo(model): topology reports no SIMD-bearing agent\n");
        return 1;
    }

    printf("=====================\n");
    printf("HSA System Attributes\n");
    printf("=====================\n");
    printf("Runtime Version:         1.1\n");
    printf("Simulated:               true\n");
    printf("\n");
    printf("==========\n");
    printf("HSA Agents\n");
    printf("==========\n");

    for (size_t i = 0; i < count; ++i) {
        printf("*******\n");
        printf("Agent %zu\n", i + 1);
        printf("*******\n");
        if (nodes[i].simd_count == 0) {
            printf("  Name:                    CPU\n");
            printf("  Device Type:             CPU\n");
        } else {
            char gfx[32];
            format_gfx_name(nodes[i].gfx_target_version, gfx, sizeof(gfx));
            printf("  Name:                    %s\n", gfx);
            printf("  Marketing Name:          AMD Simulated GPU\n");
            printf("  Device Type:             GPU\n");
            printf("  Compute Unit:            %llu\n", nodes[i].simd_count / 4ull);
            printf("  ISA Info:\n");
            printf("    ISA 1\n");
            printf("      Name:                  amdgcn-amd-amdhsa--%s\n", gfx);
        }
    }
    printf("*** Done ***\n");
    return 0;
}
