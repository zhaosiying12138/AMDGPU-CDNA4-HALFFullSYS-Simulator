/* SPDX-License-Identifier: GPL-3.0-or-later */

#include <self_amdgpu_runtime/code_object.h>

#include "sha256_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef SAGR_TRITON_TUTORIAL_PATH
#define SAGR_TRITON_TUTORIAL_PATH ""
#endif
#ifndef SAGR_TRITON_HSACO_PATH
#define SAGR_TRITON_HSACO_PATH ""
#endif

static const char *const TutorialSha256 =
    "842430949e0ccde4fbce07606cce3ac4bac36bf21b2b12619a31b795ca4029b3";
static const char *const HsacoSha256 =
    "ee8b0f892da7ab1886f17ee66f88de5c23e05a48f7f361e02bd0707c9a11826e";

static int
expect(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "triton vecadd compile gate: %s\n", message);
        return 1;
    }
    return 0;
}

static int
read_file(const char *path, uint8_t **bytes, size_t *size)
{
    FILE *file;
    long length;
    size_t read_size;

    if (path == NULL || bytes == NULL || size == NULL || path[0] == '\0')
        return 0;
    file = fopen(path, "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        if (file != NULL)
            fclose(file);
        return 0;
    }
    length = ftell(file);
    if (length <= 0L || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    *size = (size_t)length;
    *bytes = (uint8_t *)malloc(*size);
    if (*bytes == NULL) {
        fclose(file);
        return 0;
    }
    read_size = fread(*bytes, 1U, *size, file);
    fclose(file);
    if (read_size != *size) {
        free(*bytes);
        *bytes = NULL;
        *size = 0U;
        return 0;
    }
    return 1;
}

static void
hex_digest(const uint8_t digest[SAGR_SHA256_BYTES], char output[65])
{
    static const char Hex[] = "0123456789abcdef";
    unsigned index;
    for (index = 0; index < SAGR_SHA256_BYTES; ++index) {
        output[index * 2U] = Hex[digest[index] >> 4U];
        output[index * 2U + 1U] = Hex[digest[index] & 0x0fU];
    }
    output[64] = '\0';
}

static int
check_digest(const uint8_t *bytes, size_t size, const char *expected,
             char actual[65], const char *label)
{
    uint8_t digest[SAGR_SHA256_BYTES];
    sagr_sha256(bytes, size, digest);
    hex_digest(digest, actual);
    return expect(strcmp(actual, expected) == 0, label);
}

int
main(void)
{
    uint8_t *tutorial = NULL;
    uint8_t *image = NULL;
    size_t tutorial_size = 0U;
    size_t image_size = 0U;
    char tutorial_digest[65];
    char image_digest[65];
    sagr_code_object_info_t info;
    sagr_code_object_kernel_info_t kernel;
    sagr_code_object_dispatch_binding_t binding;
    uint8_t code[400];
    uint8_t kernarg[48];
    size_t code_size = 0U;
    uint32_t kernarg_size = 0U;
    int failures = 0;

    if (!read_file(SAGR_TRITON_TUTORIAL_PATH, &tutorial, &tutorial_size) ||
        !read_file(SAGR_TRITON_HSACO_PATH, &image, &image_size)) {
        fprintf(stderr, "triton vecadd compile gate: required CP17 files are missing\n");
        free(tutorial);
        free(image);
        return 77;
    }
    failures += check_digest(tutorial, tutorial_size, TutorialSha256,
                             tutorial_digest, "tutorial source hash changed");
    failures += expect(image_size == 5408U, "HSACO size changed");
    failures += check_digest(image, image_size, HsacoSha256, image_digest,
                             "HSACO hash changed");

    memset(&info, 0, sizeof(info));
    failures += expect(sagr_code_object_validate(
                           image, image_size, &info, (uint32_t)sizeof(info)) ==
                           SAGR_STATUS_SUCCESS,
                       "Triton HSACO metadata validates");
    failures += expect(info.gfx_target == SAGR_CODE_OBJECT_TARGET_GFX950 &&
                           strcmp(info.target,
                                  "amdgcn-amd-amdhsa-unknown-gfx950") == 0 &&
                           info.code_object_version == 5U &&
                           info.kernel_count == 1U &&
                           info.isa_supported_by_gemsim == 0U,
                       "Triton target and unsupported ISA boundary");

    memset(&kernel, 0, sizeof(kernel));
    failures += expect(sagr_code_object_get_kernel(
                           &info, "vecadd", &kernel,
                           (uint32_t)sizeof(kernel)) == SAGR_STATUS_SUCCESS,
                       "vecadd kernel lookup");
    failures += expect(kernel.kernarg_segment_size == 48U &&
                           kernel.kernarg_segment_align == 8U &&
                           kernel.wavefront_size == 64U &&
                           kernel.max_flat_workgroup_size == 256U &&
                           kernel.code_size == 400U &&
                           kernel.descriptor_size == 64U &&
                           kernel.descriptor_kernarg_preload == 12U,
                       "vecadd kernel metadata");

    memset(&binding, 0, sizeof(binding));
    failures += expect(sagr_code_object_describe_dispatch(
                           &info, "vecadd", image_size, &binding,
                           (uint32_t)sizeof(binding)) == SAGR_STATUS_SUCCESS,
                       "vecadd compile binding");
    failures += expect(
        (binding.flags & SAGR_CODE_OBJECT_DISPATCH_FLAG_METADATA_ONLY) != 0U &&
            (binding.flags &
             SAGR_CODE_OBJECT_DISPATCH_FLAG_REQUIRES_CODE_OBJECT_TRANSPORT) !=
                0U &&
            binding.isa_supported_by_gemsim == 0U &&
            binding.requires_explicit_code_object_transport != 0U,
        "compile-only binding remains non-executable");

    memset(code, 0, sizeof(code));
    failures += expect(sagr_code_object_materialize_kernel_code(
                           image, image_size, &binding, code, sizeof(code),
                           &code_size) == SAGR_STATUS_SUCCESS &&
                           code_size == sizeof(code),
                       "vecadd code materialization");
    memset(kernarg, 0, sizeof(kernarg));
    failures += expect(sagr_code_object_pack_kernarg(
                           &kernel, NULL, 0U, kernarg, sizeof(kernarg),
                           &kernarg_size) == SAGR_STATUS_SUCCESS &&
                           kernarg_size == sizeof(kernarg),
                       "zero-initialized vecadd kernarg packing");

    printf("{\"schema\":\"self-amdgpu-runtime.triton-vecadd-compile.v1\","
           "\"tutorial_sha256\":\"%s\",\"hsaco_sha256\":\"%s\","
           "\"hsaco_bytes\":%zu,\"target\":\"%s\","
           "\"kernel\":\"%s\",\"compile_only\":true,"
           "\"provenance_only\":true,\"compiler_invoked\":false,\"jit\":false,"
           "\"launcher\":false,"
           "\"transport\":false,\"execution\":false,"
           "\"fallback\":false}\n",
           tutorial_digest, image_digest, image_size, info.target, kernel.name);
    free(tutorial);
    free(image);
    return failures == 0 ? 0 : 1;
}
