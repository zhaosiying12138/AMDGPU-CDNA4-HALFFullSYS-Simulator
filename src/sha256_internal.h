/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_SHA256_INTERNAL_H
#define SELF_AMDGPU_RUNTIME_SHA256_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

enum { SAGR_SHA256_BYTES = 32, SAGR_SHA256_BLOCK_BYTES = 64 };

typedef struct sagr_sha256_context {
  uint32_t state[8];
  uint64_t byte_count;
  uint8_t block[SAGR_SHA256_BLOCK_BYTES];
  size_t block_size;
} sagr_sha256_context_t;

void sagr_sha256_init(sagr_sha256_context_t *context);
void sagr_sha256_update(sagr_sha256_context_t *context, const void *bytes,
                        size_t size);
void sagr_sha256_final(sagr_sha256_context_t *context,
                       uint8_t digest[SAGR_SHA256_BYTES]);
void sagr_sha256(const void *bytes, size_t size,
                 uint8_t digest[SAGR_SHA256_BYTES]);

#endif
