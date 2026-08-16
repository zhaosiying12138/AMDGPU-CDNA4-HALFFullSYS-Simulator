/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_MANAGED_SESSION_INTERNAL_H
#define SELF_AMDGPU_RUNTIME_MANAGED_SESSION_INTERNAL_H

#include <self_amdgpu_runtime/runtime.h>

/*
 * Drop a fork-inherited local session copy without sending remote cleanup or
 * terminating the parent-owned simulator process. This is an internal helper;
 * public callers use sagr_provider_discard_inherited().
 */
sagr_status_t sagr_managed_session_discard_inherited(
    sagr_managed_session_t *session);

#endif
