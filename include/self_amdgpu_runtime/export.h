/* SPDX-License-Identifier: GPL-3.0-or-later */

#ifndef SELF_AMDGPU_RUNTIME_EXPORT_H
#define SELF_AMDGPU_RUNTIME_EXPORT_H

#if defined(SELF_AMDGPU_RUNTIME_STATIC_DEFINE)
#  define SAGR_API
#elif defined(_WIN32) || defined(__CYGWIN__)
#  if defined(SELF_AMDGPU_RUNTIME_BUILDING_LIBRARY)
#    define SAGR_API __declspec(dllexport)
#  else
#    define SAGR_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define SAGR_API __attribute__((visibility("default")))
#else
#  define SAGR_API
#endif

#endif
