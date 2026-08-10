// SPDX-License-Identifier: GPL-3.0-or-later

__kernel void
vecadd(__global const float *a, __global const float *b, __global float *c,
       const uint n)
{
    const size_t index = get_global_id(0);
    if (index < n)
        c[index] = a[index] + b[index];
}
