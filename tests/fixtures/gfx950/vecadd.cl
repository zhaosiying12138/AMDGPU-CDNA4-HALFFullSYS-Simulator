// SPDX-License-Identifier: GPL-3.0-or-later

kernel void
vecadd(global const float *a, global const float *b, global float *c, uint n)
{
    size_t index = get_global_id(0);
    if (index < n)
        c[index] = a[index] + b[index];
}
