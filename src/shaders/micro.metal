// micro.metal: microbenchmark and smoke-test kernels.

// Phase 0 hello world: c = a + b. The explicit count + bounds check is the
// project-wide rule: the grid may be rounded up past the end of the buffers.
kernel void vector_add(device const float* a [[buffer(0)]],
                       device const float* b [[buffer(1)]],
                       device float*       c [[buffer(2)]],
                       constant uint&      n [[buffer(3)]],
                       uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    c[gid] = a[gid] + b[gid];
}
