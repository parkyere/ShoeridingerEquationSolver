// Ehrenfest force <grad V> = sum |psi|^2 grad V: psi (binding 0) folded
// against CENTRAL DIFFERENCES of the scalar potential (binding 4, R32),
// computed in-shader on the periodic grid -- replacing the stored 16 B/cell
// gradient field (4x the potential's footprint) with cache-hot neighbor
// taps. One vec4 partial per workgroup (binding 2); the host sums.
import complex;

[[vk::binding(0, 0)]] StructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(4, 0)]] StructuredBuffer<float, Std430DataLayout> v;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4, Std430DataLayout> partials;

struct Params {
    uint n;
    uint nx;
    uint ny;
    uint nz;
    float4 inv_2h;  // (1/(2hx), 1/(2hy), 1/(2hz), 0)
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared float4 s_acc[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID,
          uint tid : SV_GroupIndex,
          uint3 grp : SV_GroupID) {
    uint stride = WorkgroupCount().x * 256u;
    float4 acc = float4(0.0, 0.0, 0.0, 0.0);
    uint nx = params.nx;
    uint ny = params.ny;
    uint nz = params.nz;
    for (uint i = gid.x; i < params.n; i += stride) {
        float w = norm2(psi[i]);
        uint ix = i % nx;
        uint iy = (i / nx) % ny;
        uint iz = i / (nx * ny);
        uint row = iz * nx * ny + iy * nx;   // x-neighbors: same row
        uint col = iz * nx * ny + ix;        // y-neighbors: same column
        uint pil = iy * nx + ix;             // z-neighbors: same pillar
        float gx = (v[row + (ix + 1u) % nx] -
                    v[row + (ix + nx - 1u) % nx]) * params.inv_2h.x;
        float gy = (v[col + ((iy + 1u) % ny) * nx] -
                    v[col + ((iy + ny - 1u) % ny) * nx]) * params.inv_2h.y;
        float gz = (v[pil + ((iz + 1u) % nz) * nx * ny] -
                    v[pil + ((iz + nz - 1u) % nz) * nx * ny]) * params.inv_2h.z;
        acc += w * float4(gx, gy, gz, 0.0);
    }
    s_acc[tid] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            s_acc[tid] += s_acc[tid + half_len];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        partials[grp.x] = s_acc[0];
    }
}
