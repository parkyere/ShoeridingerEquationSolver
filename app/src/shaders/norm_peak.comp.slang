// Fixed-order tree reduction of |psi|^2 into one (sum, max) partial per
// workgroup; the host finishes the per-workgroup partials in double.
// Bindings: psi (readonly, 0), partials (writeonly, 2), std140 UBO n (1).
// No atomics, so the fixed-order fold stays deterministic (bit-exactness
// checks rely on this). partials pack (sum, max) -- not a complex number.
import complex;

[[vk::binding(0, 0)]] StructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(2, 0)]] RWStructuredBuffer<float2, Std430DataLayout> partials;

struct Params {
    uint n;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared float s_sum[256];
groupshared float s_max[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID,
          uint tid : SV_GroupIndex,
          uint3 grp : SV_GroupID) {
    uint stride = WorkgroupCount().x * 256u;
    float acc = 0.0;
    float mx = 0.0;
    for (uint i = gid.x; i < params.n; i += stride) {
        float d = norm2(psi[i]);
        acc += d;
        mx = max(mx, d);
    }
    s_sum[tid] = acc;
    s_max[tid] = mx;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            s_sum[tid] += s_sum[tid + half_len];
            s_max[tid] = max(s_max[tid], s_max[tid + half_len]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        partials[grp.x] = float2(s_sum[0], s_max[0]);
    }
}
