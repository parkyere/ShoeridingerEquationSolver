// Complex inner product <phi|psi> = sum conj(phi)*psi, tree-reduced into one
// complex partial per workgroup; the host sums the partials.
// Bindings: psi (0), partials (2), phi (3), std140 UBO n (1).
import complex;

[[vk::binding(0, 0)]] StructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(3, 0)]] StructuredBuffer<Complex, Std430DataLayout> phi;
[[vk::binding(2, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> partials;

struct Params {
    uint n;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared Complex s_sum[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID,
          uint tid : SV_GroupIndex,
          uint3 grp : SV_GroupID) {
    uint stride = WorkgroupCount().x * 256u;
    Complex acc = Complex(0.0, 0.0);
    for (uint i = gid.x; i < params.n; i += stride) {
        acc = acc + mul_conj(phi[i], psi[i]);
    }
    s_sum[tid] = acc;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            s_sum[tid] = s_sum[tid] + s_sum[tid + half_len];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        partials[grp.x] = s_sum[0];
    }
}
