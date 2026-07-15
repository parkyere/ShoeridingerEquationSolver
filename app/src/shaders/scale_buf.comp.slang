// P4: psi <- renorm[0] * psi. Identical to scale.comp except the factor is
// GPU-computed by norm_finalize in the same submit (fused relax renorm) and
// read from an SSBO, not a host-set UBO -- so the whole renorm records into
// one command buffer.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> data;
[[vk::binding(2, 0)]] StructuredBuffer<float, Std430DataLayout> renorm;

struct Params {
    uint n;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < params.n) {
        data[i] = renorm[0] * data[i];
    }
}
