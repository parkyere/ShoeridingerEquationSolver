// psi <- phase * psi: element-wise complex multiply against a precomputed
// phase table (split-step phase factor). One thread per element.
// Bindings: psi(0) read-write, phase(1) readonly, std140 UBO n(2).
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> psi;
[[vk::binding(1, 0)]] StructuredBuffer<float2, Std430DataLayout> phase;

struct Params {
    uint n;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    float2 a = psi[i];
    float2 b = phase[i];
    psi[i] = float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}
