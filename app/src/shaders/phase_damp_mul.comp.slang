// Fused potential kick + absorber mask: psi[i] *= exp(i * coef * V[i]) * m[i].
// coef = -dt (batch-interior full kick: two adjacent half kicks merged) or
// -dt/2 (the batch-trailing half kick). Diagonal factors commute exactly, so
// the operator equals kick-then-mask; one psi round-trip instead of two.
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> psi;
[[vk::binding(1, 0)]] StructuredBuffer<float, Std430DataLayout> v;
[[vk::binding(2, 0)]] StructuredBuffer<float, Std430DataLayout> m;

struct Params {
    uint n;
    float coef;
};
[[vk::binding(3, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    float a = params.coef * v[i];
    float c = cos(a);
    float s = sin(a);
    float2 z = psi[i];
    float w = m[i];
    psi[i] = w * float2(z.x * c - z.y * s, z.x * s + z.y * c);
}
