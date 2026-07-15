// psi <- psi - c*phi (complex c). With c = <phi|psi> this projects phi out of
// psi -- the deflation step of relax_deflated. Bindings: psi(0) read-write,
// phi(3) readonly; std140 UBO (binding 1) carries n and the complex
// coefficient c. std140 places the vec2 c at offset 8 (uint n + 4 bytes pad).
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> psi;
[[vk::binding(3, 0)]] StructuredBuffer<float2, Std430DataLayout> phi;

struct Params {
    uint n;
    float2 c;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    float2 q = phi[i];
    float2 c = params.c;
    psi[i] = psi[i] - float2(c.x * q.x - c.y * q.y, c.x * q.y + c.y * q.x);
}
