// psi <- psi - c*phi (complex c). With c = <phi|psi> this projects phi out of
// psi -- the deflation step of relax_deflated. Bindings: psi(0) read-write,
// phi(3) readonly; std140 UBO (binding 1) carries n and the complex
// coefficient c. std140 places the vec2 c at offset 8 (uint n + 4 bytes pad).
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(3, 0)]] StructuredBuffer<Complex, Std430DataLayout> phi;

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
    psi[i] = psi[i] - Complex(params.c) * phi[i];
}
