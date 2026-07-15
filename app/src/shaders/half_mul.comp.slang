// Strang half-kick without a phase table: psi[i] *= exp(i * coef * V[i]),
// coef = -dt/2, phase computed in-shader from the scalar potential (R32).
// Replaces a 134 MB complex table read with a 67 MB scalar read + sincos.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(1, 0)]] StructuredBuffer<float, Std430DataLayout> v;

struct Params {
    uint n;
    float coef;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    psi[i] = psi[i] * cis(params.coef * v[i]);
}
