// Strang half-kick without a phase table: psi[i] *= exp(i * coef * V[i]),
// coef = -dt/2, phase computed in-shader from the scalar potential (R32).
// Replaces a 134 MB complex table read with a 67 MB scalar read + sincos.
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> psi;
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
    float a = params.coef * v[i];
    float c = cos(a);
    float s = sin(a);
    float2 z = psi[i];
    psi[i] = float2(z.x * c - z.y * s, z.x * s + z.y * c);
}
