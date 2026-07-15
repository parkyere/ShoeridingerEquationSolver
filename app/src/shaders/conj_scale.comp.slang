// psi <- scale * conj(psi). With a forward FFT this composes the inverse
// transform (conj -> FFT -> conj/N). std140 UBO n + scale (binding 1).
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> data;

struct Params {
    uint n;
    float scale;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < params.n) {
        data[i] = params.scale * conj(data[i]);
    }
}
