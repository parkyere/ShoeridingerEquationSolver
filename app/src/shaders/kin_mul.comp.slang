// Kinetic factor without a phase table: psi[i] *= exp(i * coef * k^2),
// coef = -dt/2, k^2 = kx^2 + ky^2 + kz^2 from three 1D per-axis tables
// (n floats each, FFT frequency order). Replaces a 134 MB complex table
// read with three ~1 KB reads + sincos.
import complex;
import grid;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(1, 0)]] StructuredBuffer<float, Std430DataLayout> kx2;
[[vk::binding(2, 0)]] StructuredBuffer<float, Std430DataLayout> ky2;
[[vk::binding(3, 0)]] StructuredBuffer<float, Std430DataLayout> kz2;

struct Params {
    uint n;
    int nx;
    int ny;
    float coef;
};
[[vk::binding(4, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    // nx, ny powers of two: mask+shift unflatten (grid module), the per-axis
    // k^2 table lookups summed left-to-right exactly as before.
    int3 k = unflatten_pow2(i, params.nx, params.ny);
    float a = params.coef * (kx2[k.x] + ky2[k.y] + kz2[k.z]);
    psi[i] = psi[i] * cis(a);
}
