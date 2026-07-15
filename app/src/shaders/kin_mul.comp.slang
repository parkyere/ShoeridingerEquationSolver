// Kinetic factor without a phase table: psi[i] *= exp(i * coef * k^2),
// coef = -dt/2, k^2 = kx^2 + ky^2 + kz^2 from three 1D per-axis tables
// (n floats each, FFT frequency order). Replaces a 134 MB complex table
// read with three ~1 KB reads + sincos.
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> psi;
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
    // nx, ny are powers of two (FFT grid), so unflatten i -> (ix,iy,iz) with
    // mask + shift, NOT a runtime integer div/mod: the divisor is a uniform,
    // so the compiler cannot fold %/ into bit ops itself, and integer div/mod
    // is ~20+ cycles on Pascal. Bitwise-identical indices to i%nx / i/nx.
    int lx = int(firstbithigh(uint(params.nx)));
    int ly = int(firstbithigh(uint(params.ny)));
    uint ix = i & (uint(params.nx) - 1u);
    uint iy = (i >> uint(lx)) & (uint(params.ny) - 1u);
    uint iz = i >> uint(lx + ly);
    float a = params.coef * (kx2[ix] + ky2[iy] + kz2[iz]);
    float c = cos(a);
    float s = sin(a);
    float2 z = psi[i];
    psi[i] = float2(z.x * c - z.y * s, z.x * s + z.y * c);
}
