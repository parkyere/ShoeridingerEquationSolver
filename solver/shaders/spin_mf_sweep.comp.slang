// One checkerboard-parity mean-field sweep: B_eff = B_ext - 2 J sum_nn <sigma>,
// project out the n-parallel part (no torque), exact SU(2) step, then Gilbert
// damping. Same-parity sites are non-adjacent -> each writes only its own site
// and reads opposite-parity neighbours, so no intra-dispatch race. Bit-faithful
// to ses::spinlattice_step. CONTRACT: vkcheck check_spin_mf.
import complex;
import spin;

[[vk::binding(0, 0)]] RWStructuredBuffer<float4, Std430DataLayout> spinors;
[[vk::binding(1, 0)]] RWStructuredBuffer<float, Std430DataLayout> bloch;

struct Params {
    float bx, by, bz, j;
    float alpha, h, pad0, pad1;
    uint parity, nx, ny, pad2;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

float3 nbloch(uint q) {
    return float3(bloch[3u * q + 0u], bloch[3u * q + 1u], bloch[3u * q + 2u]);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.nx * params.ny) {
        return;
    }
    uint x = i % params.nx;
    uint y = i / params.nx;
    if (((x + y) & 1u) != params.parity) {
        return;
    }
    float3 eff = float3(params.bx, params.by, params.bz);
    if (x + 1u < params.nx) eff -= 2.0 * params.j * nbloch(i + 1u);
    if (x >= 1u) eff -= 2.0 * params.j * nbloch(i - 1u);
    if (y + 1u < params.ny) eff -= 2.0 * params.j * nbloch(i + params.nx);
    if (y >= 1u) eff -= 2.0 * params.j * nbloch(i - params.nx);

    float3 ni = nbloch(i);
    float3 tf = eff - dot(ni, eff) * ni;  // torque field (perp to n)
    Spinor s = load_spinor(spinors[i]);
    s = spin_step(s, tf, params.h);
    float3 p = bloch_of(s);
    if (params.alpha > 0.0) {
        float pb = dot(p, eff);
        p += params.alpha * params.h * (p * pb - eff);
        float nn = length(p);
        if (nn > 1e-12) {
            p /= nn;
            s = spinor_from_bloch(p);
        }
    }
    spinors[i] = store_spinor(s);
    bloch[3u * i + 0u] = p.x;
    bloch[3u * i + 1u] = p.y;
    bloch[3u * i + 2u] = p.z;
}
