// Synthesize psi = (u(|r|)/|r|) * Y_lm(theta,phi) per cell, from a radial
// u_nl(r) table, using the real spherical harmonics (l = 0..5). std140 UBO
// (binding 1) carries the grid + orbital params; vec3 geometry is vec4-padded.
// psi out (binding 0), radial table u (binding 5).
import complex;
import ylm;
import grid;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(5, 0)]] StructuredBuffer<float, Std430DataLayout> u;

struct Params {
    uint n;
    int nx;
    int ny;
    int l;
    float4 box_min;  // .xyz
    float4 cell_h;   // .xyz
    int m;
    int n_radial;
    float h_radial;  // radial spacing rmax/(n_radial+1)
    float rmax;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) {
        return;
    }
    int3 ijk = unflatten_moddiv(idx, params.nx, params.ny);
    float3 r3 = cell_pos(ijk, params.box_min.xyz, params.cell_h.xyz);
    float r = length(r3);
    float u_over_r = 0.0;
    if (r < params.h_radial) {
        u_over_r = u[0] / params.h_radial;      // u(0)=0 pinned: l=0 limit u_0/h
    } else if (r < params.rmax) {
        float t = r / params.h_radial - 1.0;    // r_i = (i+1) h
        int i0 = int(t);
        float frac = t - float(i0);
        float ui = (i0 + 1 < params.n_radial) ? (1.0 - frac) * u[i0] + frac * u[i0 + 1]
                                              : (1.0 - frac) * u[i0];  // outer: u(rmax)=0
        u_over_r = ui / r;
    }
    psi[idx] = Complex(u_over_r * real_Ylm(params.l, params.m, r3.x, r3.y, r3.z), 0.0);
}
