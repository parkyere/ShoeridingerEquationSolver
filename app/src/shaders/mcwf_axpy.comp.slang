// Fused MCWF no-jump update: psi[i] += sum_s (cre + i cim)_s * R_s(i) where
// R_s = (u_s(|r|)/|r|) * Y_{l_s m_s} * inv_norm_s -- the normalized
// eigenstate evaluated in-register. Replaces the per-state synth ->
// normalize -> axpy chain (3 full psi round-trips per state) with ONE pass;
// inv_norm comes from the caller's cached grid norm (a constant for a fixed
// radial table). Radial interpolation matches synth.comp exactly.
// psi (binding 0, rw), UBO (binding 1), slotted radial tables (binding 5).
import complex;
import ylm;
import grid;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(5, 0)]] StructuredBuffer<float, Std430DataLayout> u;

struct Params {
    uint n;
    int nx;
    int ny;
    int n_states;
    float4 box_min;  // .xyz
    float4 cell_h;   // .xyz
    int n_radial;    // slot stride in floats
    float h_radial;
    float rmax;
    float pad0;
    int4 lm[8];      // .x = l, .y = m
    float4 coef[8];  // .x = cre, .y = cim, .z = inv_norm
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
    float3 p = cell_pos(ijk, params.box_min.xyz, params.cell_h.xyz);
    float r = length(p);
    // Radial interpolation geometry, shared by every state (synth.comp's
    // formulas verbatim; per-state only the table values differ).
    bool origin = (r < params.h_radial);
    bool inside = (r < params.rmax);
    int i0 = 0;
    float frac = 0.0;
    if (!origin && inside) {
        float t = r / params.h_radial - 1.0;  // r_i = (i+1) h
        i0 = int(t);
        frac = t - float(i0);
    }
    Complex add = Complex(0.0, 0.0);
    for (int s = 0; s < params.n_states; ++s) {
        int base = s * params.n_radial;
        float uor;
        if (origin) {
            uor = u[base] / params.h_radial;  // u(0)=0 pinned: l=0 limit u_0/h
        } else if (inside) {
            float ua = u[base + i0];
            float ub = (i0 + 1 < params.n_radial) ? u[base + i0 + 1] : 0.0;
            float ui = (1.0 - frac) * ua + frac * ub;  // outer: u(rmax)=0
            uor = ui / r;
        } else {
            continue;
        }
        float R = uor * real_Ylm(params.lm[s].x, params.lm[s].y, p.x, p.y, p.z) *
                  params.coef[s].z;
        add = add + Complex(params.coef[s].xy) * R;
    }
    psi[idx] = psi[idx] + add;
}
