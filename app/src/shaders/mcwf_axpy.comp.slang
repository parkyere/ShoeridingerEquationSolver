// Fused MCWF no-jump update: psi[i] += sum_s (cre + i cim)_s * R_s(i) where
// R_s = (u_s(|r|)/|r|) * Y_{l_s m_s} * inv_norm_s -- the normalized
// eigenstate evaluated in-register. Replaces the per-state synth ->
// normalize -> axpy chain (3 full psi round-trips per state) with ONE pass;
// inv_norm comes from the caller's cached grid norm (a constant for a fixed
// radial table). Radial interpolation matches synth.comp exactly.
// psi (binding 0, rw), UBO (binding 1), slotted radial tables (binding 5).
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> psi;
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

static const float PI = 3.14159265358979323846;

float real_Ylm(int l, int m, float x, float y, float z) {
    float r2 = x * x + y * y + z * z;
    if (l == 0) return 1.0 / (2.0 * sqrt(PI));
    if (r2 == 0.0) return 0.0;
    if (l == 1) {
        float c = sqrt(3.0 / (4.0 * PI)) / sqrt(r2);
        if (m == -1) return c * y;
        if (m == 0) return c * z;
        return c * x;
    }
    if (l == 2) {
        float c = sqrt(15.0 / PI) / r2;
        if (m == -2) return 0.5 * c * x * y;
        if (m == -1) return 0.5 * c * y * z;
        if (m == 0) return 0.25 * sqrt(5.0 / PI) * (3.0 * z * z - r2) / r2;
        if (m == 1) return 0.5 * c * z * x;
        return 0.25 * c * (x * x - y * y);
    }
    if (l == 3) {
        float r3 = r2 * sqrt(r2);
        if (m == -3) return 0.25 * sqrt(35.0 / (2.0 * PI)) * y * (3.0 * x * x - y * y) / r3;
        if (m == -2) return 0.5 * sqrt(105.0 / PI) * x * y * z / r3;
        if (m == -1) return 0.25 * sqrt(21.0 / (2.0 * PI)) * y * (5.0 * z * z - r2) / r3;
        if (m == 0) return 0.25 * sqrt(7.0 / PI) * z * (5.0 * z * z - 3.0 * r2) / r3;
        if (m == 1) return 0.25 * sqrt(21.0 / (2.0 * PI)) * x * (5.0 * z * z - r2) / r3;
        if (m == 2) return 0.25 * sqrt(105.0 / PI) * z * (x * x - y * y) / r3;
        return 0.25 * sqrt(35.0 / (2.0 * PI)) * x * (x * x - 3.0 * y * y) / r3;
    }
    if (l == 4) {
        float r4 = r2 * r2;
        if (m == -4) return 0.75 * sqrt(35.0 / PI) * x * y * (x * x - y * y) / r4;
        if (m == -3) return 0.75 * sqrt(35.0 / (2.0 * PI)) * y * z * (3.0 * x * x - y * y) / r4;
        if (m == -2) return 0.75 * sqrt(5.0 / PI) * x * y * (7.0 * z * z - r2) / r4;
        if (m == -1) return 0.75 * sqrt(5.0 / (2.0 * PI)) * y * z * (7.0 * z * z - 3.0 * r2) / r4;
        if (m == 0) return (3.0 / 16.0) * sqrt(1.0 / PI) *
                           (35.0 * z * z * z * z - 30.0 * z * z * r2 + 3.0 * r2 * r2) / r4;
        if (m == 1) return 0.75 * sqrt(5.0 / (2.0 * PI)) * x * z * (7.0 * z * z - 3.0 * r2) / r4;
        if (m == 2) return 0.375 * sqrt(5.0 / PI) * (x * x - y * y) * (7.0 * z * z - r2) / r4;
        if (m == 3) return 0.75 * sqrt(35.0 / (2.0 * PI)) * x * z * (x * x - 3.0 * y * y) / r4;
        return (3.0 / 16.0) * sqrt(35.0 / PI) *
               (x * x * x * x - 6.0 * x * x * y * y + y * y * y * y) / r4;
    }
    // l == 5
    float r5 = r2 * r2 * sqrt(r2);
    float x2 = x * x;
    float y2 = y * y;
    float z2 = z * z;
    float zpoly = 21.0 * z2 * z2 - 14.0 * z2 * r2 + r2 * r2;
    if (m == -5) return (1.0 / 32.0) * sqrt(1386.0 / PI) *
                        y * (5.0 * x2 * x2 - 10.0 * x2 * y2 + y2 * y2) / r5;
    if (m == -4) return (1.0 / 16.0) * sqrt(3465.0 / PI) * 4.0 * x * y * (x2 - y2) * z / r5;
    if (m == -3) return (1.0 / 32.0) * sqrt(770.0 / PI) * y * (3.0 * x2 - y2) * (9.0 * z2 - r2) / r5;
    if (m == -2) return (1.0 / 8.0) * sqrt(1155.0 / PI) * 2.0 * x * y * z * (3.0 * z2 - r2) / r5;
    if (m == -1) return (1.0 / 16.0) * sqrt(165.0 / PI) * y * zpoly / r5;
    if (m == 0) return (1.0 / 16.0) * sqrt(11.0 / PI) *
                       z * (63.0 * z2 * z2 - 70.0 * z2 * r2 + 15.0 * r2 * r2) / r5;
    if (m == 1) return (1.0 / 16.0) * sqrt(165.0 / PI) * x * zpoly / r5;
    if (m == 2) return (1.0 / 8.0) * sqrt(1155.0 / PI) * (x2 - y2) * z * (3.0 * z2 - r2) / r5;
    if (m == 3) return (1.0 / 32.0) * sqrt(770.0 / PI) * x * (x2 - 3.0 * y2) * (9.0 * z2 - r2) / r5;
    if (m == 4) return (1.0 / 16.0) * sqrt(3465.0 / PI) *
                       (x2 * x2 - 6.0 * x2 * y2 + y2 * y2) * z / r5;
    return (1.0 / 32.0) * sqrt(1386.0 / PI) * x * (x2 * x2 - 10.0 * x2 * y2 + 5.0 * y2 * y2) / r5;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) {
        return;
    }
    int i = int(idx) % params.nx;
    int j = (int(idx) / params.nx) % params.ny;
    int k = int(idx) / (params.nx * params.ny);
    float3 p = params.box_min.xyz +
               float3(float(i), float(j), float(k)) * params.cell_h.xyz;
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
    float2 add = float2(0.0, 0.0);
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
        add += params.coef[s].xy * R;
    }
    psi[idx] = psi[idx] + add;
}
