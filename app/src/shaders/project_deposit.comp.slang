// Orbital-free angular projection deposit. One workgroup per radial bin
// gathers that bin's cells (primary + secondary interpolation partners, from
// the counting-sort CSR index) and accumulates Y_lm-weighted psi into 36
// components (l = 0..5), then folds each component with a WaveActiveSum + a
// fixed-order combine of the per-wave partials, written to g_lm[c*nr+bin].
// No atomics; the wave fold is order-agnostic within a wave but the per-wave
// partials sum in a fixed order -> deterministic (P3): the projection oracle
// compares within tolerance, not bit-exact.
//
// GLSL used gl_NumSubgroups/gl_SubgroupID for the per-subgroup slotting; Slang
// derives the same from WaveGetLaneCount() + SV_GroupIndex, giving each wave's
// elected lane a UNIQUE contiguous slot (0..numSg-1). Requires wave arithmetic
// (asserted by vkcheck check_device_features).
[[vk::binding(0, 0)]] StructuredBuffer<float2, Std430DataLayout> psi;
[[vk::binding(6, 0)]] StructuredBuffer<uint, Std430DataLayout> sorted_cell;
[[vk::binding(7, 0)]] StructuredBuffer<uint, Std430DataLayout> bin_off;
[[vk::binding(8, 0)]] RWStructuredBuffer<float2, Std430DataLayout> g_lm;

struct Params {
    int nx;
    int ny;
    int nr;
    float h_radial;
    float4 box_min;  // .xyz
    float4 cell_h;   // .xyz
    float dv;
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

// Transposed, re/im-split accumulators: [c*128 + tid] puts consecutive
// threads in consecutive 4 B banks. Same 36 KB total, unchanged reduction
// order. Per-wave partials for the cross-wave combine: 128 threads / the
// smallest supported wave (4) => at most 32 waves.
groupshared float acc_re[36 * 128];
groupshared float acc_im[36 * 128];
groupshared float sg_re[36 * 32];
groupshared float sg_im[36 * 32];

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 grp : SV_GroupID, uint tid : SV_GroupIndex) {
    int bin = int(grp.x);
    if (bin >= params.nr) {
        return;
    }
    for (int c = 0; c < 36; ++c) {
        acc_re[uint(c) * 128u + tid] = 0.0;
        acc_im[uint(c) * 128u + tid] = 0.0;
    }
    uint p_beg = bin_off[bin];
    uint p_end = bin_off[bin + 1];
    uint s_beg = (bin > 0) ? bin_off[bin - 1] : 0u;
    uint nP = p_end - p_beg;
    uint nS = (bin > 0) ? (bin_off[bin] - s_beg) : 0u;
    uint total = nP + nS;
    for (uint e = tid; e < total; e += 128u) {
        bool primary = (e < nP);
        uint cell = primary ? sorted_cell[p_beg + e] : sorted_cell[s_beg + (e - nP)];
        int i = int(cell) % params.nx;
        int j = (int(cell) / params.nx) % params.ny;
        int k = int(cell) / (params.nx * params.ny);
        float x = params.box_min.x + float(i) * params.cell_h.x;
        float y = params.box_min.y + float(j) * params.cell_h.y;
        float z = params.box_min.z + float(k) * params.cell_h.z;
        float r = sqrt(x * x + y * y + z * z);
        float w;
        if (primary) {
            if (r < params.h_radial) {
                w = 1.0 / params.h_radial;  // origin: constant u[0]/h (bin 0 only)
            } else {
                float t = r / params.h_radial - 1.0;
                float frac = t - float(int(t));
                w = (1.0 - frac) / r;
            }
        } else {
            if (r < params.h_radial) {
                continue;  // origin has no secondary (i0+1) contribution
            }
            float t = r / params.h_radial - 1.0;
            float frac = t - float(int(t));
            w = frac / r;
        }
        float2 pv = psi[cell] * (w * params.dv);
        for (int l = 0; l <= 5; ++l) {
            for (int m = -l; m <= l; ++m) {
                float Y = real_Ylm(l, m, x, y, z);
                uint slot = uint(l * l + (l + m)) * 128u + tid;
                acc_re[slot] += Y * pv.x;
                acc_im[slot] += Y * pv.y;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    // Each component: reduce the 128 partials with one WaveActiveSum per wave
    // (the wave's first lane banks its sum in slot = tid/waveSize), then thread
    // 0 folds the per-wave sums in a fixed order. All 128 lanes are active here
    // (the bin guard returned uniformly), so the wave ops stay convergent.
    uint waveSize = WaveGetLaneCount();
    uint numSg = (128u + waveSize - 1u) / waveSize;
    uint sgId = tid / waveSize;
    for (int c = 0; c < 36; ++c) {
        float vr = WaveActiveSum(acc_re[uint(c) * 128u + tid]);
        float vi = WaveActiveSum(acc_im[uint(c) * 128u + tid]);
        if (WaveIsFirstLane()) {
            sg_re[uint(c) * numSg + sgId] = vr;
            sg_im[uint(c) * numSg + sgId] = vi;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid == 0u) {
        for (int c = 0; c < 36; ++c) {
            float sr = 0.0;
            float si = 0.0;
            for (uint s = 0u; s < numSg; ++s) {
                sr += sg_re[uint(c) * numSg + s];
                si += sg_im[uint(c) * numSg + s];
            }
            g_lm[uint(c) * uint(params.nr) + uint(bin)] = float2(sr, si);
        }
    }
}
