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
import complex;
import ylm;
import grid;

[[vk::binding(0, 0)]] StructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(6, 0)]] StructuredBuffer<uint, Std430DataLayout> sorted_cell;
[[vk::binding(7, 0)]] StructuredBuffer<uint, Std430DataLayout> bin_off;
[[vk::binding(8, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> g_lm;

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
        int3 ijk = unflatten_moddiv(cell, params.nx, params.ny);
        float3 pos = cell_pos(ijk, params.box_min.xyz, params.cell_h.xyz);
        float x = pos.x;
        float y = pos.y;
        float z = pos.z;
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
        Complex pv = psi[cell] * (w * params.dv);
        for (int l = 0; l <= 5; ++l) {
            for (int m = -l; m <= l; ++m) {
                float Y = real_Ylm(l, m, x, y, z);
                uint slot = uint(l * l + (l + m)) * 128u + tid;
                acc_re[slot] += Y * pv.re;
                acc_im[slot] += Y * pv.im;
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
            g_lm[uint(c) * uint(params.nr) + uint(bin)] = Complex(sr, si);
        }
    }
}
