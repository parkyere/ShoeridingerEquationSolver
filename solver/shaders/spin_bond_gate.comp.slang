// Heisenberg bond gate exp(+i theta sigma_i.sigma_j) on the exact 2^N spin
// wavefunction, EXACT via sigma.sigma = 2 SWAP - 1: parallel amplitudes get
// a phase, the antiparallel pair mixes. One thread per disjoint 4-amplitude
// block (quarter_n = 2^(N-2) work-items) via the two-zero-bit index
// expansion. CONTRACT: vkcheck check_spin_step.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> data;

struct Params {
    uint quarter_n;  // 2^(N-2) work-items
    uint site_i;
    uint site_j;
    uint pad;
    float4 gate;     // phase.re, phase.im, diag.re, diag.im
    float4 off4;     // off.re, off.im, _, _
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= params.quarter_n) {
        return;
    }
    uint p0 = min(params.site_i, params.site_j);
    uint p1 = max(params.site_i, params.site_j);
    uint low = t & ((1u << p0) - 1u);
    uint mid = (t >> p0) & ((1u << (p1 - p0 - 1u)) - 1u);
    uint high = t >> (p1 - 1u);
    uint base = low | (mid << (p0 + 1u)) | (high << (p1 + 1u));
    // RED STUB: identity (gate not applied yet).
    data[base] = data[base];
}
