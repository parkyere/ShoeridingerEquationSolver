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
    // Insert two 0 bits at the two site positions to build `base` (both
    // clear); the block is {base, base|bi, base|bj, base|bi|bj}.
    uint p0 = min(params.site_i, params.site_j);
    uint p1 = max(params.site_i, params.site_j);
    uint low = t & ((1u << p0) - 1u);
    uint mid = (t >> p0) & ((1u << (p1 - p0 - 1u)) - 1u);
    uint high = t >> (p1 - 1u);
    uint base = low | (mid << (p0 + 1u)) | (high << (p1 + 1u));
    uint bi = 1u << params.site_i;
    uint bj = 1u << params.site_j;
    Complex phase = Complex(params.gate.xy);
    Complex diag = Complex(params.gate.zw);
    Complex off = Complex(params.off4.xy);
    // Parallel pairs (both up / both down): a global phase.
    data[base] = phase * data[base];
    data[base | bi | bj] = phase * data[base | bi | bj];
    // Antiparallel mix: a = (i up, j down) = base|bj; b = (i down, j up) =
    // base|bi. new(base|bj)=diag*a+off*b; new(base|bi)=off*a+diag*b.
    Complex a = data[base | bj];
    Complex b = data[base | bi];
    data[base | bj] = diag * a + off * b;
    data[base | bi] = off * a + diag * b;
}
