// Single-site SU(2) gate on the exact 2^N spin wavefunction: apply the 2x2
// matrix U to qubit `site`, mixing the disjoint amplitude pairs (lo, hi)
// where lo has bit `site` clear. One thread per pair (half_n = 2^(N-1)
// work-items) via the insert-zero-bit index expansion -- no idle threads.
// data: fp32 Complex (float2) amplitudes. CONTRACT: vkcheck check_spin_step.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> data;

struct Params {
    uint half_n;   // 2^(N-1) work-items
    uint site;     // qubit index 0..N-1
    uint pad0;
    uint pad1;
    float4 row0;   // a00.re, a00.im, a01.re, a01.im
    float4 row1;   // a10.re, a10.im, a11.re, a11.im
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= params.half_n) {
        return;
    }
    // Insert a 0 bit at position `site` to build lo (bit clear); hi = lo|bit.
    uint bit = 1u << params.site;
    uint low = t & (bit - 1u);
    uint high = (t & ~(bit - 1u)) << 1u;
    uint lo = high | low;
    uint hi = lo | bit;
    Complex up = data[lo];
    Complex dn = data[hi];
    Complex a00 = Complex(params.row0.xy);
    Complex a01 = Complex(params.row0.zw);
    Complex a10 = Complex(params.row1.xy);
    Complex a11 = Complex(params.row1.zw);
    data[lo] = a00 * up + a01 * dn;
    data[hi] = a10 * up + a11 * dn;
}
