// Shared-memory radix-2 line FFT at N=8 (NHALF=4, LOG2N=3), the small size
// used by the 3D fft3 check: run once per axis of an 8x8x8 cube. The line-
// enumeration uniforms (mod_a/mul_b/mul_c/stride/n_lines) come from a std140
// UBO and select the axis (x/y/z). See fft_line64.comp for the stage layout.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> data;

struct Params {
    int mod_a;
    int mul_b;
    int mul_c;
    int stride;
    int n_lines;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared Complex line_sm[8];

static const float kTwoPi = 6.28318530717958647692;

uint bit_reverse(uint v) { return reversebits(v) >> (32u - 3u); }

[shader("compute")]
[numthreads(4, 1, 1)]
void main(uint3 grp : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    int l = int(grp.x);
    if (l >= params.n_lines) {
        return;
    }
    int base = (l % params.mod_a) * params.mul_b + (l / params.mod_a) * params.mul_c;
    uint t = lid.x;
    uint i0 = t;
    uint i1 = t + 4u;

    line_sm[i0] = data[base + int(bit_reverse(i0)) * params.stride];
    line_sm[i1] = data[base + int(bit_reverse(i1)) * params.stride];
    GroupMemoryBarrierWithGroupSync();

    for (uint len = 2u; len <= 8u; len <<= 1u) {
        uint half_len = len >> 1u;
        uint j = t % half_len;
        uint pos = (t / half_len) * len + j;
        Complex w = cis(-kTwoPi * float(j) / float(len));
        Complex u = line_sm[pos];
        Complex v = line_sm[pos + half_len] * w;
        line_sm[pos] = u + v;
        line_sm[pos + half_len] = u - v;
        GroupMemoryBarrierWithGroupSync();
    }

    data[base + int(i0) * params.stride] = line_sm[i0];
    data[base + int(i1) * params.stride] = line_sm[i1];
}
