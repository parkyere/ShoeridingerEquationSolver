// Shared-memory radix-2 line FFT at N=256 (NHALF=128, LOG2N=8), the size for
// a 256^3 app grid. The SPIR-V is baked offline at a fixed size, so N is a
// concrete constant here. One workgroup of NHALF threads handles one line;
// the std140 UBO's line-enumeration uniforms select the lines (axis).
// See fft_line64.comp for the stage layout. A standalone 1-D FFT check drives
// it with n_lines=1, stride=1, base=0.
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> data;

struct Params {
    int mod_a;
    int mul_b;
    int mul_c;
    int stride;
    int n_lines;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared float2 line_sm[256];

static const float kTwoPi = 6.28318530717958647692;

uint bit_reverse(uint v) { return reversebits(v) >> (32u - 8u); }

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 grp : SV_GroupID, uint3 lid : SV_GroupThreadID) {
    int l = int(grp.x);
    if (l >= params.n_lines) {
        return;
    }
    int base = (l % params.mod_a) * params.mul_b + (l / params.mod_a) * params.mul_c;
    uint t = lid.x;
    uint i0 = t;
    uint i1 = t + 128u;

    line_sm[i0] = data[base + int(bit_reverse(i0)) * params.stride];
    line_sm[i1] = data[base + int(bit_reverse(i1)) * params.stride];
    GroupMemoryBarrierWithGroupSync();

    for (uint len = 2u; len <= 256u; len <<= 1u) {
        uint half_len = len >> 1u;
        uint j = t % half_len;
        uint pos = (t / half_len) * len + j;
        float ang = -kTwoPi * float(j) / float(len);
        float2 w = float2(cos(ang), sin(ang));
        float2 u = line_sm[pos];
        float2 q = line_sm[pos + half_len];
        float2 v = float2(q.x * w.x - q.y * w.y, q.x * w.y + q.y * w.x);
        line_sm[pos] = u + v;
        line_sm[pos + half_len] = u - v;
        GroupMemoryBarrierWithGroupSync();
    }

    data[base + int(i0) * params.stride] = line_sm[i0];
    data[base + int(i1) * params.stride] = line_sm[i1];
}
