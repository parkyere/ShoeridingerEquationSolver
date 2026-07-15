// Dual-filter (Kawase) downsample for the bloom chain: 4 diagonal taps + a
// weighted center, sampled with LINEAR filtering half a texel out so each
// tap integrates 4 texels. The first pass in the chain also applies a soft
// bright-pass threshold so only the luminous cloud core blooms.
[[vk::binding(0, 0)]] Sampler2D<float4> src;  // combined image sampler
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> dst;

struct Params {
    int bright_pass;
    float threshold;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int2 p = int2(gid.xy);
    uint w, h;
    dst.GetDimensions(w, h);
    int2 size = int2(w, h);
    if (p.x >= size.x || p.y >= size.y) {
        return;
    }
    float2 uv = (float2(p) + 0.5) / float2(size);
    uint sw, sh;
    src.GetDimensions(sw, sh);
    float2 half_texel = 0.5 / float2(sw, sh);

    float3 sum = src.SampleLevel(uv, 0.0).rgb * 4.0;
    sum += src.SampleLevel(uv + float2(-half_texel.x, -half_texel.y), 0.0).rgb;
    sum += src.SampleLevel(uv + float2(half_texel.x, -half_texel.y), 0.0).rgb;
    sum += src.SampleLevel(uv + float2(-half_texel.x, half_texel.y), 0.0).rgb;
    sum += src.SampleLevel(uv + float2(half_texel.x, half_texel.y), 0.0).rgb;
    float3 c = sum / 8.0;

    if (params.bright_pass != 0) {
        // Soft knee: keep everything above the threshold, feather below.
        float lum = max(c.r, max(c.g, c.b));
        float wgt = max(lum - params.threshold, 0.0);
        wgt = wgt / max(lum, 1e-4);
        c *= wgt;
    }
    dst[p] = float4(c, 1.0);
}
