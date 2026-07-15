// Final composite: accumulated HDR scene + bloom, ACES-approx filmic
// tonemap, and an interleaved-gradient-noise output dither that keeps the
// 8-bit present image banding-free. This writes the ONE image the
// presentation shell samples -- the shell stays a dumb blit.
[[vk::binding(0, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> accum;
[[vk::binding(1, 0)]] Sampler2D<float4> bloom;  // half-res, linear-upsampled
[[vk::binding(2, 0)]] [[vk::image_format("rgba8")]] RWTexture2D<float4> present;

struct Params {
    float bloom_strength;
    float exposure;
    float frame;  // temporal IGN rotation
};
[[vk::binding(3, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

// Krzysztof Narkowicz's ACES filmic approximation.
float3 aces(float3 x) {
    return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0,
                 1.0);
}

float ign(float2 p) {
    p += 5.588238 * params.frame;  // rotate the pattern per frame
    return frac(52.9829189 * frac(0.06711056 * p.x + 0.00583715 * p.y));
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int2 p = int2(gid.xy);
    uint w, h;
    present.GetDimensions(w, h);
    int2 size = int2(w, h);
    if (p.x >= size.x || p.y >= size.y) {
        return;
    }
    float2 uv = (float2(p) + 0.5) / float2(size);
    float3 hdr = accum[p].rgb;
    hdr += params.bloom_strength * bloom.SampleLevel(uv, 0.0).rgb;
    float3 ldr = aces(hdr * params.exposure);
    ldr += (ign(float2(p)) - 0.5) / 255.0;  // kill 8-bit banding
    present[p] = float4(ldr, 1.0);
}
