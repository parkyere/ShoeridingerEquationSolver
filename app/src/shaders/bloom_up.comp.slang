// Dual-filter upsample: a 3x3 tent over the smaller mip, ADDED into the
// larger mip (progressive upsampling accumulates every scale's glow).
[[vk::binding(0, 0)]] Sampler2D<float4> src;  // smaller mip (combined sampler)
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> dst;

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
    float2 t = 1.0 / float2(sw, sh);

    float3 sum = src.SampleLevel(uv, 0.0).rgb * 4.0;
    sum += src.SampleLevel(uv + float2(t.x, 0.0), 0.0).rgb * 2.0;
    sum += src.SampleLevel(uv + float2(-t.x, 0.0), 0.0).rgb * 2.0;
    sum += src.SampleLevel(uv + float2(0.0, t.y), 0.0).rgb * 2.0;
    sum += src.SampleLevel(uv + float2(0.0, -t.y), 0.0).rgb * 2.0;
    sum += src.SampleLevel(uv + float2(t.x, t.y), 0.0).rgb;
    sum += src.SampleLevel(uv + float2(-t.x, t.y), 0.0).rgb;
    sum += src.SampleLevel(uv + float2(t.x, -t.y), 0.0).rgb;
    sum += src.SampleLevel(uv + float2(-t.x, -t.y), 0.0).rgb;

    float3 up = sum / 16.0;
    float3 cur = dst[p].rgb;
    dst[p] = float4(cur + up, 1.0);
}
