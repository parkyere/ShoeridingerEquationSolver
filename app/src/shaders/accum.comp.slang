// Temporal accumulation: running average of the HDR scene while nothing
// changes (camera, params, psi all static), snap-reset on any change. The
// sample count rides in .a (the scene alpha is not needed downstream).
// Jittered raymarch + accumulation = hundreds of effective samples on a
// still frame; banding and noise melt away.
[[vk::binding(0, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> scene;
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> accum;

struct Params {
    int accumulate;  // 0 = reset (scene changed), 1 = keep averaging
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int2 p = int2(gid.xy);
    uint w, h;
    accum.GetDimensions(w, h);
    int2 size = int2(w, h);
    if (p.x >= size.x || p.y >= size.y) {
        return;
    }
    float3 cur = scene[p].rgb;
    if (params.accumulate == 0) {
        accum[p] = float4(cur, 1.0);
        return;
    }
    float4 prev = accum[p];
    float n = min(prev.a + 1.0, 512.0);  // cap the window
    float3 avg = prev.rgb + (cur - prev.rgb) / n;
    accum[p] = float4(avg, n);
}
