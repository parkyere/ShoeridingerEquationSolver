// Pack a complex fp32 field (src, binding 0) to fp16 storage (dst uint per
// cell, binding 6) via packHalf2x16 -- halves the resident-state atlas
// footprint versus fp32. The GLSL builtin (Slang's glsl module) maps to the
// GLSL.std.450 PackHalf2x16 op, so it needs NO shaderFloat16 capability
// (f32tof16 would emit one). std140 UBO n (1).
import glsl;

[[vk::binding(0, 0)]] StructuredBuffer<float2, Std430DataLayout> src;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint, Std430DataLayout> dst;

struct Params {
    uint n;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < params.n) {
        dst[i] = packHalf2x16(src[i]);
    }
}
