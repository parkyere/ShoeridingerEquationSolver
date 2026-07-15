// Unpack fp16 storage (src uint per cell, binding 6) back to a complex fp32
// field (dst, binding 0) via unpackHalf2x16 -- the inverse of pack_half. The
// GLSL builtin maps to GLSL.std.450 UnpackHalf2x16 (no shaderFloat16 needed).
// std140 UBO n (1).
import glsl;

[[vk::binding(6, 0)]] StructuredBuffer<uint, Std430DataLayout> src;
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> dst;

struct Params {
    uint n;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < params.n) {
        dst[i] = unpackHalf2x16(src[i]);
    }
}
