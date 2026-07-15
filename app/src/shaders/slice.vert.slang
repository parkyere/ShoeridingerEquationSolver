// Cross-section quad: a plane through the nucleus, normal along slice.y
// (0/1/2), at offset slice.z, spanning the box in the other two axes. Two
// triangles from SV_VertexID -- no vertex buffer. Shares the volume UBO.
// Mirrors ses::slice_quad_corner (core/cross_section.hpp). gl_VertexIndex
// (glsl module) maps to the raw SPIR-V VertexIndex builtin -- unlike
// SV_VertexID it needs no shaderDrawParameters capability.
import glsl;

struct Ubo {
    float4x4 mvp;
    float4 eye;
    float4 box_min;
    float4 box_max;
    float4 proton_center;
    float4 proton_color;
    float inv_peak;
    float absorbance;
    float proton_radius;
    float jitter_frame;
    float4 clip;
    float4 slice;  // .x enable, .y axis, .z offset, .w map mode
};
[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float3 v_world : TEXCOORD0;
};

void set_comp(inout float3 v, int i, float x) {
    if (i == 0) v.x = x;
    else if (i == 1) v.y = x;
    else v.z = x;
}
float comp(float3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

// CCW quad corners as (s, t) in {0,1}^2.
static const float2 kST[6] = {
    float2(0, 0), float2(1, 0), float2(1, 1),
    float2(0, 0), float2(1, 1), float2(0, 1)
};

[shader("vertex")]
VSOut main() {
    VSOut o;
    uint vid = uint(gl_VertexIndex);
    int a = int(ubo.slice.y + 0.5);
    int u = (a + 1) % 3;
    int w = (a + 2) % 3;
    float2 st = kST[vid];
    float3 p = float3(0.0, 0.0, 0.0);
    set_comp(p, a, ubo.slice.z);
    set_comp(p, u, lerp(comp(ubo.box_min.xyz, u), comp(ubo.box_max.xyz, u), st.x));
    set_comp(p, w, lerp(comp(ubo.box_min.xyz, w), comp(ubo.box_max.xyz, w), st.y));
    o.v_world = p;
    o.pos = mul(ubo.mvp, float4(p, 1.0));
    return o;
}
