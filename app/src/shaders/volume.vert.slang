// Volume box vertex stage: pass the world-space cube corner to the fragment
// raymarcher. Shares the volume std140 UBO (binding 0) with the fragment
// stage. The MVP carries the GL->Vulkan clip correction host-side.
struct VSIn {
    [[vk::location(0)]] float3 pos : POSITION;
};
struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float3 v_world : TEXCOORD0;
};

struct Ubo {
    float4x4 mvp;
    float4 eye;             // .xyz
    float4 box_min;         // .xyz
    float4 box_max;         // .xyz
    float4 proton_center;   // .xyz
    float4 proton_color;    // .xyz
    float inv_peak;
    float absorbance;
    float proton_radius;
    float pad0;
};
[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[shader("vertex")]
VSOut main(VSIn input) {
    VSOut o;
    o.pos = mul(ubo.mvp, float4(input.pos, 1.0));
    o.v_world = input.pos;
    return o;
}
