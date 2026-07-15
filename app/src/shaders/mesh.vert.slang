// Mesh (proton marker + axes gizmo) vertex stage. The std140 UBO (binding 0)
// is shared with the fragment stage; eye is vec4-padded. The MVP already
// carries the GL->Vulkan clip correction (y flip, depth [-1,1] -> [0,1])
// applied host-side, so the clip position is authored in the camera's GL clip
// conventions with no correction here (column-major mul matches GLSL mvp*p).
struct VSIn {
    [[vk::location(0)]] float3 pos : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float3 color : COLOR;
};
struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float3 v_normal : TEXCOORD0;
    [[vk::location(1)]] float3 v_pos : TEXCOORD1;
    [[vk::location(2)]] float3 v_color : TEXCOORD2;
};

struct Ubo {
    float4x4 mvp;
    float4 eye;  // .xyz
};
[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[shader("vertex")]
VSOut main(VSIn input) {
    VSOut o;
    o.pos = mul(ubo.mvp, float4(input.pos, 1.0));
    o.v_normal = input.normal;
    o.v_pos = input.pos;
    o.v_color = input.color;
    return o;
}
