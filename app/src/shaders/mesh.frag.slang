// Headlight shading of the proton marker + axes gizmo. Shares the vertex
// stage's std140 UBO (binding 0) for eye.
struct FSIn {
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

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    float3 n = normalize(input.v_normal);
    float3 vdir = normalize(ubo.eye.xyz - input.v_pos);
    float diffuse = abs(dot(n, vdir));
    float spec = pow(max(dot(n, vdir), 0.0), 32.0);
    float3 c = input.v_color * (0.20 + 0.80 * diffuse) + float3(0.25, 0.25, 0.25) * spec;
    return float4(c, 1.0);
}
