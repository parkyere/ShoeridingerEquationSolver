// Sample the ses_vk scene image (SHADER_READ_ONLY at hand-off).
struct FSIn {
    float4 pos : SV_Position;
    [[vk::location(0)]] float2 v_uv : TEXCOORD0;
};
[[vk::binding(0, 0)]] Sampler2D<float4> scene;  // combined image sampler

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    return scene.Sample(input.v_uv);
}
