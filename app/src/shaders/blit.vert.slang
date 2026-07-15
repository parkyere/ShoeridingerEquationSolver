// Fullscreen-triangle blit: the shell's swapchain pass. The scene
// itself is rendered by ses_vk into an offscreen image; this pass samples it
// 1:1 into the swapchain image (Dear ImGui rides the same pass after it).
// gl_VertexIndex (glsl module) maps to the raw SPIR-V VertexIndex builtin --
// unlike SV_VertexID it needs no shaderDrawParameters capability.
import glsl;

struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float2 v_uv : TEXCOORD0;
};

[shader("vertex")]
VSOut main() {
    VSOut o;
    uint vid = uint(gl_VertexIndex);
    float2 pos = float2(float((vid << 1) & 2), float(vid & 2));
    o.v_uv = pos;
    o.pos = float4(pos * 2.0 - 1.0, 0.0, 1.0);
    return o;
}
