#version 450

// Fullscreen-triangle blit: the ONLY geometry Qt still draws. The scene
// itself is rendered by ses_vk into an offscreen image; this pass samples it
// 1:1 into the QRhiWidget's backing texture.
layout(location = 0) out vec2 v_uv;

void main() {
    vec2 pos = vec2(float((gl_VertexIndex << 1) & 2), float(gl_VertexIndex & 2));
    v_uv = pos;
    gl_Position = vec4(pos * 2.0 - 1.0, 0.0, 1.0);
}
