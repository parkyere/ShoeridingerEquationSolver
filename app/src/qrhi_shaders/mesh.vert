#version 450

// QRhi/Vulkan form of kMeshVertexShader (render_shaders.hpp): the mesh (proton
// marker + axes gizmo) vertex stage. Default-block `uniform mat4 mvp` becomes a
// std140 UBO shared with the fragment stage (binding 0); eye is vec4-padded.
// The MVP already carries QRhi::clipSpaceCorrMatrix (applied host-side), so
// gl_Position is authored exactly as the GL shader.
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;

layout(std140, binding = 0) uniform Ubo {
    mat4 mvp;
    vec4 eye;  // .xyz
};

layout(location = 0) out vec3 v_normal;
layout(location = 1) out vec3 v_pos;
layout(location = 2) out vec3 v_color;

void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    v_normal = normal;
    v_pos = pos;
    v_color = color;
}
