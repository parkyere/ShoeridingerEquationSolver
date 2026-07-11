#version 450

// Mesh (proton marker + axes gizmo) vertex stage. The std140 UBO (binding 0)
// is shared with the fragment stage; eye is vec4-padded. The MVP already
// carries the GL->Vulkan clip correction (y flip, depth [-1,1] -> [0,1])
// applied host-side, so gl_Position is authored in the camera's GL clip
// conventions with no correction here.
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
