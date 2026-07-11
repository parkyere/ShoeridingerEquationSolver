#version 450

// Volume box vertex stage: pass the world-space cube corner to the fragment
// raymarcher. Shares the volume std140 UBO (binding 0) with the fragment
// stage. The MVP carries the GL->Vulkan clip correction host-side.
layout(location = 0) in vec3 pos;

layout(std140, binding = 0) uniform Ubo {
    mat4 mvp;
    vec4 eye;             // .xyz
    vec4 box_min;         // .xyz
    vec4 box_max;         // .xyz
    vec4 proton_center;   // .xyz
    vec4 proton_color;    // .xyz
    float inv_peak;
    float absorbance;
    float proton_radius;
    float pad0;
};

layout(location = 0) out vec3 v_world;

void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    v_world = pos;
}
