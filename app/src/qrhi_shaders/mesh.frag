#version 450

// Headlight shading of the proton marker + axes gizmo. Shares the vertex
// stage's std140 UBO (binding 0) for eye.
layout(location = 0) in vec3 v_normal;
layout(location = 1) in vec3 v_pos;
layout(location = 2) in vec3 v_color;

layout(std140, binding = 0) uniform Ubo {
    mat4 mvp;
    vec4 eye;  // .xyz
};

layout(location = 0) out vec4 frag;

void main() {
    vec3 n = normalize(v_normal);
    vec3 vdir = normalize(eye.xyz - v_pos);
    float diffuse = abs(dot(n, vdir));
    float spec = pow(max(dot(n, vdir), 0.0), 32.0);
    vec3 c = v_color * (0.20 + 0.80 * diffuse) + vec3(0.25) * spec;
    frag = vec4(c, 1.0);
}
