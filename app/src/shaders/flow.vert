#version 450

// Streakline vertex: instanced LINE_STRIP, one strip per streak.
//   gl_InstanceIndex = streak,  gl_VertexIndex = trail slot
//   (0 = oldest tail .. trail_len-1 = newest head).
// Premultiplied WHITE with a tail->head alpha fade (a weather-map Lagrangian
// flux streak). Fade matches core/flow.hpp::trail_fade.
layout(std430, binding = 0) readonly buffer Trail { vec4 trail[]; };
layout(std140, binding = 1) uniform Ubo {
    mat4 mvp;
    vec4 eye;             // .xyz (unused here; struct matches the shared write)
    vec4 box_min;
    vec4 box_max;
    vec4 proton_center;
    vec4 proton_color;
    float inv_peak;
    float absorbance;
    float proton_radius;
    float jitter_frame;
};
layout(push_constant) uniform Push { int trail_len; };

layout(location = 0) out vec4 v_color;

void main() {
    int slot = gl_InstanceIndex * trail_len + gl_VertexIndex;
    vec3 p = trail[slot].xyz;

    float fade = (trail_len > 1)
                     ? float(gl_VertexIndex) / float(trail_len - 1)
                     : 1.0;
    float a = fade * 0.85;         // near-full white at the head
    v_color = vec4(vec3(a), a);    // premultiplied white (coverage = a)

    gl_Position = mvp * vec4(p, 1.0);
}
