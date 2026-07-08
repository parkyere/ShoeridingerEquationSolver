#pragma once

// Rendering GLSL (mesh + volume raymarch) for the Qt viewport.
// Extracted verbatim from main.cpp (no logic change).

inline const char* kMeshVertexShader = R"(#version 430 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec3 color;
uniform mat4 mvp;
out vec3 v_normal;
out vec3 v_pos;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    v_normal = normal;
    v_pos = pos;
    v_color = color;
}
)";

inline const char* kMeshFragmentShader = R"(#version 430 core
in vec3 v_normal;
in vec3 v_pos;
in vec3 v_color;
uniform vec3 eye;
out vec4 frag;
void main() {
    vec3 n = normalize(v_normal);
    vec3 vdir = normalize(eye - v_pos);
    float diffuse = abs(dot(n, vdir));
    float spec = pow(max(dot(n, vdir), 0.0), 32.0);
    vec3 c = v_color * (0.20 + 0.80 * diffuse) + vec3(0.25) * spec;
    frag = vec4(c, 1.0);
}
)";

// ---- volume (ray-marching) shaders ----

inline const char* kVolumeVertexShader = R"(#version 430 core
layout(location = 0) in vec3 pos;
uniform mat4 mvp;
out vec3 v_world;
void main() {
    gl_Position = mvp * vec4(pos, 1.0);
    v_world = pos;
}
)";

// Transcription of the TESTED core formulas (core/volume.hpp):
// ray_box (slab), sample_alpha (Beer-Lambert), front-to-back compositing.
// Colors come from the tested phase_lut via a 1D texture; density and phase
// are derived from the complex psi stored in an RG 3D texture, so the GPU
// interpolates re/im exactly like core's sample_trilinear.
inline const char* kVolumeFragmentShader = R"(#version 430 core
in vec3 v_world;
uniform vec3 eye;
uniform vec3 box_min;
uniform vec3 box_max;
uniform float inv_peak;
uniform float absorbance;
uniform vec3 proton_center;
uniform float proton_radius;
uniform vec3 proton_color;
uniform sampler3D psi_tex;
uniform sampler1D phase_tex;
out vec4 frag;

vec2 ray_box(vec3 o, vec3 d) {
    vec3 inv = 1.0 / d;
    vec3 t1 = (box_min - o) * inv;
    vec3 t2 = (box_max - o) * inv;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    return vec2(max(max(tmin.x, tmin.y), tmin.z),
                min(min(tmax.x, tmax.y), tmax.z));
}

// Mirrors the tested ses::ray_sphere (unit direction, raw interval).
vec2 ray_sphere(vec3 o, vec3 d) {
    vec3 oc = o - proton_center;
    float b = dot(oc, d);
    float c = dot(oc, oc) - proton_radius * proton_radius;
    float disc = b * b - c;
    if (disc < 0.0) {
        return vec2(1.0, -1.0);  // empty interval = miss
    }
    float s = sqrt(disc);
    return vec2(-b - s, -b + s);
}

void main() {
    vec3 dir = normalize(v_world - eye);
    vec2 t = ray_box(eye, dir);
    float tn = max(t.x, 0.0);
    if (t.y <= tn) {
        discard;
    }

    // Terminate the march at the proton sphere: cloud IN FRONT still fogs
    // it, cloud BEHIND is correctly occluded (the earlier mesh marker was
    // fogged from both sides and vanished at the density peak).
    float t_stop = t.y;
    bool hit_proton = false;
    vec2 sp = ray_sphere(eye, dir);
    if (sp.x <= sp.y && sp.x > tn && sp.x < t_stop) {
        hit_proton = true;
        t_stop = sp.x;
    }

    const int kSteps = 384;  // ~0.42 Bohr/sample across the +-80 box
    float step_len = (t_stop - tn) / float(kSteps);
    // Per-pixel jitter of the ray start kills wood-grain banding.
    float jitter = fract(sin(dot(gl_FragCoord.xy, vec2(12.9898, 78.233))) * 43758.5453);

    vec3 C = vec3(0.0);
    float A = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        vec3 p = eye + (tn + (float(i) + jitter) * step_len) * dir;
        vec3 uvw = (p - box_min) / (box_max - box_min);
        vec2 s = texture(psi_tex, uvw).rg;
        float dens = dot(s, s) * inv_peak;
        float alpha = 1.0 - exp(-absorbance * dens * step_len);
        float phase01 = (atan(s.g, s.r) + 3.14159265358979) / 6.28318530717959;
        vec3 col = texture(phase_tex, phase01).rgb;
        float w = (1.0 - A) * alpha;
        C += w * col;
        A += w;
        if (A >= 0.999) {
            break;
        }
    }

    if (hit_proton) {
        // Shade the sphere point with the same headlight model as the mesh.
        vec3 p = eye + sp.x * dir;
        vec3 n = (p - proton_center) / proton_radius;
        float diffuse = max(dot(n, -dir), 0.0);
        float spec = pow(diffuse, 32.0);
        vec3 shaded = proton_color * (0.25 + 0.75 * diffuse) + vec3(0.2) * spec;
        C += (1.0 - A) * shaded;
        A = 1.0;
    }

    frag = vec4(C, A);  // premultiplied; blended over the clear color
}
)";
