#version 450

// Front-to-back ray-marched volume render of |psi|^2 with phase-tinted color.
// Transcribes the TESTED core/volume.hpp formulas (ray_box slab, Beer-Lambert
// alpha, compositing). psi is an RGBA32F 3D texture (re,im in .rg), colors
// from the phase LUT 1D texture. std140 UBO (binding 0, shared with the
// vertex stage); samplers at bindings 1-4.
layout(location = 0) in vec3 v_world;

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
    float jitter_frame;  // temporal rotation of the raymarch jitter
};

layout(binding = 1) uniform sampler3D psi_tex;
layout(binding = 2) uniform sampler1D phase_tex;
layout(binding = 3) uniform sampler3D occ_tex;     // dilated block max (NEAREST)
layout(binding = 4) uniform sampler3D shadow_tex;  // key-light transmittance

layout(location = 0) out vec4 frag;

vec2 ray_box(vec3 o, vec3 d) {
    vec3 inv = 1.0 / d;
    vec3 t1 = (box_min.xyz - o) * inv;
    vec3 t2 = (box_max.xyz - o) * inv;
    vec3 tmin = min(t1, t2);
    vec3 tmax = max(t1, t2);
    return vec2(max(max(tmin.x, tmin.y), tmin.z),
                min(min(tmax.x, tmax.y), tmax.z));
}

// Mirrors the tested ses::ray_sphere (unit direction, raw interval).
vec2 ray_sphere(vec3 o, vec3 d) {
    vec3 oc = o - proton_center.xyz;
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
    vec3 dir = normalize(v_world - eye.xyz);
    vec2 t = ray_box(eye.xyz, dir);
    float tn = max(t.x, 0.0);
    if (t.y <= tn) {
        discard;
    }

    // Terminate the march at the proton sphere: cloud IN FRONT still fogs
    // it, cloud BEHIND is correctly occluded.
    float t_stop = t.y;
    bool hit_proton = false;
    vec2 sp = ray_sphere(eye.xyz, dir);
    if (sp.x <= sp.y && sp.x > tn && sp.x < t_stop) {
        hit_proton = true;
        t_stop = sp.x;
    }

    const int kSteps = 384;  // ~0.42 Bohr/sample across the +-80 box
    float step_len = (t_stop - tn) / float(kSteps);
    // Interleaved gradient noise, rotated per frame (golden-ratio walk):
    // spatially it kills wood-grain banding; temporally it decorrelates the
    // residual noise so the accumulation pass averages it away.
    vec2 jp = gl_FragCoord.xy + 5.588238 * jitter_frame;
    float jitter = fract(52.9829189 * fract(0.06711056 * jp.x + 0.00583715 * jp.y));

    // Half-texel alignment: the periodic grid puts point i at
    // box_min + i*h (h = L/n), but texel i's CENTER is at (i+0.5)/n --
    // without the correction the displayed cloud sits half a cell
    // (~0.3 Bohr) off the analytic proton/world coordinates.
    vec3 half_texel = 0.5 / vec3(textureSize(psi_tex, 0));
    vec3 ext = box_max.xyz - box_min.xyz;
    // Empty-space skipping: while the (dilated) occupancy block under the
    // ray is empty, jump a whole block; otherwise fine-march a block's worth
    // at the ORIGINAL step length, so occupied regions look identical.
    float skip_len = min(min(ext.x, ext.y), ext.z) / 32.0;

    vec3 C = vec3(0.0);
    float A = 0.0;
    float t_cur = tn + jitter * step_len;
    int fine_budget = kSteps;
    while (t_cur < t_stop && fine_budget > 0) {
        vec3 pc = eye.xyz + t_cur * dir;
        vec3 uvw_c = (pc - box_min.xyz) / ext;
        if (texture(occ_tex, uvw_c).r < 1e-5) {
            t_cur += skip_len;
            continue;
        }
        for (int k = 0; k < 8 && t_cur < t_stop && fine_budget > 0; ++k) {
            vec3 p = eye.xyz + t_cur * dir;
            vec3 uvw = (p - box_min.xyz) / ext + half_texel;
            vec2 s = texture(psi_tex, uvw).rg;
            float dens = dot(s, s) * inv_peak;
            float alpha = 1.0 - exp(-absorbance * dens * step_len);
            float phase01 =
                (atan(s.g, s.r) + 3.14159265358979) / 6.28318530717959;
            vec3 col = texture(phase_tex, phase01).rgb;
            // Self-shadow: the key-light transmittance carves real 3D
            // depth into the lobes (ambient floor keeps nodes readable).
            float T = texture(shadow_tex, uvw_c).r;
            col *= 0.35 + 0.65 * T;
            float w = (1.0 - A) * alpha;
            C += w * col;
            A += w;
            t_cur += step_len;
            --fine_budget;
        }
        if (A >= 0.999) {
            break;
        }
    }

    if (hit_proton) {
        vec3 p = eye.xyz + sp.x * dir;
        vec3 n = (p - proton_center.xyz) / proton_radius;
        float diffuse = max(dot(n, -dir), 0.0);
        float spec = pow(diffuse, 32.0);
        vec3 shaded = proton_color.xyz * (0.25 + 0.75 * diffuse) + vec3(0.2) * spec;
        C += (1.0 - A) * shaded;
        A = 1.0;
    }

    frag = vec4(C, A);  // premultiplied; blended over the clear color
}
