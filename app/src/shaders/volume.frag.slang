// Front-to-back ray-marched volume render of |psi|^2 with phase-tinted color.
// Transcribes the TESTED ses.volume formulas (ray_box slab, Beer-Lambert
// alpha, compositing). psi is an RG16F 3D texture (re,im), colors
// from the phase LUT 1D texture. std140 UBO (binding 0, shared with the
// vertex stage); samplers at bindings 1-4.
import vec;

struct FSIn {
    float4 fragCoord : SV_Position;  // gl_FragCoord
    [[vk::location(0)]] float3 v_world : TEXCOORD0;
};

struct Ubo {
    float4x4 mvp;
    float4 eye;             // .xyz
    float4 box_min;         // .xyz
    float4 box_max;         // .xyz
    float4 proton_center;   // .xyz
    float4 proton_color;    // .xyz
    float inv_peak;
    float absorbance;
    float proton_radius;
    float jitter_frame;  // temporal rotation of the raymarch jitter
    float4 clip;   // .x enable, .y axis(0/1/2), .z sign(+-1), .w offset
    float4 slice;  // (slice pass only; here to share the UBO layout)
};
[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[[vk::binding(1, 0)]] Sampler3D<float4> psi_tex;
[[vk::binding(2, 0)]] Sampler1D<float4> phase_tex;
[[vk::binding(3, 0)]] Sampler3D<float4> occ_tex;     // dilated block max (NEAREST)
[[vk::binding(4, 0)]] Sampler3D<float4> shadow_tex;  // key-light transmittance

float2 ray_box(float3 o, float3 d) {
    float3 inv = 1.0 / d;
    float3 t1 = (ubo.box_min.xyz - o) * inv;
    float3 t2 = (ubo.box_max.xyz - o) * inv;
    float3 tmin = min(t1, t2);
    float3 tmax = max(t1, t2);
    return float2(max(max(tmin.x, tmin.y), tmin.z),
                  min(min(tmax.x, tmax.y), tmax.z));
}

// Mirrors the tested ses::ray_sphere (unit direction, raw interval).
float2 ray_sphere(float3 o, float3 d) {
    float3 oc = o - ubo.proton_center.xyz;
    float b = dot(oc, d);
    float c = dot(oc, oc) - ubo.proton_radius * ubo.proton_radius;
    float disc = b * b - c;
    if (disc < 0.0) {
        return float2(1.0, -1.0);  // empty interval = miss
    }
    float s = sqrt(disc);
    return float2(-b - s, -b + s);
}

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    float3 v_world = input.v_world;
    float3 dir = normalize(v_world - ubo.eye.xyz);
    float2 t = ray_box(ubo.eye.xyz, dir);
    float tn = max(t.x, 0.0);
    if (t.y <= tn) {
        discard;
    }
    float t_stop = t.y;

    // Clip plane (mirrors ses::clip_ray_interval, core/cross_section.hpp):
    // keep only the half-space clip.z*(p[axis]-offset) <= 0 so the cut face
    // reveals the interior. Restrict the ray interval to it.
    if (ubo.clip.x > 0.5) {
        int ax = int(ubo.clip.y + 0.5);
        float a = ubo.clip.z * comp(dir, ax);
        float b = ubo.clip.z * (ubo.clip.w - comp(ubo.eye.xyz, ax));
        if (abs(a) < 1e-8) {
            if (b < 0.0) discard;  // whole ray on the cut side
        } else {
            float tp = b / a;
            if (a > 0.0) t_stop = min(t_stop, tp);  // keep t <= tp
            else tn = max(tn, tp);                  // keep t >= tp
        }
        if (t_stop <= tn) discard;
    }

    // Terminate the march at the proton sphere: cloud IN FRONT still fogs
    // it, cloud BEHIND is correctly occluded.
    bool hit_proton = false;
    float2 sp = ray_sphere(ubo.eye.xyz, dir);
    if (sp.x <= sp.y && sp.x > tn && sp.x < t_stop) {
        hit_proton = true;
        t_stop = sp.x;
    }

    const int kSteps = 384;  // ~0.42 Bohr/sample across the +-80 box
    float step_len = (t_stop - tn) / float(kSteps);
    // Interleaved gradient noise, rotated per frame (golden-ratio walk):
    // spatially it kills wood-grain banding; temporally it decorrelates the
    // residual noise so the accumulation pass averages it away.
    float2 jp = input.fragCoord.xy + 5.588238 * ubo.jitter_frame;
    float jitter = frac(52.9829189 * frac(0.06711056 * jp.x + 0.00583715 * jp.y));

    // Half-texel alignment: the periodic grid puts point i at
    // box_min + i*h (h = L/n), but texel i's CENTER is at (i+0.5)/n --
    // without the correction the displayed cloud sits half a cell
    // (~0.3 Bohr) off the analytic proton/world coordinates.
    uint tw, th, tdp;
    psi_tex.GetDimensions(tw, th, tdp);
    float3 half_texel = 0.5 / float3(tw, th, tdp);
    float3 ext = ubo.box_max.xyz - ubo.box_min.xyz;
    // Empty-space skipping: while the (dilated) occupancy block under the
    // ray is empty, jump a whole block; otherwise fine-march a block's worth
    // at the ORIGINAL step length, so occupied regions look identical.
    float skip_len = min(min(ext.x, ext.y), ext.z) / 32.0;

    float3 C = float3(0.0, 0.0, 0.0);
    float A = 0.0;
    float t_cur = tn + jitter * step_len;
    int fine_budget = kSteps;
    while (t_cur < t_stop && fine_budget > 0) {
        float3 pc = ubo.eye.xyz + t_cur * dir;
        float3 uvw_c = (pc - ubo.box_min.xyz) / ext;
        if (occ_tex.Sample(uvw_c).r < 1e-5) {
            t_cur += skip_len;
            continue;
        }
        for (int k = 0; k < 8 && t_cur < t_stop && fine_budget > 0 &&
                        A < 0.999; ++k) {
            float3 p = ubo.eye.xyz + t_cur * dir;
            float3 uvw = (p - ubo.box_min.xyz) / ext + half_texel;
            float2 s = psi_tex.Sample(uvw).rg;
            float dens = dot(s, s) * ubo.inv_peak;
            float alpha = 1.0 - exp(-ubo.absorbance * dens * step_len);
            float phase01 =
                (atan2(s.g, s.r) + 3.14159265358979) / 6.28318530717959;
            float3 col = phase_tex.Sample(phase01).rgb;
            // Self-shadow: the key-light transmittance carves real 3D
            // depth into the lobes (ambient floor keeps nodes readable).
            float T = shadow_tex.Sample(uvw_c).r;
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
        float3 p = ubo.eye.xyz + sp.x * dir;
        float3 n = (p - ubo.proton_center.xyz) / ubo.proton_radius;
        float diffuse = max(dot(n, -dir), 0.0);
        float spec = pow(diffuse, 32.0);
        float3 shaded = ubo.proton_color.xyz * (0.25 + 0.75 * diffuse) +
                        float3(0.2, 0.2, 0.2) * spec;
        C += (1.0 - A) * shaded;
        A = 1.0;
    }

    return float4(C, A);  // premultiplied; blended over the clear color
}
