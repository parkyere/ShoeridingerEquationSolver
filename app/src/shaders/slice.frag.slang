// Cross-section sheet: sample psi on the plane and paint it. Map modes
// (slice.w): 0 = |psi|^2 density (magnitude ramp), 1 = Re(psi) diverging
// (blue<-0->orange, the orbital's signed lobes), 2 = phase arg(psi) (cyclic
// HSV LUT, brightness by |psi|). The card carries a base opacity so nodal
// planes read as dark bands ON a visible sheet. The colour/alpha and the
// clip-side discard MIRROR ses::slice_shade / ses::clip_ray_interval
// (core/cross_section.hpp) -- the tested single source of truth.
struct FSIn {
    float4 pos : SV_Position;
    [[vk::location(0)]] float3 v_world : TEXCOORD0;
};

struct Ubo {
    float4x4 mvp;
    float4 eye;
    float4 box_min;
    float4 box_max;
    float4 proton_center;
    float4 proton_color;
    float inv_peak;
    float absorbance;
    float proton_radius;
    float jitter_frame;
    float4 clip;
    float4 slice;  // .x enable, .y axis, .z offset, .w map mode
};
[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[[vk::binding(1, 0)]] Sampler3D<float4> psi_tex;
[[vk::binding(2, 0)]] Sampler1D<float4> phase_tex;

float comp(float3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    float3 v_world = input.v_world;
    // Respect the clip plane: the sheet is hidden on the cut-away half so it
    // does not float in the emptied space when the two planes differ.
    if (ubo.clip.x > 0.5 &&
        ubo.clip.z * (comp(v_world, int(ubo.clip.y + 0.5)) - ubo.clip.w) > 0.0) {
        discard;
    }
    float3 ext = ubo.box_max.xyz - ubo.box_min.xyz;
    uint tw, th, td;
    psi_tex.GetDimensions(tw, th, td);
    float3 half_texel = 0.5 / float3(tw, th, td);
    float3 uvw = (v_world - ubo.box_min.xyz) / ext + half_texel;
    float2 s = psi_tex.Sample(uvw).rg;
    float dens = dot(s, s) * ubo.inv_peak;      // normalized |psi|^2 in [0,1]
    float amp = sqrt(max(ubo.inv_peak, 0.0));   // shared sign/Re scale

    int mode = int(ubo.slice.w + 0.5);
    float3 col;
    float bright;
    if (mode == 1) {  // Re(psi), diverging
        float r = clamp(s.r * amp, -1.0, 1.0);
        float3 pos = float3(1.0, 0.55, 0.15);   // + lobe (warm)
        float3 neg = float3(0.15, 0.45, 1.0);   // - lobe (cool)
        col = (r >= 0.0) ? lerp(float3(0.03, 0.03, 0.03), pos, r)
                         : lerp(float3(0.03, 0.03, 0.03), neg, -r);
        bright = abs(r);
    } else if (mode == 2) {  // phase
        float ph = (atan2(s.g, s.r) + 3.14159265358979) / 6.28318530717959;
        col = phase_tex.Sample(ph).rgb;
        bright = sqrt(clamp(dens, 0.0, 1.0));
        col *= 0.25 + 0.75 * bright;
    } else {  // density (mode 0)
        float dd = clamp(dens, 0.0, 1.0);
        col = float3(pow(dd, 1.6), pow(dd, 0.9), 0.25 + 0.75 * dd);  // navy->white
        bright = dd;
    }
    // Base opacity keeps the sheet (and its nodes) visible; denser regions
    // are more solid so the cross-section still reads as the cloud.
    float alpha = clamp(0.45 + 0.5 * bright, 0.0, 0.95);
    return float4(col * alpha, alpha);  // premultiplied
}
