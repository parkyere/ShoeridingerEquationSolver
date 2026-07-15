// Volumetric self-shadow: per voxel of a coarse (64^3) volume, march from
// the voxel toward a fixed key light accumulating optical depth through
// |psi|^2, store the transmittance T = exp(-tau). The raymarcher multiplies
// its emission by (ambient + (1-ambient) T): the cloud shades ITSELF, and
// lobes/nodal structure gain real 3D depth. Rebuilt only when the volume
// (or the absorbance dial) changes.
[[vk::binding(0, 0)]] [[vk::image_format("r32f")]] RWTexture3D<float4> shadow_img;

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
};
[[vk::binding(1, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[[vk::binding(2, 0)]] Sampler3D<float4> psi_tex;  // combined image sampler

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 gid : SV_DispatchThreadID) {
    int3 c = int3(gid);
    uint w, h, d;
    shadow_img.GetDimensions(w, h, d);
    int3 size = int3(w, h, d);
    if (any(c >= size)) {
        return;
    }
    float3 kLightDir = normalize(float3(0.45, 0.80, 0.35));  // world-fixed key
    float3 ext = ubo.box_max.xyz - ubo.box_min.xyz;
    float3 p = ubo.box_min.xyz + (float3(c) + 0.5) / float3(size) * ext;

    // Distance to the box exit along the light direction.
    float3 inv = 1.0 / kLightDir;
    float3 t1 = (ubo.box_min.xyz - p) * inv;
    float3 t2 = (ubo.box_max.xyz - p) * inv;
    float3 tmax = max(t1, t2);
    float t_exit = min(min(tmax.x, tmax.y), tmax.z);

    const int kSteps = 48;
    float dl = t_exit / float(kSteps);
    uint pw, ph, pd;
    psi_tex.GetDimensions(pw, ph, pd);
    float3 half_texel = 0.5 / float3(pw, ph, pd);
    float tau = 0.0;
    for (int i = 0; i < kSteps; ++i) {
        float3 q = p + (float(i) + 0.5) * dl * kLightDir;
        float3 uvw = (q - ubo.box_min.xyz) / ext + half_texel;
        float2 s = psi_tex.SampleLevel(uvw, 0.0).xy;
        tau += dot(s, s) * ubo.inv_peak * dl;
    }
    float T = exp(-ubo.absorbance * tau);
    shadow_img[c] = float4(T, 0.0, 0.0, 0.0);
}
