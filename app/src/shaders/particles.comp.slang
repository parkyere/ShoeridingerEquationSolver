// Probability-flow STREAKLINES. Each streak owns a short trail (trail_len
// slots) of its recent positions; every animated frame the head is advected
// one Bohmian step and the trail shifts down (oldest slot drops). The velocity
// is v = j / rho, j = Im(conj(psi) grad psi) (atomic units) -- the same current
// as ses.flow::probability_current -- read from the fp32 velocity volume
// (vel_tex); density from the fp16 display volume (psi_tex). Streaks respawn
// (whole trail collapsed to a fresh seed) by rejection sampling against
// |psi|^2, so they stay |psi|^2-distributed (equivariance): the streaks you
// see ARE the probability current -- still for real eigenstates, circulating
// for m != 0, sloshing for driven/superposed states.
[[vk::binding(0, 0)]] RWStructuredBuffer<float4, Std430DataLayout> trail;

struct Params {
    float4 box_min;  // .xyz
    float4 box_max;  // .xyz
    float dt;        // display advection step (au)
    float inv_peak;  // |psi|^2 normalizer (rejection sampling)
    float lifetime;  // frames until forced respawn
    float frame;     // RNG decorrelation
    int n_streaks;
    int trail_len;
    int animate;     // 0 = hold positions (paused)
    int reset;       // 1 = collapse every trail to a fresh seed (enable frame)
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[[vk::binding(2, 0)]] Sampler3D<float4> psi_tex;  // fp16 display volume (density)
[[vk::binding(3, 0)]] Sampler3D<float4> vel_tex;  // fp32-computed Bohmian velocity

float2 psi_at(float3 p) {
    // Half-texel alignment (grid point i = texel center (i+0.5)/n).
    uint w, h, d;
    psi_tex.GetDimensions(w, h, d);
    float3 uvw = (p - params.box_min.xyz) / (params.box_max.xyz - params.box_min.xyz) +
                 0.5 / float3(w, h, d);
    return psi_tex.SampleLevel(uvw, 0.0).xy;
}

// Bohmian velocity, precomputed in fp32 from the physics psi (NOT the fp16
// display volume) so a real eigenstate reads v ~ 0 -- no spurious pulsation.
float3 vel_at(float3 p) {
    uint w, h, d;
    vel_tex.GetDimensions(w, h, d);
    float3 uvw = (p - params.box_min.xyz) / (params.box_max.xyz - params.box_min.xyz) +
                 0.5 / float3(w, h, d);
    return vel_tex.SampleLevel(uvw, 0.0).xyz;
}

// PCG-ish hash -> [0,1).
float rnd(inout uint state) {
    state = state * 747796405u + 2891336453u;
    uint w = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    w = (w >> 22u) ^ w;
    return float(w) / 4294967296.0;
}

// Rejection-sample a respawn point against |psi|^2 (a few tries; failures just
// keep the last draw and re-test next frame via the density gate).
float3 respawn_seed(inout uint rng) {
    float3 p = params.box_min.xyz;
    for (int t = 0; t < 4; ++t) {
        float3 cand = params.box_min.xyz +
                      float3(rnd(rng), rnd(rng), rnd(rng)) *
                          (params.box_max.xyz - params.box_min.xyz);
        p = cand;
        float2 cs = psi_at(cand);
        if (rnd(rng) < dot(cs, cs) * params.inv_peak) {
            break;
        }
    }
    return p;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint i = gid.x;
    if (i >= uint(params.n_streaks)) {
        return;
    }
    uint L = uint(params.trail_len);
    uint base = i * L;
    float4 head = trail[base + L - 1u];  // newest slot
    float3 p = head.xyz;
    float age = head.w;

    uint rng = i * 9781u + uint(params.frame) * 6271u + 12345u;

    float2 s = psi_at(p);
    float rho = dot(s, s);
    bool dead = age > params.lifetime || rho * params.inv_peak < 1e-4 ||
                any(p < params.box_min.xyz) ||
                any(p > params.box_max.xyz);

    if (params.reset != 0 || dead) {
        float3 seed = respawn_seed(rng);
        float na = params.lifetime * rnd(rng) * 0.25;  // stagger lifetimes
        for (uint k = 0u; k < L; ++k) {
            trail[base + k] = float4(seed, na);  // collapse the whole trail
        }
        return;
    }

    if (params.animate != 0) {
        float3 v = vel_at(p);  // fp32 Bohmian velocity (already node-guarded/clamped)
        p += v * params.dt;
        age += 1.0;

        // Shift the trail down (oldest drops) and append the new head. One
        // thread owns its L slots, and the ascending copy reads slot k+1
        // before it is overwritten at k+1, so no aliasing.
        for (uint k = 0u; k + 1u < L; ++k) {
            trail[base + k] = trail[base + k + 1u];
        }
        trail[base + L - 1u] = float4(p, age);
    }
}
