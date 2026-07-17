// Marching-cubes emit pass: same 8^3-cell blocking as classify. Each cell
// recomputes its case, a shared Hillis-Steele scan turns per-cell triangle
// counts into in-block offsets, and every triangle writes 3 interleaved
// vertices (pos3 normal3 color3 -- the mesh pipeline's exact layout) at
// slot block_offsets[wg] + prefix: a fixed function of the field, so the
// output is deterministic and cell-ordered within each block. Normals are
// interpolated central-difference gradients (down-gradient = outward,
// indices clamped at the walls -- core marching_cubes verbatim); colors are
// arg(psi) sampled trilinearly at the vertex through the cyclic HSV wheel
// (core sampling/colormap verbatim).
import complex;
import grid;

[[vk::binding(0, 0)]] StructuredBuffer<float, Std430DataLayout> den;
[[vk::binding(1, 0)]] StructuredBuffer<int, Std430DataLayout> mc_tab;
[[vk::binding(2, 0)]] StructuredBuffer<uint, Std430DataLayout> block_offsets;
[[vk::binding(3, 0)]] RWStructuredBuffer<float, Std430DataLayout> vb;
[[vk::binding(4, 0)]] StructuredBuffer<Complex, Std430DataLayout> psi;

struct Params {
    int4 npts;      // grid points per axis (.xyz)
    int4 nblk;      // cell blocks per axis (.xyz)
    float4 box_min; // .xyz
    float4 cell_h;  // .xyz
    float iso;
    int max_tris;
};
[[vk::binding(5, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

static const float PI = 3.14159265358979323846;

static const int3 kCorner[8] = {
    int3(0, 0, 0), int3(1, 0, 0), int3(1, 1, 0), int3(0, 1, 0),
    int3(0, 0, 1), int3(1, 0, 1), int3(1, 1, 1), int3(0, 1, 1)
};
static const int2 kEdge[12] = {
    int2(0, 1), int2(1, 2), int2(2, 3), int2(3, 0),
    int2(4, 5), int2(5, 6), int2(6, 7), int2(7, 4),
    int2(0, 4), int2(1, 5), int2(2, 6), int2(3, 7)
};

groupshared uint s_cnt[512];

int flat_at(int3 p) { return p.x + params.npts.x * (p.y + params.npts.y * p.z); }

float den_clamped(int3 p) {
    p = clamp(p, int3(0, 0, 0), params.npts.xyz - 1);
    return den[flat_at(p)];
}

// Central-difference density gradient (one-sided at the walls via clamping).
float3 gradient_at(int3 p) {
    return float3(
        (den_clamped(p + int3(1, 0, 0)) - den_clamped(p - int3(1, 0, 0))) /
            (2.0 * params.cell_h.x),
        (den_clamped(p + int3(0, 1, 0)) - den_clamped(p - int3(0, 1, 0))) /
            (2.0 * params.cell_h.y),
        (den_clamped(p + int3(0, 0, 1)) - den_clamped(p - int3(0, 0, 1))) /
            (2.0 * params.cell_h.z));
}

// Trilinear complex sample + cyclic HSV phase wheel (core verbatim).
float3 phase_color_at(float3 pos) {
    float3 s = (pos - params.box_min.xyz) / params.cell_h.xyz;
    int3 i0 = clamp(int3(floor(s)), int3(0, 0, 0), params.npts.xyz - 2);
    float3 t = s - float3(i0);
    Complex c00 = mix(psi[flat_at(i0)], psi[flat_at(i0 + int3(1, 0, 0))], t.x);
    Complex c10 = mix(psi[flat_at(i0 + int3(0, 1, 0))],
                      psi[flat_at(i0 + int3(1, 1, 0))], t.x);
    Complex c01 = mix(psi[flat_at(i0 + int3(0, 0, 1))],
                      psi[flat_at(i0 + int3(1, 0, 1))], t.x);
    Complex c11 = mix(psi[flat_at(i0 + int3(0, 1, 1))],
                      psi[flat_at(i0 + int3(1, 1, 1))], t.x);
    Complex c = mix(mix(c00, c10, t.y), mix(c01, c11, t.y), t.z);
    float theta = arg(c);
    float h = (theta + PI) / (2.0 * PI) * 6.0;
    h = h - 6.0 * floor(h / 6.0);
    float x = 1.0 - abs(fmod(h, 2.0) - 1.0);
    int hi = int(h);
    if (hi == 0) return float3(1.0, x, 0.0);
    if (hi == 1) return float3(x, 1.0, 0.0);
    if (hi == 2) return float3(0.0, 1.0, x);
    if (hi == 3) return float3(0.0, x, 1.0);
    if (hi == 4) return float3(x, 0.0, 1.0);
    return float3(1.0, 0.0, x);
}

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 grp : SV_GroupID, uint lidx : SV_GroupIndex) {
    int wg = int(grp.x);
    int3 blk = unflatten_moddiv(grp.x, params.nblk.x, params.nblk.y);
    int li = int(lidx);
    int3 cell = blk * 8 + unflatten_moddiv(lidx, 8, 8);

    int cube = 0;
    uint tris = 0u;
    bool live = all(cell < (params.npts.xyz - 1));
    if (live) {
        for (int c = 0; c < 8; ++c) {
            if (den[flat_at(cell + kCorner[c])] < params.iso) {
                cube |= 1 << c;
            }
        }
        tris = uint(mc_tab[4352 + cube]);
    }

    // Exclusive in-block scan of per-cell triangle counts.
    s_cnt[li] = tris;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = 1u; stride < 512u; stride <<= 1u) {
        uint v = s_cnt[li];
        uint add = (uint(li) >= stride) ? s_cnt[li - int(stride)] : 0u;
        GroupMemoryBarrierWithGroupSync();
        s_cnt[li] = v + add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint base = block_offsets[wg] + s_cnt[li] - tris;

    if (!live || tris == 0u) {
        return;
    }

    // Corner geometry once per cell.
    float3 cpos[8];
    float cval[8];
    for (int c = 0; c < 8; ++c) {
        int3 p = cell + kCorner[c];
        cpos[c] = cell_pos(p, params.box_min.xyz, params.cell_h.xyz);
        cval[c] = den[flat_at(p)];
    }

    int edges = mc_tab[cube];
    float3 ev[12];
    float3 en[12];
    for (int e = 0; e < 12; ++e) {
        if ((edges & (1 << e)) == 0) {
            continue;
        }
        int a = kEdge[e].x;
        int b = kEdge[e].y;
        float t = (params.iso - cval[a]) / (cval[b] - cval[a]);
        ev[e] = lerp(cpos[a], cpos[b], t);
        float3 ga = gradient_at(cell + kCorner[a]);
        float3 gb = gradient_at(cell + kCorner[b]);
        float3 nrm = -lerp(ga, gb, t);  // outward = down-gradient
        float len = length(nrm);
        en[e] = (len > 0.0) ? nrm / len : nrm;
    }

    for (uint tr = 0u; tr < tris; ++tr) {
        uint slot = base + tr;
        if (slot >= uint(params.max_tris)) {
            return;  // cap: offsets are monotone, the tail is a clean prefix cut
        }
        for (uint v = 0u; v < 3u; ++v) {
            int e = mc_tab[256 + cube * 16 + int(tr) * 3 + int(v)];
            uint o = slot * 27u + v * 9u;
            float3 col = phase_color_at(ev[e]);
            vb[o + 0u] = ev[e].x;
            vb[o + 1u] = ev[e].y;
            vb[o + 2u] = ev[e].z;
            vb[o + 3u] = en[e].x;
            vb[o + 4u] = en[e].y;
            vb[o + 5u] = en[e].z;
            vb[o + 6u] = col.r;
            vb[o + 7u] = col.g;
            vb[o + 8u] = col.b;
        }
    }
}
