// Occupancy build (empty-space skipping): one WORKGROUP per coarse cell,
// one thread per fine texel of its block, shared-memory max reduction. Max is
// exactly associative in fp, so the parallel reduction order changes nothing.
// Rebuilt only when the displayed volume changes. The raymarcher skips
// blocks whose (dilated) max is negligible -- most of the +-80 Bohr box.
[[vk::binding(0, 0)]] [[vk::image_format("r32f")]] RWTexture3D<float4> occ;

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

groupshared float s_max[512];

[shader("compute")]
[numthreads(8, 8, 8)]
void main(uint3 gwg : SV_GroupID, uint3 glid : SV_GroupThreadID,
          uint lidx : SV_GroupIndex) {
    // The dispatch is exactly one workgroup per occ cell (32^3), so no
    // bounds check -- every invocation must reach the barriers below.
    int3 c = int3(gwg);
    uint ow, oh, od;
    occ.GetDimensions(ow, oh, od);
    int3 osize = int3(ow, oh, od);
    uint fw, fh, fd;
    psi_tex.GetDimensions(fw, fh, fd);
    int3 fsize = int3(fw, fh, fd);
    // Per-axis fine range [begin, end) of this occ cell -- proportional,
    // NOT fsize/osize: a volume THINNER than the occ grid on an axis (the
    // 2D scenes' 4-cell display slab) would give block 0 there, an empty
    // loop, occ == 0 everywhere, and the raymarcher would skip the whole
    // cloud. When several occ cells land on the same fine plane they all
    // (redundantly, correctly) read it.
    int3 begin = (c * fsize) / osize;
    int3 end = max(begin + int3(1, 1, 1), ((c + int3(1, 1, 1)) * fsize) /
                                              osize);
    // One lattice point per thread; stride when the block outgrows the 8^3
    // workgroup (a 512^3 volume gives 16^3 blocks).
    float m = 0.0;
    for (int k = begin.z + int(glid.z); k < end.z; k += 8) {
        for (int j = begin.y + int(glid.y); j < end.y; j += 8) {
            for (int i = begin.x + int(glid.x); i < end.x; i += 8) {
                float2 s = psi_tex.Load(int4(int3(i, j, k), 0)).xy;
                m = max(m, dot(s, s));
            }
        }
    }
    uint tid = lidx;
    s_max[tid] = m;
    GroupMemoryBarrierWithGroupSync();
    for (uint h = 256u; h > 0u; h >>= 1u) {
        if (tid < h) {
            s_max[tid] = max(s_max[tid], s_max[tid + h]);
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        occ[c] = float4(s_max[0] * ubo.inv_peak, 0.0, 0.0, 0.0);
    }
}
