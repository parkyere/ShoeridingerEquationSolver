// Marching-cubes count pass: one workgroup per 8^3 CELL block, one thread
// per cell. Case index from the 8 corner densities vs iso; the block's
// triangle total (shared atomic, order-independent int sum) lands in
// block_sums[wg] for the scan pass. Tables live in one int SSBO uploaded
// from core/marching_cubes_tables.hpp (single source of truth):
// [0,256) edge table, [256,4352) tri table 256x16, [4352,4608) tri counts.
[[vk::binding(0, 0)]] StructuredBuffer<float, Std430DataLayout> den;
[[vk::binding(1, 0)]] StructuredBuffer<int, Std430DataLayout> mc_tab;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> block_sums;

struct Params {
    int4 npts;      // grid points per axis (.xyz)
    int4 nblk;      // cell blocks per axis (.xyz)
    float4 box_min; // .xyz
    float4 cell_h;  // .xyz
    float iso;
    int max_tris;
};
[[vk::binding(3, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

static const int3 kCorner[8] = {
    int3(0, 0, 0), int3(1, 0, 0), int3(1, 1, 0), int3(0, 1, 0),
    int3(0, 0, 1), int3(1, 0, 1), int3(1, 1, 1), int3(0, 1, 1)
};

groupshared uint s_total;

int flat_at(int3 p) { return p.x + params.npts.x * (p.y + params.npts.y * p.z); }

[shader("compute")]
[numthreads(512, 1, 1)]
void main(uint3 grp : SV_GroupID, uint lidx : SV_GroupIndex) {
    if (lidx == 0u) {
        s_total = 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    int wg = int(grp.x);
    int3 blk = int3(wg % params.nblk.x, (wg / params.nblk.x) % params.nblk.y,
                    wg / (params.nblk.x * params.nblk.y));
    int li = int(lidx);
    int3 cell = blk * 8 + int3(li % 8, (li / 8) % 8, li / 64);

    if (all(cell < (params.npts.xyz - 1))) {
        int cube = 0;
        for (int c = 0; c < 8; ++c) {
            if (den[flat_at(cell + kCorner[c])] < params.iso) {
                cube |= 1 << c;
            }
        }
        uint tris = uint(mc_tab[4352 + cube]);
        if (tris > 0u) {
            InterlockedAdd(s_total, tris);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lidx == 0u) {
        block_sums[wg] = s_total;
    }
}
