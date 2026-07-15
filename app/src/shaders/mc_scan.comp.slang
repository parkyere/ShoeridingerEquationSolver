// Marching-cubes scan pass: ONE workgroup exclusive-scans block_sums into
// block_offsets (deterministic integer arithmetic -- every emit slot is a
// fixed function of the field, never of dispatch order). Thread t serially
// scans its contiguous chunk; a Hillis-Steele pass over the 1024 chunk
// totals stitches them. Thread 0 also writes the indirect draw command
// (vertexCount = 3 * min(total, max_tris)) and the raw/clamped totals.
[[vk::binding(0, 0)]] StructuredBuffer<uint, Std430DataLayout> block_sums;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint, Std430DataLayout> block_offsets;
// [0]=vertexCount [1]=instanceCount [2]=firstVertex [3]=firstInstance
// [4]=raw triangle total [5]=clamped triangle total
[[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> draw;

struct Params {
    int nblocks;
    int max_tris;
};
[[vk::binding(3, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared uint s_chunk[1024];

[shader("compute")]
[numthreads(1024, 1, 1)]
void main(uint lidx : SV_GroupIndex) {
    uint t = lidx;
    uint chunk = (uint(params.nblocks) + 1023u) / 1024u;
    uint lo = t * chunk;
    uint hi = min(lo + chunk, uint(params.nblocks));

    uint sum = 0u;
    for (uint b = lo; b < hi; ++b) {
        sum += block_sums[b];
    }
    s_chunk[t] = sum;
    GroupMemoryBarrierWithGroupSync();

    // Hillis-Steele inclusive scan over chunk totals, then shift down.
    for (uint stride = 1u; stride < 1024u; stride <<= 1u) {
        uint v = s_chunk[t];
        uint add = (t >= stride) ? s_chunk[t - stride] : 0u;
        GroupMemoryBarrierWithGroupSync();
        s_chunk[t] = v + add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint base = (t > 0u) ? s_chunk[t - 1u] : 0u;

    uint run = base;
    for (uint b = lo; b < hi; ++b) {
        block_offsets[b] = run;
        run += block_sums[b];
    }

    if (t == 1023u) {
        uint total = s_chunk[1023];
        uint clamped = min(total, uint(params.max_tris));
        draw[0] = 3u * clamped;
        draw[1] = 1u;
        draw[2] = 0u;
        draw[3] = 0u;
        draw[4] = total;
        draw[5] = clamped;
    }
}
