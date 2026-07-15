// Dipole matrix element <to| r |from> = sum conj(to)*from * (x,y,z): three
// complex reductions (one per axis) folded in parallel and written as 6 floats
// per workgroup. Bindings: to(0) / from(3) readonly, partials(2) writeonly,
// std140 UBO(1) with vec4-padded grid geometry.
[[vk::binding(0, 0)]] StructuredBuffer<float2, Std430DataLayout> fto;
[[vk::binding(3, 0)]] StructuredBuffer<float2, Std430DataLayout> ffrom;
[[vk::binding(2, 0)]] RWStructuredBuffer<float, Std430DataLayout> partials;

struct Params {
    uint n;
    int nx;
    int ny;
    int pad0;
    float4 box_min;  // .xyz
    float4 cell_h;   // .xyz
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

groupshared float2 sx[256];
groupshared float2 sy[256];
groupshared float2 sz[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID,
          uint tid : SV_GroupIndex,
          uint3 grp : SV_GroupID) {
    uint stride = WorkgroupCount().x * 256u;
    float2 ax = float2(0.0, 0.0);
    float2 ay = float2(0.0, 0.0);
    float2 az = float2(0.0, 0.0);
    for (uint idx = gid.x; idx < params.n; idx += stride) {
        float2 a = fto[idx];
        float2 b = ffrom[idx];
        float2 c = float2(a.x * b.x + a.y * b.y, a.x * b.y - a.y * b.x);  // conj(to)*from
        int i = int(idx) % params.nx;
        int j = (int(idx) / params.nx) % params.ny;
        int k = int(idx) / (params.nx * params.ny);
        float3 r = params.box_min.xyz +
                   float3(float(i), float(j), float(k)) * params.cell_h.xyz;
        ax += c * r.x;
        ay += c * r.y;
        az += c * r.z;
    }
    sx[tid] = ax;
    sy[tid] = ay;
    sz[tid] = az;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            sx[tid] += sx[tid + half_len];
            sy[tid] += sy[tid + half_len];
            sz[tid] += sz[tid + half_len];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        uint g = grp.x;
        partials[6u * g + 0u] = sx[0].x;
        partials[6u * g + 1u] = sx[0].y;
        partials[6u * g + 2u] = sy[0].x;
        partials[6u * g + 3u] = sy[0].y;
        partials[6u * g + 4u] = sz[0].x;
        partials[6u * g + 5u] = sz[0].y;
    }
}
