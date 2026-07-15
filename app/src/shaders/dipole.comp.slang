// Dipole matrix element <to| r |from> = sum conj(to)*from * (x,y,z): three
// complex reductions (one per axis) folded in parallel and written as 6 floats
// per workgroup. Bindings: to(0) / from(3) readonly, partials(2) writeonly,
// std140 UBO(1) with vec4-padded grid geometry.
import complex;
import grid;

[[vk::binding(0, 0)]] StructuredBuffer<Complex, Std430DataLayout> fto;
[[vk::binding(3, 0)]] StructuredBuffer<Complex, Std430DataLayout> ffrom;
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

groupshared Complex sx[256];
groupshared Complex sy[256];
groupshared Complex sz[256];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID,
          uint tid : SV_GroupIndex,
          uint3 grp : SV_GroupID) {
    uint stride = WorkgroupCount().x * 256u;
    Complex ax = Complex(0.0, 0.0);
    Complex ay = Complex(0.0, 0.0);
    Complex az = Complex(0.0, 0.0);
    for (uint idx = gid.x; idx < params.n; idx += stride) {
        Complex c = mul_conj(fto[idx], ffrom[idx]);  // conj(to)*from
        int3 ijk = unflatten_moddiv(idx, params.nx, params.ny);
        float3 r = cell_pos(ijk, params.box_min.xyz, params.cell_h.xyz);
        ax = ax + c * r.x;
        ay = ay + c * r.y;
        az = az + c * r.z;
    }
    sx[tid] = ax;
    sy[tid] = ay;
    sz[tid] = az;
    GroupMemoryBarrierWithGroupSync();
    for (uint half_len = 128u; half_len > 0u; half_len >>= 1u) {
        if (tid < half_len) {
            sx[tid] = sx[tid] + sx[tid + half_len];
            sy[tid] = sy[tid] + sy[tid + half_len];
            sz[tid] = sz[tid] + sz[tid + half_len];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (tid == 0u) {
        uint g = grp.x;
        partials[6u * g + 0u] = sx[0].re;
        partials[6u * g + 1u] = sx[0].im;
        partials[6u * g + 2u] = sy[0].re;
        partials[6u * g + 3u] = sy[0].im;
        partials[6u * g + 4u] = sz[0].re;
        partials[6u * g + 5u] = sz[0].im;
    }
}
