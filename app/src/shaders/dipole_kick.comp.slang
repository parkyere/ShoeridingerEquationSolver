// One dipole half-kick psi <- exp(-i theta axis.r) psi, per grid cell.
// The grid geometry (box_min/cell_h/axis) is passed as vec4 in the std140
// UBO -- padding to 16 bytes sidesteps std140's vec3 alignment subtleties;
// only .xyz is used.
import complex;
import grid;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;

struct Params {
    uint n;
    int nx;
    int ny;
    float theta;
    float4 box_min;  // .xyz
    float4 cell_h;   // .xyz
    float4 axis;     // .xyz
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) {
        return;
    }
    int3 ijk = unflatten_pow2(idx, params.nx, params.ny);
    float3 r = cell_pos(ijk, params.box_min.xyz, params.cell_h.xyz);
    float ang = -params.theta * dot(params.axis.xyz, r);
    psi[idx] = psi[idx] * cis(ang);
}
