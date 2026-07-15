// One dipole half-kick psi <- exp(-i theta axis.r) psi, per grid cell.
// The grid geometry (box_min/cell_h/axis) is passed as vec4 in the std140
// UBO -- padding to 16 bytes sidesteps std140's vec3 alignment subtleties;
// only .xyz is used.
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> psi;

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
    // nx, ny power-of-2 (FFT grid): unflatten with mask/shift, not a runtime
    // integer div/mod by a uniform (~20+ cyc on Pascal). Same indices.
    uint lx = uint(firstbithigh(uint(params.nx)));
    uint ly = uint(firstbithigh(uint(params.ny)));
    int i = int(idx & (uint(params.nx) - 1u));
    int j = int((idx >> lx) & (uint(params.ny) - 1u));
    int k = int(idx >> (lx + ly));
    float3 r = params.box_min.xyz +
               float3(float(i), float(j), float(k)) * params.cell_h.xyz;
    float ang = -params.theta * dot(params.axis.xyz, r);
    float2 w = float2(cos(ang), sin(ang));
    float2 a = psi[idx];
    psi[idx] = float2(a.x * w.x - a.y * w.y, a.x * w.y + a.y * w.x);
}
