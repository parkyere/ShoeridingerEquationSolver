// Inverse of bridge_store, for verification only: read the volume texture
// back into an SSBO so the bridge can be checked without a 3D-texture
// readback. The RG16F texel quantizes psi, so the round-trip is checked
// against a CPU-side fp16 round-to-nearest of the SSBO contents.
// out SSBO (binding 0), image3D in (binding 1), grid dims UBO (binding 2).
[[vk::binding(0, 0)]] RWStructuredBuffer<float2, Std430DataLayout> outp;
[[vk::binding(1, 0)]] [[vk::image_format("rg16f")]] RWTexture3D<float4> img;

struct Params {
    int nx;
    int ny;
    int nz;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    int idx = int(tid.x);
    if (idx >= params.nx * params.ny * params.nz) {
        return;
    }
    // nx, ny power-of-2 (FFT grid): unflatten with mask/shift, not a runtime
    // integer div/mod by a uniform (~20+ cyc on Pascal). Same indices.
    int lx = int(firstbithigh(uint(params.nx)));
    int ly = int(firstbithigh(uint(params.ny)));
    int i = idx & (params.nx - 1);
    int j = (idx >> lx) & (params.ny - 1);
    int k = idx >> (lx + ly);
    outp[idx] = img[int3(i, j, k)].xy;
}
