// Copy the psi SSBO into the 3D volume texture the renderer samples, one
// texel per grid cell. The volume is RG16F (re, im): display-only fp16 --
// physics stays in the fp32 SSBO; the store converts (round-to-nearest).
// psi in (binding 0), image3D out (binding 1), grid dims UBO (binding 2).
import complex;
import grid;

[[vk::binding(0, 0)]] StructuredBuffer<Complex, Std430DataLayout> psi;
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
    int3 c = unflatten_pow2(tid.x, params.nx, params.ny);
    img[c] = float4(psi[idx].v, 0.0, 0.0);
}
