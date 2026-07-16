// Bohmian velocity field v = j / rho, j = Im(conj(psi) grad psi), computed in
// fp32 straight from the PHYSICS psi SSBO (NOT the fp16 display volume). A real
// eigenstate then gives v ~ 0 exactly, so the streaklines stay still (the fp16
// display volume broke that cancellation -> spurious pulsating radial flow).
// Written to a LOW-RES rgba16f volume the streaklines trilinear-sample; v is
// smooth so coarse resolution is fine, and it keeps the memory cost small.
// Matches ses.flow::probability_current.
import complex;

[[vk::binding(0, 0)]] StructuredBuffer<Complex, Std430DataLayout> psi;  // fp32
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture3D<float4> vel;  // M^3

struct Params {
    int nx;
    int ny;
    int nz;
    int m;         // velocity-volume side (low-res)
    float hx;      // full-res cell spacing (au)
    float hy;
    float hz;
    float vmax;    // clamp the low-density blowup
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

Complex psi_at(int i, int j, int k) {
    i = clamp(i, 0, params.nx - 1);
    j = clamp(j, 0, params.ny - 1);
    k = clamp(k, 0, params.nz - 1);
    return psi[i + params.nx * (j + params.ny * k)];  // x-fastest (grid.flat)
}

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 gid : SV_DispatchThreadID) {
    int3 I = int3(gid);
    if (I.x >= params.m || I.y >= params.m || I.z >= params.m) {
        return;
    }
    // Coarse sample point mapped to the full-res grid (block centre).
    int3 s = int3(params.nx / params.m, params.ny / params.m, params.nz / params.m);
    int i = I.x * s.x + s.x / 2;
    int j = I.y * s.y + s.y / 2;
    int k = I.z * s.z + s.z / 2;

    Complex p = psi_at(i, j, k);
    float rho = norm2(p);
    Complex dx = (psi_at(i + 1, j, k) - psi_at(i - 1, j, k)) / (2.0 * params.hx);
    Complex dy = (psi_at(i, j + 1, k) - psi_at(i, j - 1, k)) / (2.0 * params.hy);
    Complex dz = (psi_at(i, j, k + 1) - psi_at(i, j, k - 1)) / (2.0 * params.hz);
    // j = Im(conj(psi) grad psi) per component = mul_conj(psi, dpsi).im.
    float3 jc = float3(mul_conj(p, dx).im, mul_conj(p, dy).im, mul_conj(p, dz).im);
    float3 v = jc / max(rho, 1e-12);
    float sp = length(v);
    if (sp > params.vmax) {
        v *= params.vmax / sp;
    }
    vel[I] = float4(v, 0.0);
}
