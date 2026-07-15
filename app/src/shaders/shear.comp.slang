// One shear of the three-shear exact rotation (magnetic paramagnetic term).
// In the mixed representation -- FFT'd along freq_axis, real along coord_axis
// -- each element gets a phase exp(-i k_f * coeff * coord), shifting every
// line by coeff times its perpendicular coordinate. std140 UBO carries the
// grid + shear geometry (binding 1); scalars pack tightly, so the block
// matches the host struct field order.
import complex;
import grid;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;

struct Params {
    uint n;
    int nx;
    int ny;
    int nz;
    int freq_axis;   // 0=x,1=y,2=z: the FFT (freq) axis
    int coord_axis;  // the shift-coordinate axis (perpendicular)
    int nf;          // length of the freq axis
    float kscale;    // 2*pi / L_freq
    float cmin;      // shift-coordinate axis min
    float ch;        // shift-coordinate axis spacing
    float coeff;     // shear coefficient
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) {
        return;
    }
    int3 v = unflatten_moddiv(idx, params.nx, params.ny);
    int e[3] = { v.x, v.y, v.z };
    int f = e[params.freq_axis];
    int p = e[params.coord_axis];
    int sf = (f < params.nf / 2) ? f : f - params.nf;  // signed fftfreq
    float kf = params.kscale * float(sf);
    float d = params.coeff * (params.cmin + float(p) * params.ch);
    psi[idx] = psi[idx] * cis(-kf * d);
}
