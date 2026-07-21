// One x-bond parity group of PeierlsLattice2D: exact 2x2 rotation on disjoint
// bonds (i,j)-(i+1,j) with Peierls link u. mix = i*sin (real) or sinh (imag).
// One thread per LEFT cell (i even/odd = parity); disjoint -> race-free.
// CONTRACT: vkcheck check_lattice2d_step / check_lattice2d_relax.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(1, 0)]] StructuredBuffer<Complex, Std430DataLayout> link_x;

struct Params {
    uint nx;
    uint ny;
    uint parity;
    float c;
    float mix_re;
    float mix_im;
    float pad0;
    float pad1;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.nx * params.ny) {
        return;
    }
    uint i = idx % params.nx;
    if ((i & 1u) != params.parity || i + 1u >= params.nx) {
        return;
    }
    uint j = idx / params.nx;
    uint row = j * params.nx;
    Complex mix = Complex(params.mix_re, params.mix_im);
    Complex u = link_x[row + i];
    Complex pa = psi[row + i];
    Complex pb = psi[row + i + 1u];
    psi[row + i] = pa * params.c + mix * u * pb;
    psi[row + i + 1u] = mix * conj(u) * pa + pb * params.c;
}
