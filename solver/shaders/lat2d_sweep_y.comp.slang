// One y-bond parity group of PeierlsLattice2D: exact 2x2 rotation on disjoint
// bonds (i,j)-(i,j+1) with Peierls link u. mix = i*sin (real) or sinh (imag).
// One thread per LOWER cell (j even/odd = parity); disjoint -> race-free.
// CONTRACT: vkcheck check_lattice2d_step / check_lattice2d_relax.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(1, 0)]] StructuredBuffer<Complex, Std430DataLayout> link_y;

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
    uint j = idx / params.nx;
    if ((j & 1u) != params.parity || j + 1u >= params.ny) {
        return;
    }
    uint i = idx % params.nx;
    uint lo = j * params.nx + i;
    uint hi = (j + 1u) * params.nx + i;
    Complex mix = Complex(params.mix_re, params.mix_im);
    Complex u = link_y[lo];
    Complex pa = psi[lo];
    Complex pb = psi[hi];
    psi[lo] = pa * params.c + mix * u * pb;
    psi[hi] = mix * conj(u) * pa + pb * params.c;
}
