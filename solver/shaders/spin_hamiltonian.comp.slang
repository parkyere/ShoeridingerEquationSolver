// Sparse matvec out = H*in for H = 1/2 sum_i B.sigma_i - J sum_bonds
// sigma_i.sigma_j (16 sites). Each thread owns amplitude m and gathers 16
// single-flip (field) + nb double-flip (bond off-diagonal) + diagonal terms.
// fp32 mirror of ses::hamiltonian_apply. CONTRACT: vkcheck check_spin_hamiltonian.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> inbuf;
[[vk::binding(1, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> outbuf;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint, Std430DataLayout> bonds;  // 2*nb

struct Params {
    float bx, by, bz, j;
    uint nb, n, pad0, pad1;
};
[[vk::binding(3, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint m = tid.x;
    if (m >= params.n) {
        return;
    }
    Complex cm = inbuf[m];
    Complex field = Complex(0.0, 0.0);
    for (uint i = 0u; i < 16u; ++i) {
        uint bi = 1u << i;
        bool up = (m & bi) == 0u;
        Complex flip = inbuf[m ^ bi];
        // sigma_x: flip; sigma_y: (up ? -i : +i)*flip; sigma_z: (up?+1:-1)*cm
        Complex fy = up ? Complex(flip.im, -flip.re) : Complex(-flip.im, flip.re);
        field = field + flip * params.bx + fy * params.by +
                cm * (params.bz * (up ? 1.0 : -1.0));
    }
    Complex acc = field * 0.5;
    for (uint k = 0u; k < params.nb; ++k) {
        uint a = bonds[2u * k];
        uint b = bonds[2u * k + 1u];
        bool sa = (m & (1u << a)) != 0u;
        bool sb = (m & (1u << b)) != 0u;
        Complex term = cm * (sa == sb ? 1.0 : -1.0);
        if (sa != sb) {
            term = term + inbuf[(m ^ (1u << a)) ^ (1u << b)] * 2.0;
        }
        acc = acc + term * (-params.j);
    }
    outbuf[m] = acc;
}
