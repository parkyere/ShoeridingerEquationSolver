// Apply ONE fused k-qubit gate to the exact 2^16 state: one thread per orbit
// (the 2^k amplitudes sharing the non-gate bits), gather -> dense 2^k x 2^k
// matmul -> scatter. Orbits partition the state so threads never race; this is
// the GPU mirror of ses::apply_fused. qpack = qubit list, 4 bits each (k<=6).
// CONTRACT: vkcheck check_spin_fused_gate.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> state;
[[vk::binding(1, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> gmat;

struct Params {
    uint n_orbits;
    uint k;
    uint qpack;
    uint pad;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= params.n_orbits) {
        return;
    }
    uint k = params.k;
    uint dk = 1u << k;
    // Mask of the gate's qubit bit positions.
    uint qmask = 0u;
    for (uint b = 0u; b < k; ++b) {
        qmask |= 1u << ((params.qpack >> (4u * b)) & 0xFu);
    }
    // base = insert 0 at each gate bit position; t's bits fill the rest (low->high).
    uint base = 0u;
    uint tb = 0u;
    for (uint p = 0u; p < 16u; ++p) {
        if (((qmask >> p) & 1u) == 0u) {
            if (((t >> tb) & 1u) != 0u) {
                base |= 1u << p;
            }
            tb++;
        }
    }
    // gather the orbit's 2^k amplitudes
    Complex amp[64];
    for (uint a = 0u; a < dk; ++a) {
        uint idx = base;
        for (uint b = 0u; b < k; ++b) {
            if (((a >> b) & 1u) != 0u) {
                idx |= 1u << ((params.qpack >> (4u * b)) & 0xFu);
            }
        }
        amp[a] = state[idx];
    }
    // out = mat * amp, then scatter
    for (uint r = 0u; r < dk; ++r) {
        Complex acc = Complex(0.0, 0.0);
        for (uint c = 0u; c < dk; ++c) {
            acc = acc + gmat[r * dk + c] * amp[c];
        }
        uint idx = base;
        for (uint b = 0u; b < k; ++b) {
            if (((r >> b) & 1u) != 0u) {
                idx |= 1u << ((params.qpack >> (4u * b)) & 0xFu);
            }
        }
        state[idx] = acc;
    }
}
