// Mean-field lattice: refresh every site's Bloch vector from its spinor before a
// step's checkerboard sweeps (mirrors the per-step snapshot in spinlattice_step).
// CONTRACT: vkcheck check_spin_mf.
import complex;
import spin;

[[vk::binding(0, 0)]] RWStructuredBuffer<float4, Std430DataLayout> spinors;
[[vk::binding(1, 0)]] RWStructuredBuffer<float, Std430DataLayout> bloch;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= 16u) {
        return;
    }
    float3 b = bloch_of(load_spinor(spinors[i]));
    bloch[3u * i + 0u] = b.x;
    bloch[3u * i + 1u] = b.y;
    bloch[3u * i + 2u] = b.z;
}
