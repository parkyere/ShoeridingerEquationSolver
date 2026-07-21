// Chebyshev recurrence term: T_k = (factor/R) H T_{k-1} - T_{k-2} (factor=2) and
// accumulate result += c_k T_k. hbuf = H*T_{k-1}. Params bound as a DYNAMIC UBO,
// one 256-byte slot per k. CONTRACT: vkcheck check_spin_chebyshev.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> hbuf;
[[vk::binding(1, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> tprev;
[[vk::binding(2, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> tnext;
[[vk::binding(3, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> result;

struct Params {
    float ckre, ckim, R, factor;
};
[[vk::binding(4, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint m = tid.x;
    Complex tn = hbuf[m] * (params.factor / params.R) - tprev[m];
    tnext[m] = tn;
    result[m] = result[m] + Complex(params.ckre, params.ckim) * tn;
}
