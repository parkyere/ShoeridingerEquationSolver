// Chebyshev bootstrap: T_1 = (H/R) T_0 = hbuf/R, and result = c0 T_0 + c1 T_1
// (T_0 = state). CONTRACT: vkcheck check_spin_chebyshev.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> stateb;
[[vk::binding(1, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> hbuf;
[[vk::binding(2, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> tcur;
[[vk::binding(3, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> result;

struct Params {
    float c0re, c0im, c1re, c1im;
    float R, pad0, pad1, pad2;
};
[[vk::binding(4, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint m = tid.x;
    Complex t1 = hbuf[m] * (1.0 / params.R);
    tcur[m] = t1;
    result[m] = Complex(params.c0re, params.c0im) * stateb[m] +
                Complex(params.c1re, params.c1im) * t1;
}
