// |psi|^2 into a scalar R32 buffer: the marching-cubes field. One read of
// psi per cell; classify/emit then sample density (and its gradients)
// without touching the complex field again.
[[vk::binding(0, 0)]] StructuredBuffer<float2, Std430DataLayout> psi;
[[vk::binding(1, 0)]] RWStructuredBuffer<float, Std430DataLayout> den;

struct Params {
    uint n;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) {
        return;
    }
    float2 c = psi[idx];
    den[idx] = dot(c, c);
}
