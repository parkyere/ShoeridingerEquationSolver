// data <- scale * data (the fp32 drift renormalization). One thread per
// element. Bindings: data(0) read-write std430 SSBO, std140 UBO n + scale (1).
// std140/std430 pinned per-buffer so the layout byte-matches the C++ upload.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> data;

struct Params {
    uint n;
    float scale;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i < params.n) {
        data[i] = params.scale * data[i];
    }
}
