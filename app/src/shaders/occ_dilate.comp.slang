// Occupancy dilation: 3x3x3 neighborhood max, so a block adjacent to any
// occupied block also reads occupied. With NEAREST sampling and block-length
// skips, a ray can then never jump over cloud it should have marched.
[[vk::binding(0, 0)]] [[vk::image_format("r32f")]] RWTexture3D<float4> dst;
[[vk::binding(1, 0)]] [[vk::image_format("r32f")]] RWTexture3D<float4> src;

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 gid : SV_DispatchThreadID) {
    int3 c = int3(gid);
    uint w, h, d;
    dst.GetDimensions(w, h, d);
    int3 size = int3(w, h, d);
    if (any(c >= size)) {
        return;
    }
    float m = 0.0;
    for (int k = -1; k <= 1; ++k) {
        for (int j = -1; j <= 1; ++j) {
            for (int i = -1; i <= 1; ++i) {
                int3 q = clamp(c + int3(i, j, k), int3(0, 0, 0), size - 1);
                m = max(m, src[q].r);
            }
        }
    }
    dst[c] = float4(m, 0.0, 0.0, 0.0);
}
