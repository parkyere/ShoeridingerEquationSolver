// Streakline fragment: the vertex already carries premultiplied white
// (rgb = alpha), so just pass it through -- the premultiplied-alpha blend
// composites the fading white line over the HDR scene.
struct FSIn {
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 v_color : TEXCOORD0;
};

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    return input.v_color;
}
