#version 450

// Streakline fragment: the vertex already carries premultiplied white
// (rgb = alpha), so just pass it through -- the premultiplied-alpha blend
// composites the fading white line over the HDR scene.
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 frag;

void main() {
    frag = v_color;
}
