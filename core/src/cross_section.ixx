module;
#include <complex>
#include <algorithm>
#include <cmath>
export module ses.cross_section;

import ses.colormap;
import ses.vec;
import ses.grid;


// Cross-section display logic: the clip-plane ray-interval clamp, the slice
// quad geometry, and the slice colour mapping. This is the SINGLE SOURCE OF
// TRUTH the shaders mirror -- volume.frag (clip), slice.vert (quad),
// slice.frag (sample + colour). Only the final composited image stays
// visual-only (dump-frame), exactly as ses.volume backs the volume
// raymarch.


export namespace ses {

// Restrict a ray's [tn, t_stop] parameter interval to the half-space
// sign*(p[axis]-offset) <= 0, where p = eye + t*dir (only the axis components
// enter). visible = false when the whole interval is on the cut-away side.
struct ClipInterval {
    double tn;
    double t_stop;
    bool visible;
};
inline ClipInterval clip_ray_interval(double tn, double t_stop, double eye_c,
                                      double dir_c, double sign,
                                      double offset) noexcept {
    const double a = sign * dir_c;
    const double b = sign * (offset - eye_c);
    if (std::abs(a) < 1e-8) {
        return ClipInterval{tn, t_stop, b >= 0.0};  // parallel: one side whole
    }
    const double tp = b / a;
    if (a > 0.0) {
        t_stop = std::min(t_stop, tp);  // keep t <= tp
    } else {
        tn = std::max(tn, tp);  // keep t >= tp
    }
    return ClipInterval{tn, t_stop, t_stop > tn};
}

// World position of corner k (0..5, two CCW triangles) of the slice quad: the
// plane through `offset` along `axis`, spanning the grid box in the other two
// axes.
inline Vec3d slice_quad_corner(int axis, double offset, const Grid3D& g,
                               int k) noexcept {
    static const double kST[6][2] = {{0, 0}, {1, 0}, {1, 1},
                                     {0, 0}, {1, 1}, {0, 1}};
    const int u = (axis + 1) % 3;
    const int w = (axis + 2) % 3;
    const Grid1D* ax[3] = {&g.x, &g.y, &g.z};
    double c[3];
    c[axis] = offset;
    c[u] = ax[u]->xmin + kST[k][0] * (ax[u]->xmax - ax[u]->xmin);
    c[w] = ax[w]->xmin + kST[k][1] * (ax[w]->xmax - ax[w]->xmin);
    return Vec3d{c[0], c[1], c[2]};
}

// Slice colour + sheet opacity for a sampled psi. map: 0 density (magnitude
// ramp), 1 Re(psi) diverging (warm +, cool -), 2 phase (wheel tinted by
// |psi|). Mirrors slice.frag's `col`/`alpha`.
struct SliceShade {
    Rgb col;
    double alpha;
};
inline SliceShade slice_shade(int map, std::complex<double> psi,
                              double inv_peak) noexcept {
    const double dens = std::norm(psi) * inv_peak;
    const double amp = std::sqrt(std::max(inv_peak, 0.0));
    Rgb col{};
    double bright;
    if (map == 1) {  // Re(psi), diverging about a dark midpoint
        const double r = std::clamp(psi.real() * amp, -1.0, 1.0);
        const double m = std::abs(r);
        const Rgb tgt = (r >= 0.0) ? Rgb{1.0, 0.55, 0.15} : Rgb{0.15, 0.45, 1.0};
        col = Rgb{0.03 + (tgt.r - 0.03) * m, 0.03 + (tgt.g - 0.03) * m,
                  0.03 + (tgt.b - 0.03) * m};
        bright = m;
    } else if (map == 2) {  // phase
        col = phase_color(std::atan2(psi.imag(), psi.real()));
        bright = std::sqrt(std::clamp(dens, 0.0, 1.0));
        const double tint = 0.25 + 0.75 * bright;
        col = Rgb{col.r * tint, col.g * tint, col.b * tint};
    } else {  // density
        const double d = std::clamp(dens, 0.0, 1.0);
        col = magnitude_color(d);
        bright = d;
    }
    return SliceShade{col, std::clamp(0.45 + 0.5 * bright, 0.0, 0.95)};
}

}  // namespace ses
