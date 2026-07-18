module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.phasor;
export import ses.colormap;
export import ses.field;
export import ses.grid;
import ses.parallel;


// Phasor-curve geometry for the 1D scenes: the wavefunction drawn as ONE 3D
// curve (adjacent grid points connected by a line strip), not a phase-colored
// field. Vertices are consecutive (x, y, z) float triples:
//
//     curve_i = ( x_i, r cos(phi), r sin(phi) )
//     r = r_scale * |psi_i|^2      (amplitude-SQUARED radius)
//     phi = arg(psi_i)             (phase as geometric twist about x)
//
// A stationary state rotates rigidly about the x axis at angular rate E_n:
// the phase is visible as rotation, no color involved. Float output is
// display-facing only (the render vertex format); physics stays double.


export namespace ses {

// Wavefunction phasor curve, one vertex per grid point, x ascending.
// Algebraic form of the polar mapping: with r = s |psi|^2 and phi = arg,
//     r cos(phi) = s |psi| Re(psi),   r sin(phi) = s |psi| Im(psi)
// -- identical values, no transcendentals (the 64k-point curve rebuilds
// every frame).
inline std::vector<float> phasor_curve(const Field1D& psi, double r_scale) {
    const Grid1D& g = psi.grid();
    std::vector<float> v(static_cast<std::size_t>(3 * g.n));
    parallel_for(g.n, [&](int i) {
        const double a = r_scale * std::abs(psi[i]);
        const std::size_t o = static_cast<std::size_t>(3 * i);
        v[o + 0] = static_cast<float>(g.coord(i));
        v[o + 1] = static_cast<float>(a * psi[i].real());
        v[o + 2] = static_cast<float>(a * psi[i].imag());
    });
    return v;
}

// The phasor tube's SHADOW on the z = 0 plane: the curve sweeps radius
// r = r_scale |psi|^2 about the x axis, so its silhouette projected onto
// the xy plane is the band y in [-r, +r] -- a TRIANGLE_STRIP, two vertices
// per grid point (lower edge, then upper: strip winding). Where |psi|^2 ~ 0
// the band is zero-height and disappears by construction.
inline std::vector<float> density_band(const Field1D& psi, double r_scale) {
    const Grid1D& g = psi.grid();
    std::vector<float> v(static_cast<std::size_t>(6 * g.n));
    parallel_for(g.n, [&](int i) {
        const float x = static_cast<float>(g.coord(i));
        const float r = static_cast<float>(r_scale * std::norm(psi[i]));
        const std::size_t o = static_cast<std::size_t>(6 * i);
        v[o + 0] = x;
        v[o + 1] = -r;
        v[o + 2] = 0.0f;
        v[o + 3] = x;
        v[o + 4] = r;
        v[o + 5] = 0.0f;
    });
    return v;
}

// Per-vertex rgba for density_band: the phase wheel (the SAME ses::phase_color
// the 3D volume view samples, so hue means the same thing in every scene),
// premultiplied for the overlay blend; both band vertices share the color.
inline std::vector<float> phase_band_colors(const Field1D& psi, float alpha) {
    const Grid1D& g = psi.grid();
    std::vector<float> c(static_cast<std::size_t>(8 * g.n));
    // The 64k atan2 calls are the priciest display math in the 1D scenes:
    // chunked across the pool (disjoint writes, deterministic).
    parallel_for(g.n, [&](int i) {
        const Rgb col = phase_color(std::arg(psi[i]));
        const std::size_t o = static_cast<std::size_t>(8 * i);
        const float r = static_cast<float>(col.r) * alpha;
        const float gg = static_cast<float>(col.g) * alpha;
        const float b = static_cast<float>(col.b) * alpha;
        c[o + 0] = r;
        c[o + 1] = gg;
        c[o + 2] = b;
        c[o + 3] = alpha;
        c[o + 4] = r;
        c[o + 5] = gg;
        c[o + 6] = b;
        c[o + 7] = alpha;
    });
    return c;
}

// Potential polyline in the z = 0 plane: (x_i, min(v_i * e_scale, y_clamp), 0).
// y_clamp keeps steep walls (parabola edges, barrier tops) inside the frame.
inline std::vector<float> potential_curve(const Grid1D& g, const std::vector<double>& pot,
                                          double e_scale, double y_clamp) {
    std::vector<float> v(static_cast<std::size_t>(3 * g.n));
    for (int i = 0; i < g.n; ++i) {
        const std::size_t o = static_cast<std::size_t>(3 * i);
        const double y = std::min(pot[static_cast<std::size_t>(i)] * e_scale, y_clamp);
        v[o + 0] = static_cast<float>(g.coord(i));
        v[o + 1] = static_cast<float>(y);
        v[o + 2] = 0.0f;
    }
    return v;
}

}  // namespace ses
