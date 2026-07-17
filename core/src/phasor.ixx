module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.phasor;
export import ses.field;
export import ses.grid;


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
    for (int i = 0; i < g.n; ++i) {
        const double a = r_scale * std::abs(psi[i]);
        const std::size_t o = static_cast<std::size_t>(3 * i);
        v[o + 0] = static_cast<float>(g.coord(i));
        v[o + 1] = static_cast<float>(a * psi[i].real());
        v[o + 2] = static_cast<float>(a * psi[i].imag());
    }
    return v;
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
