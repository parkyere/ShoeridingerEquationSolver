#pragma once

// Exact, unitary rotation of a 3D field about a coordinate axis via the
// three-shear (Paeth) decomposition in the perpendicular plane,
//     R(theta) = Bshear(-tan(theta/2)) . Cshear(sin theta) . Bshear(-tan(theta/2)),
// each shear a per-line shift applied exactly by the Fourier shift theorem
// (X(k) *= e^{-i k d}). Norm-conserving, no interpolation blur; |theta| < pi.
// Used for the paramagnetic (B/2) L_axis factor of the magnetic propagator.

#include <core/complex.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
import ses.grid;
#include <core/spectral.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

namespace rotation_detail {

inline constexpr const Grid1D& axis_grid(const Grid3D& g, int a) noexcept {
    return a == 0 ? g.x : (a == 1 ? g.y : g.z);
}

// Shift every line along `freq_axis` by d = coeff * (its `coord_axis`
// coordinate), exactly, via the Fourier shift theorem X(k) *= e^{-i k d}.
// freq_axis != coord_axis (both lie in the rotation plane).
inline void axis_shear(Field3D& f, int freq_axis, int coord_axis, double coeff) {
    const Grid3D& g = f.grid();
    const int n[3] = {g.x.n, g.y.n, g.z.n};
    const int s[3] = {1, g.x.n, g.x.n * g.y.n};  // x-fastest strides
    const Grid1D& ca = axis_grid(g, coord_axis);
    const std::vector<double> kf = wavenumbers(axis_grid(g, freq_axis));
    const int nf = n[freq_axis];
    const int sf = s[freq_axis];

    int perp[2];  // the two axes perpendicular to freq_axis (ascending)
    int pc = 0;
    for (int a = 0; a < 3; ++a) {
        if (a != freq_axis) {
            perp[pc++] = a;
        }
    }

    std::vector<Complex<double>> line(static_cast<std::size_t>(nf));
    std::vector<Complex<double>>& a = f.data();
    for (int v = 0; v < n[perp[1]]; ++v) {
        for (int u = 0; u < n[perp[0]]; ++u) {
            const int base = u * s[perp[0]] + v * s[perp[1]];
            const int cidx = (coord_axis == perp[0]) ? u : v;
            const double d = coeff * ca.coord(cidx);
            for (int t = 0; t < nf; ++t) {
                line[static_cast<std::size_t>(t)] =
                    a[static_cast<std::size_t>(base + t * sf)];
            }
            fft(line);
            for (int t = 0; t < nf; ++t) {
                const double ph = -kf[static_cast<std::size_t>(t)] * d;
                line[static_cast<std::size_t>(t)] =
                    line[static_cast<std::size_t>(t)] *
                    Complex<double>{std::cos(ph), std::sin(ph)};
            }
            ifft(line);
            for (int t = 0; t < nf; ++t) {
                a[static_cast<std::size_t>(base + t * sf)] =
                    line[static_cast<std::size_t>(t)];
            }
        }
    }
}

// The two in-plane axes (b, c) for a rotation about `axis`, right-handed
// (b x c = axis): z->(x,y), x->(y,z), y->(z,x).
inline constexpr void plane_axes(int axis, int& b, int& c) noexcept {
    b = (axis + 1) % 3;
    c = (axis + 2) % 3;
}

}  // namespace rotation_detail

// Rotate the field about coordinate `axis` (0=x,1=y,2=z) by theta (active,
// right-handed), exactly and unitarily. |theta| < pi.
inline void rotate_axis(Field3D& f, int axis, double theta) {
    int b = 0;
    int c = 0;
    rotation_detail::plane_axes(axis, b, c);
    const double t = std::tan(0.5 * theta);
    const double s = std::sin(theta);
    rotation_detail::axis_shear(f, b, c, -t);
    rotation_detail::axis_shear(f, c, b, s);
    rotation_detail::axis_shear(f, b, c, -t);
}

// Rotate about +z (the common case).
inline void rotate_z(Field3D& f, double theta) { rotate_axis(f, 2, theta); }

}  // namespace ses
