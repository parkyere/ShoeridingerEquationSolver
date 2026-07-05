#pragma once

// Real spherical harmonics (l <= 2) and 3D orbital synthesis (T7): with the
// radial engine's u_nl(r), the 3D eigenstate is EXACTLY psi = (u/r) Y_lm --
// separation of variables replaces the imaginary-time ladder for building
// the tracked manifold. Cartesian polynomial forms avoid trigonometry and
// keep the nodal planes exact on the grid.

#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/radial.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

// Real Y_lm evaluated at direction (x, y, z)/r. Convention (m = -l..l):
//   l=1: -1 -> y, 0 -> z, +1 -> x
//   l=2: -2 -> xy, -1 -> yz, 0 -> 3z^2-r^2, +1 -> zx, +2 -> x^2-y^2
inline double real_spherical_harmonic(int l, int m, double x, double y, double z) {
    constexpr double kPi = 3.14159265358979323846;
    const double r2 = x * x + y * y + z * z;
    if (l == 0) {
        return 1.0 / (2.0 * std::sqrt(kPi));
    }
    if (r2 == 0.0) {
        return 0.0;  // direction undefined; l > 0 harmonics vanish with r^l
    }
    if (l == 1) {
        const double c = std::sqrt(3.0 / (4.0 * kPi)) / std::sqrt(r2);
        switch (m) {
            case -1: return c * y;
            case 0: return c * z;
            default: return c * x;
        }
    }
    // l == 2
    const double c = std::sqrt(15.0 / kPi) / r2;
    switch (m) {
        case -2: return 0.5 * c * x * y;
        case -1: return 0.5 * c * y * z;
        case 0: return 0.25 * std::sqrt(5.0 / kPi) * (3.0 * z * z - r2) / r2;
        case 1: return 0.5 * c * z * x;
        default: return 0.25 * c * (x * x - y * y);
    }
}

// psi(x, y, z) = (u(r)/r) Y_lm, u linearly interpolated from the radial
// grid (u(0) = 0 pins the inner segment, so u/r -> u_0/h as r -> 0, the
// correct l = 0 limit; higher l vanish there through Y's r^l factor).
// Returns the grid-normalized field.
inline Field3D synthesize_orbital(const Grid3D& g, const RadialGrid& rg,
                                  const std::vector<double>& u, int l, int m) {
    const double h = rg.h();
    Field3D psi{g};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                const double r = std::sqrt(x * x + y * y + z * z);
                double u_over_r = 0.0;
                if (r < h) {
                    u_over_r = u[0] / h;
                } else if (r < rg.rmax) {
                    const double t = r / h - 1.0;  // r_i = (i+1) h
                    const std::size_t i0 = static_cast<std::size_t>(t);
                    const double frac = t - static_cast<double>(i0);
                    const double ui =
                        (i0 + 1 < u.size())
                            ? (1.0 - frac) * u[i0] + frac * u[i0 + 1]
                            : (1.0 - frac) * u[i0];  // outermost: u(rmax) = 0
                    u_over_r = ui / r;
                }
                const double value =
                    u_over_r * real_spherical_harmonic(l, m, x, y, z);
                psi(i, j, k) = Complex<double>{value, 0.0};
            }
        }
    }
    normalize(psi);
    return psi;
}

}  // namespace ses
