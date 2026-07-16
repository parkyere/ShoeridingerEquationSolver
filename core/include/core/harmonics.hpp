#pragma once

// Real spherical harmonics (l <= 5) and 3D orbital synthesis:
// psi = (u_nl(r)/r) Y_lm from the radial engine's u tables. Cartesian
// polynomial forms keep the nodal planes exact on the grid.

#include <complex>
#include <core/field.hpp>
import ses.grid;
#include <core/radial.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

// Y_lm normalization constants, hoisted out of real_spherical_harmonic:
// MSVC /fp:precise does not constant-fold sqrt(literal), so evaluating them
// inline would cost one libm call per grid point in fill_orbital.
namespace ynorm {
inline constexpr double kPi = 3.14159265358979323846;
inline const double sqrt_pi = std::sqrt(kPi);                    // l = 0
inline const double s3_4pi = std::sqrt(3.0 / (4.0 * kPi));       // l = 1
inline const double s15_pi = std::sqrt(15.0 / kPi);              // l = 2
inline const double s5_pi = std::sqrt(5.0 / kPi);
inline const double s35_2pi = std::sqrt(35.0 / (2.0 * kPi));     // l = 3
inline const double s105_pi = std::sqrt(105.0 / kPi);
inline const double s21_2pi = std::sqrt(21.0 / (2.0 * kPi));
inline const double s7_pi = std::sqrt(7.0 / kPi);
inline const double s35_pi = std::sqrt(35.0 / kPi);              // l = 4
inline const double s5_2pi = std::sqrt(5.0 / (2.0 * kPi));
inline const double s1_pi = std::sqrt(1.0 / kPi);
inline const double s1386_pi = std::sqrt(1386.0 / kPi);          // l = 5
inline const double s3465_pi = std::sqrt(3465.0 / kPi);
inline const double s770_pi = std::sqrt(770.0 / kPi);
inline const double s1155_pi = std::sqrt(1155.0 / kPi);
inline const double s165_pi = std::sqrt(165.0 / kPi);
inline const double s11_pi = std::sqrt(11.0 / kPi);
}  // namespace ynorm

// Real Y_lm evaluated at direction (x, y, z)/r. Convention (m = -l..l):
//   l=1: -1 -> y, 0 -> z, +1 -> x
//   l=2: -2 -> xy, -1 -> yz, 0 -> 3z^2-r^2, +1 -> zx, +2 -> x^2-y^2
//   l=3: -3 -> y(3x^2-y^2), -2 -> xyz, -1 -> y(5z^2-r^2), 0 -> z(5z^2-3r^2),
//        +1 -> x(5z^2-r^2), +2 -> z(x^2-y^2), +3 -> x(x^2-3y^2)
//   l=4: -4 -> xy(x^2-y^2), -3 -> yz(3x^2-y^2), -2 -> xy(7z^2-r^2),
//        -1 -> yz(7z^2-3r^2), 0 -> 35z^4-30z^2r^2+3r^4, +1 -> xz(7z^2-3r^2),
//        +2 -> (x^2-y^2)(7z^2-r^2), +3 -> xz(x^2-3y^2), +4 -> x^4-6x^2y^2+y^4
//   l=5 (canonical (x+iy)^m real/imag parts): -5 -> y(5x^4-10x^2y^2+y^4),
//        -4 -> xy(x^2-y^2)z, -3 -> y(3x^2-y^2)(9z^2-r^2), -2 -> xyz(3z^2-r^2),
//        -1 -> y(21z^4-14z^2r^2+r^4), 0 -> z(63z^4-70z^2r^2+15r^4),
//        +1 -> x(21z^4-14z^2r^2+r^4), +2 -> (x^2-y^2)z(3z^2-r^2),
//        +3 -> x(x^2-3y^2)(9z^2-r^2), +4 -> (x^4-6x^2y^2+y^4)z,
//        +5 -> x(x^4-10x^2y^2+5y^4)
inline double real_spherical_harmonic(int l, int m, double x, double y, double z) noexcept {
    const double r2 = x * x + y * y + z * z;
    if (l == 0) {
        return 1.0 / (2.0 * ynorm::sqrt_pi);
    }
    if (r2 == 0.0) {
        return 0.0;  // direction undefined; l > 0 harmonics vanish with r^l
    }
    if (l == 1) {
        const double c = ynorm::s3_4pi / std::sqrt(r2);
        switch (m) {
            case -1: return c * y;
            case 0: return c * z;
            default: return c * x;
        }
    }
    if (l == 2) {
        const double c = ynorm::s15_pi / r2;
        switch (m) {
            case -2: return 0.5 * c * x * y;
            case -1: return 0.5 * c * y * z;
            case 0: return 0.25 * ynorm::s5_pi * (3.0 * z * z - r2) / r2;
            case 1: return 0.5 * c * z * x;
            default: return 0.25 * c * (x * x - y * y);
        }
    }
    if (l == 3) {
        const double r3 = r2 * std::sqrt(r2);
        switch (m) {
            case -3: return 0.25 * ynorm::s35_2pi * y * (3.0 * x * x - y * y) / r3;
            case -2: return 0.5 * ynorm::s105_pi * x * y * z / r3;
            case -1: return 0.25 * ynorm::s21_2pi * y * (5.0 * z * z - r2) / r3;
            case 0: return 0.25 * ynorm::s7_pi * z * (5.0 * z * z - 3.0 * r2) / r3;
            case 1: return 0.25 * ynorm::s21_2pi * x * (5.0 * z * z - r2) / r3;
            case 2: return 0.25 * ynorm::s105_pi * z * (x * x - y * y) / r3;
            default: return 0.25 * ynorm::s35_2pi * x * (x * x - 3.0 * y * y) / r3;
        }
    }
    if (l == 4) {
        // l == 4; default case is m = +4.
        const double r4 = r2 * r2;
        switch (m) {
            case -4: return 0.75 * ynorm::s35_pi * x * y * (x * x - y * y) / r4;
            case -3: return 0.75 * ynorm::s35_2pi * y * z * (3.0 * x * x - y * y) / r4;
            case -2: return 0.75 * ynorm::s5_pi * x * y * (7.0 * z * z - r2) / r4;
            case -1: return 0.75 * ynorm::s5_2pi * y * z * (7.0 * z * z - 3.0 * r2) / r4;
            case 0:
                return (3.0 / 16.0) * ynorm::s1_pi *
                       (35.0 * z * z * z * z - 30.0 * z * z * r2 + 3.0 * r2 * r2) / r4;
            case 1: return 0.75 * ynorm::s5_2pi * x * z * (7.0 * z * z - 3.0 * r2) / r4;
            case 2: return 0.375 * ynorm::s5_pi * (x * x - y * y) * (7.0 * z * z - r2) / r4;
            case 3: return 0.75 * ynorm::s35_2pi * x * z * (x * x - 3.0 * y * y) / r4;
            default:
                return (3.0 / 16.0) * ynorm::s35_pi *
                       (x * x * x * x - 6.0 * x * x * y * y + y * y * y * y) / r4;
        }
    }
    // l == 5; default case is m = +5 (orthonormality pinned by tests).
    const double r5 = r2 * r2 * std::sqrt(r2);
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    const double zpoly = 21.0 * z2 * z2 - 14.0 * z2 * r2 + r2 * r2;  // m = +-1
    switch (m) {
        case -5:
            return (1.0 / 32.0) * ynorm::s1386_pi *
                   y * (5.0 * x2 * x2 - 10.0 * x2 * y2 + y2 * y2) / r5;
        case -4:
            return (1.0 / 16.0) * ynorm::s3465_pi *
                   4.0 * x * y * (x2 - y2) * z / r5;
        case -3:
            return (1.0 / 32.0) * ynorm::s770_pi *
                   y * (3.0 * x2 - y2) * (9.0 * z2 - r2) / r5;
        case -2:
            return (1.0 / 8.0) * ynorm::s1155_pi *
                   2.0 * x * y * z * (3.0 * z2 - r2) / r5;
        case -1:
            return (1.0 / 16.0) * ynorm::s165_pi * y * zpoly / r5;
        case 0:
            return (1.0 / 16.0) * ynorm::s11_pi *
                   z * (63.0 * z2 * z2 - 70.0 * z2 * r2 + 15.0 * r2 * r2) / r5;
        case 1:
            return (1.0 / 16.0) * ynorm::s165_pi * x * zpoly / r5;
        case 2:
            return (1.0 / 8.0) * ynorm::s1155_pi *
                   (x2 - y2) * z * (3.0 * z2 - r2) / r5;
        case 3:
            return (1.0 / 32.0) * ynorm::s770_pi *
                   x * (x2 - 3.0 * y2) * (9.0 * z2 - r2) / r5;
        case 4:
            return (1.0 / 16.0) * ynorm::s3465_pi *
                   (x2 * x2 - 6.0 * x2 * y2 + y2 * y2) * z / r5;
        default:
            return (1.0 / 32.0) * ynorm::s1386_pi *
                   x * (x2 * x2 - 10.0 * x2 * y2 + 5.0 * y2 * y2) / r5;
    }
}

// psi = (u(r)/r) Y_lm with u linearly interpolated (u/r -> u[0]/h as r -> 0).
// fill_orbital writes the UN-normalized field; synthesize_orbital normalizes.
// CONTRACT: core/projection.hpp's deposit shape mirrors this interpolation.
inline void fill_orbital(Field3D& psi, const Grid3D& g, const RadialGrid& rg,
                         const std::vector<double>& u, int l, int m) noexcept {
    const double h = rg.h();
    // Disjoint z-slabs: bitwise-deterministic parallelism (project rule).
#pragma omp parallel for
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
                psi(i, j, k) = std::complex<double>{value, 0.0};
            }
        }
    }
}

inline Field3D synthesize_orbital(const Grid3D& g, const RadialGrid& rg,
                                  const std::vector<double>& u, int l, int m) {
    Field3D psi{g};
    fill_orbital(psi, g, rg, u, l, m);
    normalize(psi);
    return psi;
}

}  // namespace ses
