#pragma once

// Real spherical harmonics (l <= 5) and 3D orbital synthesis: with the
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
    if (l == 2) {
        const double c = std::sqrt(15.0 / kPi) / r2;
        switch (m) {
            case -2: return 0.5 * c * x * y;
            case -1: return 0.5 * c * y * z;
            case 0: return 0.25 * std::sqrt(5.0 / kPi) * (3.0 * z * z - r2) / r2;
            case 1: return 0.5 * c * z * x;
            default: return 0.25 * c * (x * x - y * y);
        }
    }
    if (l == 3) {
        const double r3 = r2 * std::sqrt(r2);
        switch (m) {
            case -3: return 0.25 * std::sqrt(35.0 / (2.0 * kPi)) * y * (3.0 * x * x - y * y) / r3;
            case -2: return 0.5 * std::sqrt(105.0 / kPi) * x * y * z / r3;
            case -1: return 0.25 * std::sqrt(21.0 / (2.0 * kPi)) * y * (5.0 * z * z - r2) / r3;
            case 0: return 0.25 * std::sqrt(7.0 / kPi) * z * (5.0 * z * z - 3.0 * r2) / r3;
            case 1: return 0.25 * std::sqrt(21.0 / (2.0 * kPi)) * x * (5.0 * z * z - r2) / r3;
            case 2: return 0.25 * std::sqrt(105.0 / kPi) * z * (x * x - y * y) / r3;
            default: return 0.25 * std::sqrt(35.0 / (2.0 * kPi)) * x * (x * x - 3.0 * y * y) / r3;
        }
    }
    if (l == 4) {
        // l == 4 (g): the n = 5 shell's highest angular momentum. Same
        // Cartesian-polynomial / r^l convention; default case is m = +4.
        const double r4 = r2 * r2;
        switch (m) {
            case -4: return 0.75 * std::sqrt(35.0 / kPi) * x * y * (x * x - y * y) / r4;
            case -3: return 0.75 * std::sqrt(35.0 / (2.0 * kPi)) * y * z * (3.0 * x * x - y * y) / r4;
            case -2: return 0.75 * std::sqrt(5.0 / kPi) * x * y * (7.0 * z * z - r2) / r4;
            case -1: return 0.75 * std::sqrt(5.0 / (2.0 * kPi)) * y * z * (7.0 * z * z - 3.0 * r2) / r4;
            case 0:
                return (3.0 / 16.0) * std::sqrt(1.0 / kPi) *
                       (35.0 * z * z * z * z - 30.0 * z * z * r2 + 3.0 * r2 * r2) / r4;
            case 1: return 0.75 * std::sqrt(5.0 / (2.0 * kPi)) * x * z * (7.0 * z * z - 3.0 * r2) / r4;
            case 2: return 0.375 * std::sqrt(5.0 / kPi) * (x * x - y * y) * (7.0 * z * z - r2) / r4;
            case 3: return 0.75 * std::sqrt(35.0 / (2.0 * kPi)) * x * z * (x * x - 3.0 * y * y) / r4;
            default:
                return (3.0 / 16.0) * std::sqrt(35.0 / kPi) *
                       (x * x * x * x - 6.0 * x * x * y * y + y * y * y * y) / r4;
        }
    }
    // l == 5 (h): the n = 6 shell's highest angular momentum. Canonical
    // (x+iy)^m real / imaginary parts over r^5; default case is m = +5. The
    // constants and polynomials are verified harmonic and orthonormal over
    // the sphere (see tests/harmonics_test.cpp).
    const double r5 = r2 * r2 * std::sqrt(r2);
    const double x2 = x * x;
    const double y2 = y * y;
    const double z2 = z * z;
    const double zpoly = 21.0 * z2 * z2 - 14.0 * z2 * r2 + r2 * r2;  // m = +-1
    switch (m) {
        case -5:
            return (1.0 / 32.0) * std::sqrt(1386.0 / kPi) *
                   y * (5.0 * x2 * x2 - 10.0 * x2 * y2 + y2 * y2) / r5;
        case -4:
            return (1.0 / 16.0) * std::sqrt(3465.0 / kPi) *
                   4.0 * x * y * (x2 - y2) * z / r5;
        case -3:
            return (1.0 / 32.0) * std::sqrt(770.0 / kPi) *
                   y * (3.0 * x2 - y2) * (9.0 * z2 - r2) / r5;
        case -2:
            return (1.0 / 8.0) * std::sqrt(1155.0 / kPi) *
                   2.0 * x * y * z * (3.0 * z2 - r2) / r5;
        case -1:
            return (1.0 / 16.0) * std::sqrt(165.0 / kPi) * y * zpoly / r5;
        case 0:
            return (1.0 / 16.0) * std::sqrt(11.0 / kPi) *
                   z * (63.0 * z2 * z2 - 70.0 * z2 * r2 + 15.0 * r2 * r2) / r5;
        case 1:
            return (1.0 / 16.0) * std::sqrt(165.0 / kPi) * x * zpoly / r5;
        case 2:
            return (1.0 / 8.0) * std::sqrt(1155.0 / kPi) *
                   (x2 - y2) * z * (3.0 * z2 - r2) / r5;
        case 3:
            return (1.0 / 32.0) * std::sqrt(770.0 / kPi) *
                   x * (x2 - 3.0 * y2) * (9.0 * z2 - r2) / r5;
        case 4:
            return (1.0 / 16.0) * std::sqrt(3465.0 / kPi) *
                   (x2 * x2 - 6.0 * x2 * y2 + y2 * y2) * z / r5;
        default:
            return (1.0 / 32.0) * std::sqrt(1386.0 / kPi) *
                   x * (x2 * x2 - 10.0 * x2 * y2 + 5.0 * y2 * y2) / r5;
    }
}

// psi(x, y, z) = (u(r)/r) Y_lm, u linearly interpolated from the radial
// grid (u(0) = 0 pins the inner segment, so u/r -> u_0/h as r -> 0, the
// correct l = 0 limit; higher l vanish there through Y's r^l factor).
// fill_orbital writes the UN-normalized field in place; synthesize_orbital
// normalizes and returns. The un-normalized form is the single source of truth
// the orbital-free projection (core/projection.hpp) reorganizes -- its radial
// deposit shape must mirror this interpolation exactly.
inline void fill_orbital(Field3D& psi, const Grid3D& g, const RadialGrid& rg,
                         const std::vector<double>& u, int l, int m) {
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
                psi(i, j, k) = Complex<double>{value, 0.0};
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
