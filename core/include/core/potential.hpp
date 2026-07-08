#pragma once

// Potential builders: real-valued V sampled on the grid, fed to the
// split-operator propagator.
//
// The soft-Coulomb regularization -Z/sqrt(dx^2 + a^2) is the standard 1D
// stand-in for the singular -Z/|x|: finite everywhere (deepest value -Z/a at
// the nucleus), so a grid point may sit exactly on the nucleus.

#include <core/grid.hpp>
#include <core/vec.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

// V(x) = 1/2 omega^2 (x - x0)^2
inline std::vector<double> harmonic_potential(const Grid1D& g, double omega, double x0 = 0.0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double dx = g.coord(i) - x0;
        v[static_cast<std::size_t>(i)] = 0.5 * omega * omega * dx * dx;
    }
    return v;
}

// V(x) = -Z / sqrt((x - x0)^2 + a^2)
inline std::vector<double> soft_coulomb_potential(const Grid1D& g, double Z, double a,
                                                  double x0 = 0.0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double dx = g.coord(i) - x0;
        v[static_cast<std::size_t>(i)] = -Z / std::sqrt(dx * dx + a * a);
    }
    return v;
}

// V(r) = 1/2 omega^2 |r - c|^2  (isotropic, flat x-fastest storage)
inline std::vector<double> harmonic_potential(const Grid3D& g, double omega, Vec3d c) {
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double dx = g.x.coord(i) - c.x;
                const double dy = g.y.coord(j) - c.y;
                const double dz = g.z.coord(k) - c.z;
                v[static_cast<std::size_t>(g.flat(i, j, k))] =
                    0.5 * omega * omega * (dx * dx + dy * dy + dz * dz);
            }
        }
    }
    return v;
}

// V(r) = -Z / sqrt(|r - c|^2 + a^2)
inline std::vector<double> soft_coulomb_potential(const Grid3D& g, double Z, double a, Vec3d c) {
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double dx = g.x.coord(i) - c.x;
                const double dy = g.y.coord(j) - c.y;
                const double dz = g.z.coord(k) - c.z;
                v[static_cast<std::size_t>(g.flat(i, j, k))] =
                    -Z / std::sqrt(dx * dx + dy * dy + dz * dz + a * a);
            }
        }
    }
    return v;
}

// Integral of 1/r over the UNIT cube centered at the origin (a pure geometric
// constant, verified to 7 digits by cube quadrature). The average of -Z/r over
// a cubic cell of side h centered on the nucleus is therefore -Z * this / h --
// finite, and the correct discrete stand-in for the singular nucleus cell.
inline constexpr double kCoulombCellAverage = 2.3800774;

// V(r) = -Z / |r - c|, the BARE Coulomb potential, with the one grid cell that
// coincides with the nucleus c (where -Z/r would be -infinity) replaced by the
// analytic cell average -Z * kCoulombCellAverage / h. Unlike soft_coulomb (which
// rounds off the WHOLE well and shifts every level up), this regularizes ONLY the
// singular cell and keeps every other cell exact, so the spectrum converges to
// textbook hydrogen as h -> 0. Assumes cubic cells (h = g.x.spacing()) and the
// nucleus on a grid point (the app's convention); off-point nuclei hit no exact
// singularity and simply get -Z/r throughout.
inline std::vector<double> regularized_coulomb_potential(const Grid3D& g, double Z, Vec3d c) {
    const double h = g.x.spacing();
    const double center = -Z * kCoulombCellAverage / h;
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double dx = g.x.coord(i) - c.x;
                const double dy = g.y.coord(j) - c.y;
                const double dz = g.z.coord(k) - c.z;
                const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                v[static_cast<std::size_t>(g.flat(i, j, k))] =
                    (r < 1e-6 * h) ? center : -Z / r;
            }
        }
    }
    return v;
}

// Boundary absorbing mask for the FFT (periodic) box: 1.0 in the interior,
// tapering smoothly to 0 within `width` of each wall (per-axis cos^2 ramps
// multiplied -- a rounded-box absorber). Multiplying psi by this each real-time
// step damps outgoing (ionized) flux at the edges instead of letting the
// periodic grid wrap it around to the opposite wall. The interior is exactly 1,
// so the bound atom is untouched (and it is applied only in real time, never
// during imaginary-time state prep).
inline std::vector<double> absorbing_mask(const Grid3D& g, double width) {
    auto axis = [width](const Grid1D& ax, int i) {
        const double x = ax.coord(i);
        const double d_lo = x - ax.xmin;
        const double d_hi = ax.xmax - x;
        const double d = d_lo < d_hi ? d_lo : d_hi;  // distance to nearest wall
        if (d >= width) {
            return 1.0;
        }
        const double t = d / width;  // 0 at the wall, 1 at the layer inner edge
        const double s = std::sin(0.5 * 3.14159265358979323846 * t);
        return s * s;  // smooth cos^2 ramp
    };
    std::vector<double> m(static_cast<std::size_t>(g.size()));
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                m[static_cast<std::size_t>(g.flat(i, j, k))] =
                    axis(g.x, i) * axis(g.y, j) * axis(g.z, k);
            }
        }
    }
    return m;
}

}  // namespace ses
