module;
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
export module ses.potential;
export import ses.grid;
export import ses.vec;


// Potential builders: real-valued V sampled on the grid, fed to the
// split-operator propagator. Atomic units.


export namespace ses {

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

// V(x) = vb ((x/a)^2 - 1)^2: symmetric quartic double well -- minima V = 0
// at +-a, barrier vb at the origin (the tunneling-oscillation scene).
inline std::vector<double> double_well_potential(const Grid1D& g, double vb, double a) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double s = g.coord(i) / a;
        const double q = s * s - 1.0;
        v[static_cast<std::size_t>(i)] = vb * q * q;
    }
    return v;
}

// V(x) = -v0 sech^2((x - x0)/a): Poschl-Teller; reflectionless at v0 = l(l+1)/(2 a^2).
inline std::vector<double> poschl_teller_potential(const Grid1D& g, double v0,
                                                   double a, double x0 = 0.0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double c = std::cosh((g.coord(i) - x0) / a);
        v[static_cast<std::size_t>(i)] = -v0 / (c * c);
    }
    return v;
}

// V(x) = d (1 - e^{-alpha (x - x0)})^2: Morse well.
// E_n = w0 (n + 1/2) - (alpha^2/2)(n + 1/2)^2, w0 = alpha sqrt(2 d).
inline std::vector<double> morse_potential(const Grid1D& g, double d,
                                           double alpha, double x0) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double q = 1.0 - std::exp(-alpha * (g.coord(i) - x0));
        v[static_cast<std::size_t>(i)] = d * q * q;
    }
    return v;
}

// Integral of 1/r over the unit cube centered at the origin (verified to 7
// digits by quadrature); cell average of -Z/r over the nucleus cell is
// -Z * this / h.
inline constexpr double kCoulombCellAverage = 2.3800774;

// Rectangular barrier: V = v0 on the half-open interval [x_lo, x_hi),
// 0 elsewhere (the 1D textbook tunneling potential).
inline std::vector<double> barrier_potential(const Grid1D& g, double v0,
                                             double x_lo, double x_hi) {
    std::vector<double> v(static_cast<std::size_t>(g.n), 0.0);
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        if (x >= x_lo && x < x_hi) {
            v[static_cast<std::size_t>(i)] = v0;
        }
    }
    return v;
}

// Rectangular barrier slab: V = v0 on the half-open x-slab [x_lo, x_hi),
// 0 elsewhere; y/z-independent (the tunneling scenario's potential).
inline std::vector<double> barrier_potential(const Grid3D& g, double v0,
                                             double x_lo, double x_hi) {
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (int i = 0; i < g.x.n; ++i) {
        const double x = g.x.coord(i);
        if (x < x_lo || x >= x_hi) {
            continue;
        }
        for (int k = 0; k < g.z.n; ++k) {
            for (int j = 0; j < g.y.n; ++j) {
                v[static_cast<std::size_t>(g.flat(i, j, k))] = v0;
            }
        }
    }
    return v;
}

// V(r) = -Z / |r - c|, bare Coulomb, with only the singular nucleus cell
// replaced by the analytic cell average -Z * kCoulombCellAverage / h (see
// docs/ARCHITECTURE.md for why not soft-Coulomb). Assumes cubic cells and the
// nucleus on a grid point; an off-point nucleus simply gets -Z/r throughout.
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

// Multi-center regularized bare Coulomb: the superposition of -Z/|r - c_k|
// with each EXACT-hit nucleus cell taking its own analytic average (plus
// the other centers' regular -Z/r there). Fixed nuclei, one electron --
// the Born-Oppenheimer molecular potentials (H2+ and friends). Centers on
// grid points keep every regularized cell honest; an off-point center
// simply contributes bare -Z/r throughout (same convention as the
// single-center builder).
inline std::vector<double> regularized_coulomb_potential(
    const Grid3D& g, double Z, const std::vector<Vec3d>& centers) {
    const double h = g.x.spacing();
    const double center_v = -Z * kCoulombCellAverage / h;
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (const Vec3d& c : centers) {
        for (int k = 0; k < g.z.n; ++k) {
            for (int j = 0; j < g.y.n; ++j) {
                for (int i = 0; i < g.x.n; ++i) {
                    const double dx = g.x.coord(i) - c.x;
                    const double dy = g.y.coord(j) - c.y;
                    const double dz = g.z.coord(k) - c.z;
                    const double r = std::sqrt(dx * dx + dy * dy + dz * dz);
                    v[static_cast<std::size_t>(g.flat(i, j, k))] +=
                        (r < 1e-6 * h) ? center_v : -Z / r;
                }
            }
        }
    }
    return v;
}

// Snap a point to the nearest grid point (per axis, clamped to the valid
// coords -- xmax is not a point on the periodic grid). Molecular centers
// MUST sit on grid points: the bare multi-center builder regularizes only
// exact-hit nucleus cells, and an off-grid center would hand its nearest
// cell an arbitrary -Z/r depth. (Project rule: bare regularized Coulomb
// everywhere; soft Coulomb is avoided.)
inline Vec3d snap_to_grid(const Grid3D& g, Vec3d p) {
    auto axis = [](const Grid1D& ax, double x) {
        const double h = ax.spacing();
        double i = std::round((x - ax.xmin) / h);
        i = std::min(std::max(i, 0.0), static_cast<double>(ax.n - 1));
        return ax.xmin + i * h;
    };
    return {axis(g.x, p.x), axis(g.y, p.y), axis(g.z, p.z)};
}

// Absorbing mask for the periodic box: exactly 1 in the interior, cos^2
// ramp to 0 within `width` of each wall. psi *= mask each real-time step
// damps outgoing flux instead of periodic wrap-around; never applied during
// imaginary-time state prep.
inline std::vector<double> absorbing_mask(const Grid1D& g, double width) {
    std::vector<double> m(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        const double d_lo = x - g.xmin;
        const double d_hi = g.xmax - x;
        const double d = d_lo < d_hi ? d_lo : d_hi;  // distance to nearest wall
        if (d >= width) {
            m[static_cast<std::size_t>(i)] = 1.0;
            continue;
        }
        const double t = d / width;  // 0 at the wall, 1 at the layer inner edge
        const double s = std::sin(0.5 * 3.14159265358979323846 * t);
        m[static_cast<std::size_t>(i)] = s * s;  // smooth cos^2 ramp
    }
    return m;
}

// 3D mask: the cell value is the x*y*z product of the per-axis 1D ramps
// (3n sin calls instead of 3n^3; identical factor formula as above).
inline std::vector<double> absorbing_mask(const Grid3D& g, double width) {
    const std::vector<double> mx = absorbing_mask(g.x, width);
    const std::vector<double> my = absorbing_mask(g.y, width);
    const std::vector<double> mz = absorbing_mask(g.z, width);
    std::vector<double> m(static_cast<std::size_t>(g.size()));
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                m[static_cast<std::size_t>(g.flat(i, j, k))] =
                    mx[static_cast<std::size_t>(i)] *
                    my[static_cast<std::size_t>(j)] *
                    mz[static_cast<std::size_t>(k)];
            }
        }
    }
    return m;
}

}  // namespace ses
