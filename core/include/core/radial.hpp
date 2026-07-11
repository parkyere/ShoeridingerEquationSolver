#pragma once

// Radial engine: bound levels and E1 lifetimes for every orbital with
// n <= n_max. Central V(r) reduces the eigenproblem exactly to 1D per l:
//     u'' = 2 (V + l(l+1)/(2 r^2) - E) u,  u(0) = u(R) = 0,  psi = (u/r) Y_lm.
// 3-point FD on r_i = i h -> symmetric tridiagonal H; eigenvalues by Sturm
// bisection, eigenvectors by shifted inverse iteration. The k-th state for a
// given l has k radial nodes and n = l + 1 + k.

#include <core/decay.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

struct RadialGrid {
    double rmax{};
    int n{};  // interior points; r_i = (i+1) h, h = rmax / (n+1)

    double h() const { return rmax / (n + 1); }
    double r(int i) const { return (i + 1) * h(); }
};

// Symmetric tridiagonal radial Hamiltonian for angular momentum l.
struct RadialHamiltonian {
    std::vector<double> diag;
    double off{};  // constant off-diagonal -1/(2 h^2)
};

inline RadialHamiltonian radial_hamiltonian(const RadialGrid& g,
                                            const std::vector<double>& v, int l) {
    const double h = g.h();
    const double inv_h2 = 1.0 / (h * h);
    RadialHamiltonian ham;
    ham.off = -0.5 * inv_h2;
    ham.diag.resize(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double r = g.r(i);
        ham.diag[static_cast<std::size_t>(i)] =
            inv_h2 + v[static_cast<std::size_t>(i)] +
            0.5 * static_cast<double>(l) * (l + 1) / (r * r);
    }
    return ham;
}

// Sturm count: the number of eigenvalues strictly below e, via the sign
// count of the LDL^T pivots q_i = (d_i - e) - off^2 / q_{i-1}.
inline int sturm_count_below(const RadialHamiltonian& ham, double e) {
    const double off2 = ham.off * ham.off;
    int count = 0;
    double q = 1.0;
    for (std::size_t i = 0; i < ham.diag.size(); ++i) {
        q = (ham.diag[i] - e) - (i == 0 ? 0.0 : off2 / q);
        if (q == 0.0) {
            q = -1e-300;  // graze: nudge off the singularity, counted below
        }
        if (q < 0.0) {
            ++count;
        }
    }
    return count;
}

struct RadialState {
    double energy{};
    std::vector<double> u;  // normalized: sum u_i^2 h = 1
};

namespace radial_detail {

// Thomas solve of (T - sigma I) x = b for symmetric tridiagonal T; the
// slight shift keeps the pivots nonzero near an eigenvalue, and inverse
// iteration thrives on the near-singularity.
inline void shifted_thomas_solve(const RadialHamiltonian& ham, double sigma,
                                 std::vector<double>& x) {
    const std::size_t n = ham.diag.size();
    std::vector<double> c(n);   // modified superdiagonal factors
    std::vector<double> d(n);   // modified rhs
    double piv = ham.diag[0] - sigma;
    if (std::abs(piv) < 1e-300) {
        piv = 1e-300;
    }
    c[0] = ham.off / piv;
    d[0] = x[0] / piv;
    for (std::size_t i = 1; i < n; ++i) {
        piv = (ham.diag[i] - sigma) - ham.off * c[i - 1];
        if (std::abs(piv) < 1e-300) {
            piv = 1e-300;
        }
        c[i] = ham.off / piv;
        d[i] = (x[i] - ham.off * d[i - 1]) / piv;
    }
    x[n - 1] = d[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) {
        x[i] = d[i] - c[i] * x[i + 1];
    }
}

inline void normalize_u(std::vector<double>& u, double h) {
    double s = 0.0;
    for (const double x : u) {
        s += x * x;
    }
    const double inv = 1.0 / std::sqrt(s * h);
    for (double& x : u) {
        x *= inv;
    }
}

}  // namespace radial_detail

// The k-th eigenstate (k = 0, 1, ... = number of radial nodes) of the given
// radial Hamiltonian: Sturm bisection to machine width, then a few rounds
// of shifted inverse iteration.
inline RadialState radial_eigenstate(const RadialGrid& g, const RadialHamiltonian& ham,
                                     int k) {
    double lo = ham.diag[0];
    double hi = ham.diag[0];
    for (const double d : ham.diag) {
        lo = std::min(lo, d);
        hi = std::max(hi, d);
    }
    lo -= 2.0 * std::abs(ham.off);
    hi += 2.0 * std::abs(ham.off);
    // Bisect until the interval isolates exactly the k-th eigenvalue.
    for (int it = 0; it < 200 && (hi - lo) > 1e-13 * (1.0 + std::abs(lo)); ++it) {
        const double mid = 0.5 * (lo + hi);
        if (sturm_count_below(ham, mid) <= k) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    RadialState s;
    s.energy = 0.5 * (lo + hi);

    // Inverse iteration with a shift nudged off the eigenvalue.
    const double sigma = s.energy + 1e-9 * (1.0 + std::abs(s.energy));
    s.u.assign(ham.diag.size(), 1.0);
    radial_detail::normalize_u(s.u, g.h());
    for (int it = 0; it < 4; ++it) {
        radial_detail::shifted_thomas_solve(ham, sigma, s.u);
        radial_detail::normalize_u(s.u, g.h());
    }
    // Sign convention: positive near the origin.
    if (s.u[0] < 0.0) {
        for (double& x : s.u) {
            x = -x;
        }
    }
    return s;
}

// Rint = integral u1(r) r u2(r) dr (midpoint rule on the uniform grid).
inline double radial_dipole_integral(const RadialGrid& g, const std::vector<double>& u1,
                                     const std::vector<double>& u2) {
    double s = 0.0;
    for (int i = 0; i < g.n; ++i) {
        s += u1[static_cast<std::size_t>(i)] * g.r(i) * u2[static_cast<std::size_t>(i)];
    }
    return s * g.h();
}

// Level-averaged E1 rate (summed over final m/polarizations, averaged over
// initial m): A = einstein_a(omega, max(l,l')/(2 l_upper + 1) * Rint^2).
inline double einstein_a_level(double omega, int l_upper, int l_lower,
                               double radial_integral) {
    const double lmax = static_cast<double>(std::max(l_upper, l_lower));
    const double factor = lmax / (2.0 * l_upper + 1.0);
    return einstein_a(omega, factor * radial_integral * radial_integral);
}

struct LevelInfo {
    int n{};
    int l{};
    double energy{};
    double lifetime{};  // au; 0 = stable (no open E1 channel)
};

// All bound levels with n <= n_max: energies from the radial eigensolver,
// lifetimes from the full downward E1 rate sums.
inline std::vector<LevelInfo> bound_level_table(const RadialGrid& g,
                                                const std::vector<double>& v,
                                                int n_max) {
    struct Solved {
        int n;
        int l;
        RadialState state;
    };
    std::vector<Solved> solved;
    for (int l = 0; l < n_max; ++l) {
        const RadialHamiltonian ham = radial_hamiltonian(g, v, l);
        for (int k = 0; l + 1 + k <= n_max; ++k) {
            solved.push_back(Solved{l + 1 + k, l, radial_eigenstate(g, ham, k)});
        }
    }
    // Gaps below the FD resolution (h^2 ~ 1e-5..1e-4, e.g. hydrogen 2s/2p)
    // are degenerate physics: their omega^3-suppressed rates (A < 1e-14)
    // would fake a decay path for E1-stable levels.
    constexpr double kDegenerateGap = 1e-4;
    std::vector<LevelInfo> table;
    table.reserve(solved.size());
    for (const Solved& up : solved) {
        double a_sum = 0.0;
        for (const Solved& dn : solved) {
            if (std::abs(up.l - dn.l) != 1 ||
                dn.state.energy >= up.state.energy - kDegenerateGap) {
                continue;
            }
            const double omega = up.state.energy - dn.state.energy;
            const double rint = radial_dipole_integral(g, up.state.u, dn.state.u);
            a_sum += einstein_a_level(omega, up.l, dn.l, rint);
        }
        table.push_back(LevelInfo{up.n, up.l, up.state.energy,
                                  a_sum > 0.0 ? 1.0 / a_sum : 0.0});
    }
    return table;
}

}  // namespace ses
