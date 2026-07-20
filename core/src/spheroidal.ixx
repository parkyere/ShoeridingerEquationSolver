module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.spheroidal;
export import ses.grid;
export import ses.field;
export import ses.vec;


// Exact prolate-spheroidal separation of the H2+ electronic Schrodinger
// equation (atomic units, Z=1 each, foci AT the nuclei, internuclear R).
// psi = Lambda(xi) M(eta) e^{i m phi}, xi in [1,inf), eta in [-1,1].
// Energy param p^2 = -E_elec R^2/2 (E_elec = -2p^2/R^2, ELECTRONIC, no 1/R).
// Separated ODEs (Madsen-Peek; signs verified vs Scott arXiv:physics/0607081,
// Turbiner arXiv:1401.8009):
//   angular: d/deta[(1-eta^2)M'] + [ -A + p^2 eta^2 - m^2/(1-eta^2) ] M = 0
//   radial : d/dxi [(xi^2-1)L'] + [  A + 2R xi - p^2 xi^2 - m^2/(xi^2-1) ] L = 0
// Method: cell-centered FD -> symmetric tridiagonals (variable off-diagonals);
// angular gives A_{n_eta}(p^2), radial (weight xi^2, symmetrized) gives
// p^2_rad(A) for branch n_xi; couple by the 1D root g(p^2)=p^2_rad-p^2.
// Same reduce-to-1D-and-synthesize spirit as the hydrogen radial atlas.


export namespace ses {

struct H2plusOrbital {
    int m = 0;         // |m|: 0 sigma, 1 pi, 2 delta
    int n_eta = 0;     // interior nodes of M(eta)
    int n_xi = 0;      // interior nodes of Lambda(xi)
    int parity = 1;    // (-1)^(n_eta+m): +1 gerade, -1 ungerade
    double energy = 0.0;  // E_elec (Ha), no nuclear 1/R
    double p = 0.0;       // sqrt(-E_elec R^2/2)
    double R = 0.0;
    std::vector<double> eta;     // cell centers in [-1,1]
    std::vector<double> M;       // M(eta), grid-normalized sign(max)>0
    std::vector<double> xi;      // cell centers in [1, xi_max]
    std::vector<double> lambda;  // Lambda(xi)
};

namespace spheroidal_detail {

constexpr double kPi = 3.14159265358979323846;
constexpr int kNeta = 400;       // angular grid
constexpr int kNxi = 1200;       // radial grid (covers the wide box + high MOs)
constexpr double kXiReach = 40.0;  // radial reach in bohr (xi_max = 1 + 2*reach/R)

// Symmetric tridiagonal (diag d, off b[i] = T_{i,i+1}). Sturm count: number
// of eigenvalues strictly below e (LDL^T pivot sign count, variable off).
inline int sturm_below(const std::vector<double>& d,
                       const std::vector<double>& b, double e) noexcept {
    int count = 0;
    double q = 1.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        const double off2 = i == 0 ? 0.0 : b[i - 1] * b[i - 1];
        q = (d[i] - e) - (i == 0 ? 0.0 : off2 / q);
        if (q == 0.0) {
            q = -1e-300;
        }
        if (q < 0.0) {
            ++count;
        }
    }
    return count;
}

// The k-th eigenvalue from the TOP (k=0 largest), by Sturm bisection.
inline double eig_from_top(const std::vector<double>& d,
                           const std::vector<double>& b, int k) {
    const int n = static_cast<int>(d.size());
    double lo = d[0];
    double hi = d[0];
    for (std::size_t i = 0; i < d.size(); ++i) {
        double radius = 0.0;
        if (i > 0) {
            radius += std::abs(b[i - 1]);
        }
        if (i + 1 < d.size()) {
            radius += std::abs(b[i]);
        }
        lo = std::min(lo, d[i] - radius);
        hi = std::max(hi, d[i] + radius);
    }
    // The k-th from top = the (n-1-k)-th from bottom: sturm_below(mid) counts
    // eigenvalues below mid; we want the largest e with count <= n-1-k.
    const int target = n - 1 - k;
    for (int it = 0; it < 200 && (hi - lo) > 1e-12 * (1.0 + std::abs(hi));
         ++it) {
        const double mid = 0.5 * (lo + hi);
        if (sturm_below(d, b, mid) <= target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

// Eigenvector for eigenvalue e by shifted inverse iteration (Thomas solve of
// the near-singular (T - e) system, a few rounds).
inline std::vector<double> eigvec(const std::vector<double>& d,
                                  const std::vector<double>& b, double e) {
    const std::size_t n = d.size();
    std::vector<double> x(n, 1.0);
    const double sigma = e + 1e-8 * (1.0 + std::abs(e));
    std::vector<double> c(n), rhs(n);
    for (int iter = 0; iter < 3; ++iter) {
        // Thomas solve (T - sigma) y = x.
        double piv = d[0] - sigma;
        if (std::abs(piv) < 1e-300) {
            piv = 1e-300;
        }
        c[0] = (n > 1 ? b[0] : 0.0) / piv;
        rhs[0] = x[0] / piv;
        for (std::size_t i = 1; i < n; ++i) {
            piv = (d[i] - sigma) - b[i - 1] * c[i - 1];
            if (std::abs(piv) < 1e-300) {
                piv = 1e-300;
            }
            c[i] = (i + 1 < n ? b[i] : 0.0) / piv;
            rhs[i] = (x[i] - b[i - 1] * rhs[i - 1]) / piv;
        }
        x[n - 1] = rhs[n - 1];
        for (std::size_t i = n - 1; i-- > 0;) {
            x[i] = rhs[i] - c[i] * x[i + 1];
        }
        double s = 0.0;
        for (double v : x) {
            s += v * v;
        }
        const double inv = 1.0 / std::sqrt(s);
        for (double& v : x) {
            v *= inv;
        }
    }
    return x;
}

// Angular operator diag/off for a trial p^2 (eigenvalue = A, Madsen-Peek).
inline void angular_operator(double p2, int m, std::vector<double>& eta,
                             std::vector<double>& d, std::vector<double>& b) {
    const double deta = 2.0 / kNeta;
    eta.resize(kNeta);
    d.resize(kNeta);
    b.assign(kNeta - 1, 0.0);
    for (int i = 0; i < kNeta; ++i) {
        eta[i] = -1.0 + (i + 0.5) * deta;
    }
    const double inv2 = 1.0 / (deta * deta);
    for (int i = 0; i < kNeta; ++i) {
        const double etl = -1.0 + i * deta;         // face below
        const double eth = -1.0 + (i + 1) * deta;   // face above
        const double cl = (1.0 - etl * etl) * inv2;  // 0 at eta=-1
        const double ch = (1.0 - eth * eth) * inv2;  // 0 at eta=+1
        const double e2 = eta[i] * eta[i];
        d[i] = -(cl + ch) + p2 * e2 - m * m / (1.0 - e2);
        if (i + 1 < kNeta) {
            b[i] = ch;
        }
    }
}

// A_{n_eta}(p^2): the (n_eta)-th eigenvalue from the top of the angular op.
inline double angular_A(double p2, int m, int n_eta) {
    std::vector<double> eta, d, b;
    angular_operator(p2, m, eta, d, b);
    return eig_from_top(d, b, n_eta);
}

// Radial generalized eigenproblem L Lambda = p^2 (xi^2) Lambda for fixed A;
// symmetrized by S=diag(xi). Returns the (n_xi)-th p^2 from the top and
// (optionally) the eigenvector Lambda.
inline double radial_p2(double A, int m, double R, int n_xi, double xi_max,
                        std::vector<double>* xi_out,
                        std::vector<double>* lam_out) {
    const double dxi = (xi_max - 1.0) / kNxi;
    const double inv2 = 1.0 / (dxi * dxi);
    std::vector<double> xi(kNxi), s(kNxi);
    std::vector<double> ld(kNxi), lb(kNxi - 1, 0.0);  // L diag/off
    for (int i = 0; i < kNxi; ++i) {
        xi[i] = 1.0 + (i + 0.5) * dxi;
        s[i] = xi[i];  // sqrt(weight xi^2)
    }
    for (int i = 0; i < kNxi; ++i) {
        const double xl = 1.0 + i * dxi;          // face below (xi=1 -> 0)
        const double xh = 1.0 + (i + 1) * dxi;    // face above
        const double cl = (xl * xl - 1.0) * inv2;
        const double ch = (xh * xh - 1.0) * inv2;  // top face = Dirichlet
        const double x = xi[i];
        ld[i] = -(cl + ch) + A + 2.0 * R * x - m * m / (x * x - 1.0);
        if (i + 1 < kNxi) {
            lb[i] = ch;
        }
    }
    // Symmetrized M = S^{-1} L S^{-1}: eigenvalues are p^2.
    std::vector<double> md(kNxi), mb(kNxi - 1);
    for (int i = 0; i < kNxi; ++i) {
        md[i] = ld[i] / (s[i] * s[i]);
        if (i + 1 < kNxi) {
            mb[i] = lb[i] / (s[i] * s[i + 1]);
        }
    }
    const double p2 = eig_from_top(md, mb, n_xi);
    if (xi_out != nullptr && lam_out != nullptr) {
        *xi_out = xi;
        const std::vector<double> y = eigvec(md, mb, p2);
        lam_out->resize(kNxi);
        for (int i = 0; i < kNxi; ++i) {
            (*lam_out)[i] = y[i] / s[i];  // undo the symmetrizing scale
        }
    }
    return p2;
}

}  // namespace spheroidal_detail

// Solve one H2+ orbital (m>=0, n_eta, n_xi node counts) at internuclear R.
inline H2plusOrbital h2plus_orbital(double R, int m, int n_eta, int n_xi) {
    using namespace spheroidal_detail;
    m = std::abs(m);
    // xi_max: physical reach ~ (R/2)(xi_max-1) = kXiReach bohr, covering the
    // display box so higher (diffuse) orbitals are represented, not truncated.
    const double xi_max = 1.0 + 2.0 * kXiReach / R;

    // Couple E (via p^2) and A: root of g(p^2) = radial_p2(A(p^2)) - p^2.
    auto g = [&](double p2) {
        const double A = angular_A(p2, m, n_eta);
        return radial_p2(A, m, R, n_xi, xi_max, nullptr, nullptr) - p2;
    };

    // Bracket: scan p^2 upward for a sign change of g (g>0 at small p^2 since
    // the radial well is deep; g<0 once p^2 overshoots).
    double lo = 1e-4;
    double glo = g(lo);
    double hi = lo;
    double ghi = glo;
    const double p2_ceiling = R * R + 8.0;  // bound branch crosses below this
    bool bracketed = false;
    for (double p2 = 0.05; p2 <= p2_ceiling; p2 += 0.05) {
        const double gp = g(p2);
        if (glo * gp <= 0.0) {
            hi = p2;
            ghi = gp;
            bracketed = true;
            break;
        }
        lo = p2;
        glo = gp;
    }
    double p2 = 0.5 * (lo + hi);
    if (bracketed) {
        for (int it = 0; it < 100 && (hi - lo) > 1e-11; ++it) {
            p2 = 0.5 * (lo + hi);
            const double gm = g(p2);
            if (glo * gm <= 0.0) {
                hi = p2;
                (void)ghi;
            } else {
                lo = p2;
                glo = gm;
            }
        }
        p2 = 0.5 * (lo + hi);
    }

    H2plusOrbital o;
    o.m = m;
    o.n_eta = n_eta;
    o.n_xi = n_xi;
    o.parity = ((n_eta + m) % 2 == 0) ? +1 : -1;
    o.p = std::sqrt(std::max(0.0, p2));
    o.energy = -2.0 * p2 / (R * R);
    o.R = R;

    // Capture the two profiles at the converged (p^2, A).
    const double A = angular_A(p2, m, n_eta);
    std::vector<double> ad, ab;
    angular_operator(p2, m, o.eta, ad, ab);
    o.M = eigvec(ad, ab, A);
    radial_p2(A, m, R, n_xi, xi_max, &o.xi, &o.lambda);

    // Sign convention: make the bulk peak positive for reproducible display.
    auto fix_sign = [](std::vector<double>& f) {
        double mx = 0.0;
        double val = 0.0;
        for (double v : f) {
            if (std::abs(v) > mx) {
                mx = std::abs(v);
                val = v;
            }
        }
        if (val < 0.0) {
            for (double& v : f) {
                v = -v;
            }
        }
    };
    fix_sign(o.M);
    fix_sign(o.lambda);
    return o;
}

// Atlas: the lowest bound orbitals (E_elec < 0), sorted by energy, up to
// max_states. Sweeps small (m, n_eta, n_xi) and keeps the bound ones.
inline std::vector<H2plusOrbital> h2plus_atlas(double R, int max_states) {
    std::vector<H2plusOrbital> out;
    for (int shell = 0; shell <= 6 && static_cast<int>(out.size()) <
                                          max_states + 12; ++shell) {
        for (int m = 0; m <= shell; ++m) {
            for (int n_eta = 0; n_eta + m <= shell; ++n_eta) {
                const int n_xi = shell - m - n_eta;
                if (n_xi < 0) {
                    continue;
                }
                H2plusOrbital o = h2plus_orbital(R, m, n_eta, n_xi);
                if (o.energy < -1e-3) {  // bound
                    out.push_back(std::move(o));
                }
            }
        }
    }
    std::sort(out.begin(), out.end(),
              [](const H2plusOrbital& a, const H2plusOrbital& b) {
                  return a.energy < b.energy;
              });
    if (static_cast<int>(out.size()) > max_states) {
        out.resize(static_cast<std::size_t>(max_states));
    }
    return out;
}

// Sample a real orbital onto a Cartesian grid. Molecular axis = x, nuclei at
// +-R/2. partner: 0 = cos(m phi) (y-lobed), 1 = sin(m phi) (z-lobed); m=0
// ignores it. Grid-normalized.
inline Field3D synthesize_h2plus(const Grid3D& g, const H2plusOrbital& o,
                                 int partner) {
    Field3D f{g};
    const double R = o.R;
    const double half = 0.5 * R;
    // `high_zero`: eta (compact [-1,1]) clamps to the edge cell value at
    // both ends; xi (semi-infinite) clamps to the front but decays to 0
    // beyond xi_max.
    auto interp = [](const std::vector<double>& xs,
                     const std::vector<double>& ys, double x, bool high_zero) {
        const int n = static_cast<int>(xs.size());
        if (n == 0) {
            return 0.0;
        }
        if (x <= xs.front()) {
            return ys.front();
        }
        if (x >= xs.back()) {
            return high_zero ? 0.0 : ys.back();
        }
        const double dx = xs[1] - xs[0];
        const double t = (x - xs[0]) / dx;
        const int i = std::clamp(static_cast<int>(t), 0, n - 2);
        const double frac = t - i;
        return ys[static_cast<std::size_t>(i)] * (1.0 - frac) +
               ys[static_cast<std::size_t>(i + 1)] * frac;
    };
    for (int k = 0; k < g.z.n; ++k) {
        const double z = g.z.coord(k);
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double rA =
                    std::sqrt((x + half) * (x + half) + y * y + z * z);
                const double rB =
                    std::sqrt((x - half) * (x - half) + y * y + z * z);
                double xi = (rA + rB) / R;
                double eta = (rA - rB) / R;
                xi = std::max(1.0, xi);
                eta = std::clamp(eta, -1.0, 1.0);
                double ang = 1.0;
                if (o.m > 0) {
                    const double rho = std::sqrt(y * y + z * z);
                    const double phi = std::atan2(z, y);
                    (void)rho;
                    ang = partner == 0 ? std::cos(o.m * phi)
                                       : std::sin(o.m * phi);
                }
                const double val = interp(o.xi, o.lambda, xi, true) *
                                   interp(o.eta, o.M, eta, false) * ang;
                f(i, j, k) = std::complex<double>{val, 0.0};
            }
        }
    }
    const double nrm = norm_sq(f);
    if (nrm > 0.0) {
        const double inv = 1.0 / std::sqrt(nrm);
        for (auto& c : f.data()) {
            c *= inv;
        }
    }
    return f;
}

}  // namespace ses
