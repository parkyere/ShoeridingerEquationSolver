module;
#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>
export module ses.ladder;
export import ses.field;
export import ses.grid;
import ses.fft;
import ses.spectral;


// 1D harmonic-oscillator ladder operators (atomic units, m = hbar = 1):
//
//     a    = sqrt(omega/2) x + 1/sqrt(2 omega) d/dx
//     adag = sqrt(omega/2) x - 1/sqrt(2 omega) d/dx
//
// acting on eigenstates: adag|n> = sqrt(n+1)|n+1>, a|n> = sqrt(n)|n-1>,
// a|0> = 0. The derivative is spectral (FFT ik multiply), consistent with
// the split-operator periodic grid, so repeated applications climb/descend
// the Fock chain cleanly to near machine epsilon.
//
// Both entry points return ||O psi||^2 BEFORE the internal renormalization
// (the sqrt(n+1) / sqrt(n) counting weight squared) and leave psi normalized.
// ladder_lower is guarded: ||a psi||^2 = <N> can vanish (a|0> = 0), in which
// case psi is left UNCHANGED and the tiny return value is the caller's
// forbidden-transition signal. ladder_raise needs no guard:
// ||adag psi||^2 = <N> + 1 >= 1 on any normalized state.


export namespace ses {

namespace ladder_detail {

// (sqrt(omega/2) x + deriv_sign / sqrt(2 omega) d/dx) psi, spectral d/dx.
inline std::vector<std::complex<double>> apply(const Field1D& psi, double omega,
                                               double deriv_sign) {
    const Grid1D& g = psi.grid();
    std::vector<std::complex<double>> dpsi = psi.data();
    fft(dpsi);
    const std::vector<double> k = wavenumbers(g);
    for (std::size_t j = 0; j < dpsi.size(); ++j) {
        dpsi[j] *= std::complex<double>{0.0, k[j]};
    }
    ifft(dpsi);

    const double cx = std::sqrt(0.5 * omega);
    const double cd = deriv_sign / std::sqrt(2.0 * omega);
    std::vector<std::complex<double>> out(dpsi.size());
    for (int i = 0; i < psi.size(); ++i) {
        const std::size_t s = static_cast<std::size_t>(i);
        out[s] = cx * g.coord(i) * psi[i] + cd * dpsi[s];
    }
    return out;
}

// ||v||^2 with the grid weight h (the Field1D norm convention).
inline double norm_sq_h(const std::vector<std::complex<double>>& v, double h) noexcept {
    double acc = 0.0;
    for (const std::complex<double>& z : v) {
        acc += std::norm(z);
    }
    return acc * h;
}

// Normalize `v` into psi's storage.
inline void store_normalized(Field1D& psi, const std::vector<std::complex<double>>& v,
                             double norm2) noexcept {
    const double inv = 1.0 / std::sqrt(norm2);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = inv * v[static_cast<std::size_t>(i)];
    }
}

}  // namespace ladder_detail

// adag: psi <- adag psi / ||adag psi||; returns ||adag psi||^2 (n+1 on |n>).
inline double ladder_raise(Field1D& psi, double omega) {
    const std::vector<std::complex<double>> out =
        ladder_detail::apply(psi, omega, -1.0);
    const double norm2 = ladder_detail::norm_sq_h(out, psi.grid().spacing());
    ladder_detail::store_normalized(psi, out, norm2);
    return norm2;
}

// The exact HO eigenstate |n> built DIRECTLY from the normalized
// Hermite-Gauss recurrence in x-space (atomic units, m = hbar = 1):
//     psi_0 = (omega/pi)^{1/4} exp(-omega x^2 / 2)
//     psi_1 = sqrt(2 omega) x psi_0
//     psi_{k} = sqrt(2 omega / k) x psi_{k-1} - sqrt((k-1)/k) psi_{k-2}
// The NORMALIZED recurrence keeps every intermediate O(1) (no 2^n n!
// overflow) and -- crucially -- uses NO derivative, so unlike the ladder
// chain it suffers no spectral round-off amplification: it is exact to
// round-off for every level the grid can represent. A final discrete
// normalize absorbs the sampling error. This is the ground-truth oracle
// the ladder is measured against, and lets the scene jump to any level.
inline Field1D ho_eigenstate(const Grid1D& g, double omega, int n) {
    const double pi = 3.14159265358979323846;
    Field1D prev{g};   // psi_{k-2}
    Field1D cur{g};    // psi_{k-1}
    const double a0 = std::pow(omega / pi, 0.25);
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        cur[i] = a0 * std::exp(-0.5 * omega * x * x);  // psi_0
    }
    for (int k = 1; k <= n; ++k) {
        const double c1 = std::sqrt(2.0 * omega / k);
        const double c2 = std::sqrt(static_cast<double>(k - 1) / k);
        Field1D next{g};
        for (int i = 0; i < g.n; ++i) {
            next[i] = c1 * g.coord(i) * cur[i] - c2 * prev[i];
        }
        prev = std::move(cur);
        cur = std::move(next);
    }
    normalize(cur);
    return cur;
}

// Noise-free ladder rung for a state KNOWN to be the eigenstate |n_from>
// (up to a global phase; the caller's Var(H) classifier is the gate). The
// raw spectral operator is still what ACTS: it supplies the counting
// norm^2 (n+1 / n) and the global phase of the result. Only the state
// body is then rebuilt from the direct Hermite oracle carrying that phase
// -- the same mathematical object (adag|n> = sqrt(n+1)|n+1>) computed by
// the stable route, so the round-off floor RESETS at every rung instead
// of compounding. Descending is the payoff: the raw chain's noise gains
// (derivative k_max/sqrt(2w), x-term x_max*sqrt(w/2)) amplify residue
// FASTER on the way down (signal shrinks as sqrt(n)), which showed up as
// visible high-k garbage; stable rungs kill it, and the usable range
// becomes the grid's representability ceiling (ho_level_cap), not the
// raw-chain noise cap. Down at the ground still refuses via the operator
// itself (returns ~0, psi untouched). If psi is NOT the claimed
// eigenstate (oracle overlap < 1/2), the raw result is kept as-is: the
// caller misclassified, and the honest operator output stands.
// The REPRESENTABILITY ceiling: the largest level whose direct Hermite
// oracle is still faithful on the grid (discrete energy within 0.1% of
// (n+1/2)w). Box-limited for a soft well (wide turning points), Nyquist-
// band-limited for a stiff one; peaks near omega = k_max/x_max where the
// two meet. This is what caps the STABLE rungs -- far above ladder_cap.
inline int ho_level_cap(const Grid1D& g, double omega) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        v[static_cast<std::size_t>(i)] = 0.5 * omega * omega * x * x;
    }
    const std::vector<double> k = wavenumbers(g);
    // The Hermite chain kept ACROSS levels (ho_eigenstate's recurrence,
    // advanced one rung per iteration): O(total grid work) instead of the
    // O(levels^2) of rebuilding each level from scratch -- this runs on
    // every omega-slider notch. Energy checks are scale-invariant, so the
    // missing per-level discrete renormalization changes nothing.
    const double pi = 3.14159265358979323846;
    Field1D prev{g};
    Field1D cur{g};
    const double a0 = std::pow(omega / pi, 0.25);
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        cur[i] = a0 * std::exp(-0.5 * omega * x * x);
    }
    int cap = 0;
    for (int n = 1; n <= 400; ++n) {
        const double c1 = std::sqrt(2.0 * omega / n);
        const double c2 = std::sqrt(static_cast<double>(n - 1) / n);
        Field1D next{g};
        for (int i = 0; i < g.n; ++i) {
            next[i] = c1 * g.coord(i) * cur[i] - c2 * prev[i];
        }
        prev = std::move(cur);
        cur = std::move(next);
        // <H> spectrally (T in k-space) + V in real space, scale-invariant.
        std::vector<std::complex<double>> phi = cur.data();
        fft(phi);
        double num_t = 0.0;
        double den_k = 0.0;
        for (std::size_t j = 0; j < phi.size(); ++j) {
            const double w = std::norm(phi[j]);
            num_t += 0.5 * k[j] * k[j] * w;
            den_k += w;
        }
        double num_v = 0.0;
        double den_x = 0.0;
        for (int i = 0; i < g.n; ++i) {
            const double w = std::norm(cur[i]);
            num_v += v[static_cast<std::size_t>(i)] * w;
            den_x += w;
        }
        const double e = num_t / den_k + num_v / den_x;
        const double e_exact = (n + 0.5) * omega;
        if (std::abs(e - e_exact) > 1e-3 * e_exact) {
            break;
        }
        cap = n;
    }
    return cap;
}

// Ladder step computed in the truncated Fock basis |0..n_top> -- the
// superposition counterpart of ladder_rung_stable. Project c_n = <n|psi>,
// act EXACTLY on the coefficients (adag: c'_{n+1} = sqrt(n+1) c_n;
// a: c'_n = sqrt(n+1) c_{n+1}), resynthesize from the noise-free oracles.
// The same linear operator, computed in the basis where it is trivial --
// no spectral derivative, so it works at ANY grid k_max (the raw chain's
// noise gain grows with k_max and dies on fine grids). *out_residual gets
// the input's outside-band weight; psi is written only when the band holds
// the state (residual <= 1/2) AND the result does not vanish (annihilation
// on lowering the ground). Returns the counting norm^2 inside the band
// (== <N>+1 / <N> for in-band states).
inline double ladder_fock(Field1D& psi, double omega, bool up, int n_top,
                          double* out_residual = nullptr,
                          double vanish_eps = 1e-6) {
    const Grid1D& g = psi.grid();
    const double h = g.spacing();
    const int nb = (up ? n_top + 1 : n_top) + 1;  // up shifts into n_top+1
    std::vector<Field1D> basis;
    basis.reserve(static_cast<std::size_t>(nb));
    for (int n = 0; n < nb; ++n) {
        basis.push_back(ho_eigenstate(g, omega, n));
    }
    std::vector<std::complex<double>> c(static_cast<std::size_t>(n_top + 1));
    double inside = 0.0;
    for (int n = 0; n <= n_top; ++n) {
        std::complex<double> acc{};
        for (int i = 0; i < g.n; ++i) {
            acc += std::conj(basis[static_cast<std::size_t>(n)][i]) * psi[i];
        }
        c[static_cast<std::size_t>(n)] = acc * h;
        inside += std::norm(c[static_cast<std::size_t>(n)]);
    }
    const double residual = std::max(0.0, 1.0 - inside / norm_sq(psi));
    if (out_residual != nullptr) {
        *out_residual = residual;
    }
    std::vector<std::complex<double>> d(static_cast<std::size_t>(nb));
    double norm2 = 0.0;
    if (up) {
        for (int n = 0; n <= n_top; ++n) {
            d[static_cast<std::size_t>(n + 1)] =
                std::sqrt(n + 1.0) * c[static_cast<std::size_t>(n)];
            norm2 += (n + 1.0) * std::norm(c[static_cast<std::size_t>(n)]);
        }
    } else {
        for (int n = 0; n + 1 <= n_top; ++n) {
            d[static_cast<std::size_t>(n)] =
                std::sqrt(n + 1.0) * c[static_cast<std::size_t>(n + 1)];
            norm2 += (n + 1.0) * std::norm(c[static_cast<std::size_t>(n + 1)]);
        }
    }
    if (residual > 0.5 || norm2 < vanish_eps) {
        return norm2;  // outside the band, or annihilated: psi untouched
    }
    const double inv = 1.0 / std::sqrt(norm2);
    for (int i = 0; i < g.n; ++i) {
        std::complex<double> acc{};
        for (int n = 0; n < nb; ++n) {
            acc += d[static_cast<std::size_t>(n)] *
                   basis[static_cast<std::size_t>(n)][i];
        }
        psi[i] = inv * acc;
    }
    return norm2;
}

// The largest Fock level the FFT ladder reaches cleanly on a given grid --
// MEASURED, not modeled. a-dag carries two competing round-off gains: the
// derivative term k_max/sqrt(2w) (worse for a soft, wide well) and the x
// term x_max*sqrt(w/2) (worse for a stiff well whose tail round-off is
// leveraged by the large |x| box edge), plus a boundary/periodicity effect
// when a wide state does not fit the box. The net clean cap is non-monotone
// (it peaks near w = k_max/x_max) and no simple closed form captures all
// three mechanisms -- so we probe it directly: raise from the ground and
// return the last level still matching the direct Hermite oracle to within
// `defect_tol` (fidelity defect ~ (amplitude noise)^2; 1e-6 ~ 0.1% amplitude,
// below display relevance). ~cap FFTs, run only when omega changes.
inline int ladder_cap(const Grid1D& g, double omega, double defect_tol = 1e-6) {
    Field1D psi = ho_eigenstate(g, omega, 0);
    int cap = 0;
    for (int n = 1; n <= 64; ++n) {
        ladder_raise(psi, omega);
        const Field1D oracle = ho_eigenstate(g, omega, n);
        std::complex<double> ov{};
        for (int i = 0; i < g.n; ++i) {
            ov += std::conj(psi[i]) * oracle[i];
        }
        ov *= g.spacing();  // grid-weighted overlap (both normalized)
        if (1.0 - std::norm(ov) > defect_tol) {
            break;  // ladder diverged from the clean eigenstate here
        }
        cap = n;
    }
    return cap;
}

// a: psi <- a psi / ||a psi|| unless annihilated (||a psi||^2 < vanish_eps,
// e.g. the ground state), in which case psi is untouched; returns ||a psi||^2
// (n on |n>, <N> in general) either way.
inline double ladder_lower(Field1D& psi, double omega, double vanish_eps = 1e-6) {
    const std::vector<std::complex<double>> out =
        ladder_detail::apply(psi, omega, +1.0);
    const double norm2 = ladder_detail::norm_sq_h(out, psi.grid().spacing());
    if (norm2 < vanish_eps) {
        return norm2;
    }
    ladder_detail::store_normalized(psi, out, norm2);
    return norm2;
}

// Noise-free ladder rung for a state KNOWN to be the eigenstate |n_from>
// (up to a global phase; the caller's Var(H) classifier is the gate). The
// raw spectral operator is still what ACTS: it supplies the counting
// norm^2 (n+1 / n) and the global phase of the result. Only the state
// body is then rebuilt from the direct Hermite oracle carrying that phase
// -- the same mathematical object (adag|n> = sqrt(n+1)|n+1>) computed by
// the stable route, so the round-off floor RESETS at every rung instead
// of compounding. Descending is the payoff: the raw chain's noise gains
// (derivative k_max/sqrt(2w), x-term x_max*sqrt(w/2)) amplify residue
// FASTER on the way down (the signal shrinks as sqrt(n)), which showed up
// as visible high-k garbage in the scene; stable rungs kill it, and the
// usable range becomes the grid's representability ceiling (ho_level_cap),
// not the raw-chain noise cap. Down at the ground still refuses via the
// operator itself (returns ~0, psi untouched). If psi is NOT the claimed
// eigenstate (oracle overlap < 1/2), the raw result is kept as-is: the
// caller misclassified, and the honest operator output stands.
inline double ladder_rung_stable(Field1D& psi, double omega, int n_from,
                                 bool up, double vanish_eps = 1e-6) {
    const Grid1D& g = psi.grid();
    Field1D raw = psi;
    double norm2 = 0.0;
    if (up) {
        norm2 = ladder_raise(raw, omega);
    } else {
        norm2 = ladder_lower(raw, omega, vanish_eps);
        if (norm2 < vanish_eps) {
            return norm2;  // annihilation: psi untouched, caller's signal
        }
    }
    const int target = up ? n_from + 1 : n_from - 1;
    const Field1D oracle = ho_eigenstate(g, omega, target);
    std::complex<double> ov{};
    for (int i = 0; i < g.n; ++i) {
        ov += std::conj(oracle[i]) * raw[i];
    }
    ov *= g.spacing();
    if (std::norm(ov) < 0.5) {
        psi = std::move(raw);  // misclassified input: keep the raw output
        return norm2;
    }
    const std::complex<double> phase = ov / std::abs(ov);
    for (int i = 0; i < g.n; ++i) {
        psi[i] = phase * oracle[i];
    }
    return norm2;
}

}  // namespace ses
