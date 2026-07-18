module;
#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.bloch;
export import ses.field;
export import ses.grid;
import ses.fft;
import ses.spectral;


// 1D periodic-lattice (solid-state) support for V(x) = V0 sin^2(kL x) --
// SMOOTH on purpose: Kronig-Penney's kinks would Gibbs-ring in the FFT
// plane-wave basis, sin^2 keeps spectral accuracy (and is exactly the
// optical-lattice / Mathieu problem). Lattice constant a = pi / kL,
// reciprocal vector G = 2 kL.
//
// Band structure: V0 sin^2 = V0/2 - (V0/4)(e^{iGx} + e^{-iGx}) has ONE
// harmonic, so the central equation in the plane-wave basis {q + m G} is
// symmetric TRIDIAGONAL: diagonal (q + m G)^2/2 + V0/2, off-diagonal
// -V0/4. lattice_bands finds its lowest eigenvalues by Sturm bisection
// (deterministic, no iteration-order noise).
//
// Bloch driving: a uniform force F is the potential -F x, which breaks
// the periodic box -- but in the comoving gauge it is EXACTLY the
// time-dependent uniform vector potential A(t) = -F t. The kinetic phase
// e^{-i (k - A)^2 dt / 2} is rebuilt each step at the MIDPOINT A (the
// midpoint rule is exact for a linear A), so the free-particle limit
// reproduces uniform acceleration to round-off.


export namespace ses {

namespace bloch_detail {

// Eigenvalues of the symmetric tridiagonal T (diag d, off-diagonal o)
// strictly below lambda, via the RATIO form of the Sturm sequence
// (q_i = (d_i - lambda) - o_{i-1}^2 / q_{i-1}; count the negatives).
// The product form dies when a bisection midpoint hits an eigenvalue of
// a leading minor exactly (the sequence goes 0 and stays 0 -- with the
// dyadic midpoints of an integer bracket this HAPPENS); the ratio form
// just nudges a zero negative and carries on (LAPACK dlaebz style).
inline int sturm_count(const std::vector<double>& d,
                       const std::vector<double>& o, double lambda) {
    int count = 0;
    double q = 1.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        const double off2 = i > 0 ? o[i - 1] * o[i - 1] : 0.0;
        if (q == 0.0) {
            q = -1e-300;
        }
        q = (d[i] - lambda) - off2 / q;
        if (q < 0.0) {
            ++count;
        }
    }
    return count;
}

}  // namespace bloch_detail

// The lowest n_bands energies E_n(q) of V(x) = v0 sin^2(kl x) at
// quasimomentum q, by exact diagonalization of the (tridiagonal) central
// equation over plane waves q + m G, m = -M..M.
inline std::vector<double> lattice_bands(double v0, double kl, double q,
                                         int n_bands) {
    const double g2 = 2.0 * kl;  // reciprocal vector
    const int m_max = std::max(8, n_bands + 6);
    const int n = 2 * m_max + 1;
    std::vector<double> d(static_cast<std::size_t>(n));
    std::vector<double> o(static_cast<std::size_t>(n - 1), -0.25 * v0);
    for (int m = -m_max; m <= m_max; ++m) {
        const double k = q + m * g2;
        d[static_cast<std::size_t>(m + m_max)] = 0.5 * k * k + 0.5 * v0;
    }
    // Sturm bisection per band index (bracket: Gershgorin bounds).
    double lo = d[0];
    double hi = d[0];
    for (std::size_t i = 0; i < d.size(); ++i) {
        const double r = (i > 0 ? std::abs(o[i - 1]) : 0.0) +
                         (i + 1 < d.size() ? std::abs(o[i]) : 0.0);
        lo = std::min(lo, d[i] - r);
        hi = std::max(hi, d[i] + r);
    }
    std::vector<double> bands;
    bands.reserve(static_cast<std::size_t>(n_bands));
    for (int band = 0; band < n_bands; ++band) {
        double a = lo;
        double b = hi;
        for (int it = 0; it < 200; ++it) {
            const double mid = 0.5 * (a + b);
            if (bloch_detail::sturm_count(d, o, mid) <= band) {
                a = mid;
            } else {
                b = mid;
            }
        }
        bands.push_back(0.5 * (a + b));
    }
    return bands;
}

// Split-operator with the comoving-gauge tilt A(t) = -F t: the SAME
// Strang structure as SplitOperator1D, but the kinetic phase table is
// rebuilt every step with the midpoint drift.
class TiltedSplitOperator1D {
public:
    TiltedSplitOperator1D(const Grid1D& g, const std::vector<double>& potential,
                          double dt, double force)
        : dt_(dt), force_(force), k_(wavenumbers(g)) {
        assert(static_cast<int>(potential.size()) == g.n);
        half_v_.resize(potential.size());
        for (std::size_t i = 0; i < potential.size(); ++i) {
            const double th = -0.5 * potential[i] * dt;
            half_v_[i] = std::complex<double>{std::cos(th), std::sin(th)};
        }
        kinetic_.resize(k_.size());
    }

    double dt() const noexcept { return dt_; }
    // Accumulated drift A(t) = F t: the packet's quasimomentum shift
    // (the scene's Brillouin-zone marker reads it).
    double drift() const noexcept { return force_ * t_; }
    void reset_time() noexcept { t_ = 0.0; }

    void step(Field1D& psi, int nsteps = 1) {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            const double a_mid = -force_ * (t_ + 0.5 * dt_);
            for (std::size_t j = 0; j < k_.size(); ++j) {
                const double km = k_[j] - a_mid;
                const double th = -0.5 * km * km * dt_;
                kinetic_[j] =
                    std::complex<double>{std::cos(th), std::sin(th)};
            }
            apply_phase(half_v_, psi.data());
            fft(psi.data());
            apply_phase(kinetic_, psi.data());
            ifft(psi.data());
            apply_phase(half_v_, psi.data());
            t_ += dt_;
        }
    }

private:
    static void apply_phase(const std::vector<std::complex<double>>& phase,
                            std::vector<std::complex<double>>& a) noexcept {
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = a[i] * phase[i];
        }
    }

    double dt_;
    double force_;
    double t_ = 0.0;
    std::vector<double> k_;
    std::vector<std::complex<double>> half_v_;
    std::vector<std::complex<double>> kinetic_;
};

}  // namespace ses
