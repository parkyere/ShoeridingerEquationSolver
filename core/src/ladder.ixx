module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>
export module ses.ladder;
export import ses.field;
export import ses.grid;
import ses.fft;
import ses.parallel;
import ses.spectral;


// 1D HO ladder operators (atomic units, m = hbar = 1):
//     a, adag = sqrt(omega/2) x +- 1/sqrt(2 omega) d/dx;
//     adag|n> = sqrt(n+1)|n+1>, a|n> = sqrt(n)|n-1>, a|0> = 0.
// The derivative is spectral (FFT ik multiply), consistent with the
// split-operator periodic grid, so repeated applications climb/descend
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

// ---- scaled Hermite chain ------------------------------------------------
// The plain recurrence seeds psi_0 = a0 exp(-omega x^2 / 2), which leaves
// double range past omega x^2 / 2 > ~745 -- the seed becomes EXACT ZERO
// there, and a zero seed stays zero at every level (the recurrence only
// multiplies and mixes), silently deleting the outer lobes of deep states
// (at n = 1200, omega = 1 that is ~44% of the probability). The chain is
// therefore carried per point as mantissa * 2^exponent. Every rescale is a
// power of two -- EXACT in binary floating point -- so wherever the plain
// chain never left double range the mantissa path computes bitwise the
// same values: the representation extends the range, never the arithmetic.

constexpr double kChainHi = 0x1p+500;  // rescale down above this ...
constexpr double kChainLo = 0x1p-500;  // ... or up when BOTH fall below
constexpr double kChainDn = 0x1p-512;
constexpr double kChainUp = 0x1p+512;
constexpr int kChainExp = 512;

// psi_0(x) as (mantissa, exponent): exact (exponent 0) wherever the plain
// exp is a normal double; log2-domain split otherwise (relative error
// ~|log2 psi_0| * eps ~ 1e-11, in a region physically below 1e-290).
inline void chain_seed(double a0, double omega, double x, double& m, int& e) {
    const double arg = 0.5 * omega * x * x;
    if (arg < 690.0) {
        m = a0 * std::exp(-arg);
        e = 0;
        return;
    }
    const double bits = -arg * 1.4426950408889634;  // log2(psi_0 / a0)
    const int shift = static_cast<int>(std::floor(bits));
    m = a0 * std::exp2(bits - shift);
    e = shift;
}

// One recurrence rung on a scaled (prev, cur) pair sharing one exponent.
// c1x = sqrt(2 omega / k) * x, c2 = sqrt((k-1)/k) -- the same expressions,
// in the same order, as the plain chain (bitwise-identical when no rescale
// triggers; a rescale only triggers outside (2^-500, 2^500), where the
// plain chain was already denormal or could never arrive).
inline void chain_advance(double c1x, double c2, double& p, double& c,
                          int& e) {
    const double next = c1x * c - c2 * p;
    p = c;
    c = next;
    const double ac = std::abs(c);
    if (ac > kChainHi || std::abs(p) > kChainHi) {
        p *= kChainDn;
        c *= kChainDn;
        e += kChainExp;
    } else if (ac < kChainLo && std::abs(p) < kChainLo && e != 0) {
        p *= kChainUp;
        c *= kChainUp;
        e -= kChainExp;
    }
}

// The whole-grid scaled chain: SoA rows (p, c, e) advanced level by level.
// advance_to runs TRANSPOSED (each worker walks its own points through all
// pending levels with the shared coefficient tables) -- one pool dispatch
// per block, per-point work independent, so the result is deterministic.
// Copying the object is the snapshot used by ho_level_cap's bisection.
class ScaledChain {
  public:
    ScaledChain(const Grid1D& g, double omega)
        : g_(&g),
          omega_(omega),
          p_(static_cast<std::size_t>(g.n), 0.0),
          c_(static_cast<std::size_t>(g.n)),
          e_(static_cast<std::size_t>(g.n), 0) {
        const double pi = 3.14159265358979323846;
        const double a0 = std::pow(omega / pi, 0.25);
        parallel_for(g.n, [&](int i) {
            const std::size_t s = static_cast<std::size_t>(i);
            chain_seed(a0, omega_, g_->coord(i), c_[s], e_[s]);
        });
    }

    int level() const noexcept { return level_; }

    void advance_to(int target) {
        if (target <= level_) {
            return;
        }
        ensure_coeffs(target);
        const int from = level_;
        parallel_ranges(g_->n, [&](int, int begin, int end) {
            for (int i = begin; i < end; ++i) {
                const std::size_t s = static_cast<std::size_t>(i);
                const double x = g_->coord(i);
                double pp = p_[s];
                double cc = c_[s];
                int ee = e_[s];
                for (int k = from + 1; k <= target; ++k) {
                    chain_advance(r1_[static_cast<std::size_t>(k)] * x,
                                  r2_[static_cast<std::size_t>(k)], pp, cc,
                                  ee);
                }
                p_[s] = pp;
                c_[s] = cc;
                e_[s] = ee;
            }
        });
        level_ = target;
    }

    // The chain's value at point i (0 when genuinely below double range).
    double value(int i) const noexcept {
        const std::size_t s = static_cast<std::size_t>(i);
        return std::ldexp(c_[s], e_[s]);
    }

  private:
    void ensure_coeffs(int target) {
        const std::size_t need = static_cast<std::size_t>(target) + 1;
        while (r1_.size() < need) {
            const double k = static_cast<double>(r1_.size());
            r1_.push_back(std::sqrt(2.0 * omega_ / k));
            r2_.push_back(std::sqrt((k - 1.0) / k));
        }
    }

    const Grid1D* g_;
    double omega_;
    int level_ = 0;
    std::vector<double> r1_{0.0};  // sqrt(2 omega / k), index k >= 1
    std::vector<double> r2_{0.0};  // sqrt((k-1)/k)
    std::vector<double> p_;        // mantissa psi_{level-1}
    std::vector<double> c_;        // mantissa psi_{level}
    std::vector<int> e_;           // shared per-point exponent
};

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
// Carried as a per-point scaled (mantissa, 2^exponent) chain so the seed
// never leaves double range: deep levels keep their outer lobes past the
// plain exp's underflow wall (|x| ~ 38.6/sqrt(omega)), and the ceiling
// becomes the honest grid physics (box vs Nyquist), not the FP floor.
// Eigenbasis decomposition of psi up to e_max: (E_n, |<n|psi>|^2) pairs,
// E_n = (n + 1/2) omega -- the LINEAR-COMBINATION spectrum the HUD
// strip stacks (not emitted photons: what the cloud IS made of).
// CONTRACT: tests/ho_spectrum_test.cpp.
inline std::vector<std::pair<double, double>> ho1d_spectrum(
    const Field1D& /*psi*/, double /*omega*/, double /*e_max*/) {
    return {};  // RED stub
}

inline Field1D ho_eigenstate(const Grid1D& g, double omega, int n) {
    ladder_detail::ScaledChain chain{g, omega};
    chain.advance_to(n);
    Field1D cur{g};
    parallel_for(g.n, [&](int i) { cur[i] = chain.value(i); });
    normalize(cur);
    return cur;
}

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
    // Faithful = (a) CONTAINED and (b) energy-exact.
    // (a) catches box truncation, which the energy check CANNOT: a clipped
    //     Hermite slice still satisfies k(x)^2/2 + V = E locally at every
    //     sample, so its grid energy stays within ~1e-5 of (n+1/2)w even
    //     with the turning points outside the box (measured at w = 4 on
    //     the 65536-pt scene grid, where energy alone ran the cap into
    //     the arbitrary probe bound). A representable eigenstate has its
    //     tail decayed to nothing at the edges: boundary density below
    //     1e-6 of the bulk peak.
    // (b) catches the Nyquist side (aliased spectral <T>) and any leftover
    //     construction error; scale-invariant, so the unnormalized chain
    //     is fine.
    std::vector<std::complex<double>> phi(static_cast<std::size_t>(g.n));
    auto faithful = [&](const ladder_detail::ScaledChain& chain, int n) {
        parallel_for(g.n, [&](int i) {
            phi[static_cast<std::size_t>(i)] = chain.value(i);
        });
        double num_v = 0.0;
        double den_x = 0.0;
        double bulk = 0.0;
        for (std::size_t j = 0; j < phi.size(); ++j) {
            const double w = std::norm(phi[j]);
            num_v += v[j] * w;
            den_x += w;
            bulk = std::max(bulk, w);
        }
        const double edge = std::max(std::norm(phi.front()),
                                     std::norm(phi.back()));
        if (edge > 1e-6 * bulk) {
            return false;  // leaks out of the box: not this grid's state
        }
        fft(phi);
        double num_t = 0.0;
        double den_k = 0.0;
        for (std::size_t j = 0; j < phi.size(); ++j) {
            const double w = std::norm(phi[j]);
            num_t += 0.5 * k[j] * k[j] * w;
            den_k += w;
        }
        const double e = num_t / den_k + num_v / den_x;
        const double e_exact = (n + 0.5) * omega;
        return std::abs(e - e_exact) <= 1e-3 * e_exact;
    };
    // Probe bound: a SAFETY NET above the physics ceilings, not the cap.
    // Near n_box (level energy = edge potential) faithfulness fades over
    // the Airy transition width (~|dV/dx| * (2 omega^2 x)^{-1/3} in energy,
    // a few hundred levels on fine grids), so the measured boundary can
    // overhang n_box slightly; 1.5x + 64 is comfortably past it on both
    // the box and the Nyquist side. The scan STOPS at the first failed
    // check, so the generous bound costs nothing -- it only guards the
    // loop. GRID-DERIVED, not a constant (the old fixed 400 silently
    // clipped fine grids whose true ceiling sits in the thousands).
    const double x_edge = std::min(std::abs(g.xmin), std::abs(g.xmax));
    const double k_max = 3.14159265358979323846 / g.spacing();
    const double n_box = 0.5 * omega * x_edge * x_edge;
    const double n_nyq = 0.5 * k_max * k_max / omega;
    const int bound = std::max(
        1,
        static_cast<int>(std::min(1.5 * std::min(n_box, n_nyq), 1048576.0)) +
            64);
    // Representability is MONOTONE in n (higher n = wider turning points
    // AND higher k content: once lost, never regained). Stride scan with
    // the scaled chain, snapshot (= chain copy) at the last good check,
    // then BISECT the last window from the snapshot: O(chain) grid work
    // plus ~(bound/stride + log2 stride) FFT energy checks -- this runs
    // on every omega-slider apply, at up to 64k points per level.
    const int stride = std::clamp(bound / 64, 16, 512);
    ladder_detail::ScaledChain chain{g, omega};
    ladder_detail::ScaledChain snap = chain;  // rows at last_good
    int last_good = 0;
    int first_bad = -1;
    while (chain.level() < bound) {
        chain.advance_to(std::min(chain.level() + stride, bound));
        if (faithful(chain, chain.level())) {
            last_good = chain.level();
            snap = chain;
        } else {
            first_bad = chain.level();
            break;
        }
    }
    if (first_bad < 0) {
        return last_good;  // clean through the physics bound
    }
    while (first_bad - last_good > 1) {
        const int mid = last_good + (first_bad - last_good) / 2;
        chain = snap;
        chain.advance_to(mid);
        if (faithful(chain, mid)) {
            last_good = mid;
            snap = std::move(chain);
        } else {
            first_bad = mid;
        }
    }
    return last_good;
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
    const double psi_n2 = norm_sq(psi);
    // Pass 1: project c_n = <n|psi> level by level off ONE scaled chain --
    // no basis storage (the old stored basis was O(band * grid) memory,
    // which forbade deep bands). EARLY STOP once the scanned band has
    // captured all but 1e-10 of the state: every unscanned |c_n|^2 is then
    // bounded by that leftover, so the residual claim stays honest and the
    // common case (a low superposition under a deep band top) costs only
    // the levels the state actually occupies. Per-level sums use the
    // chunk-ordered parallel reduction (bitwise-deterministic).
    struct DotAcc {
        std::complex<double> dot{};
        double n2 = 0.0;
        DotAcc& operator+=(const DotAcc& o) {
            dot += o.dot;
            n2 += o.n2;
            return *this;
        }
    };
    ladder_detail::ScaledChain chain{g, omega};
    std::vector<std::complex<double>> c;
    std::vector<double> inv_norm;  // 1 / ||chain level||_h, cached for pass 2
    double inside = 0.0;
    for (int n = 0; n <= n_top; ++n) {
        chain.advance_to(n);
        const DotAcc acc =
            parallel_sum(g.n, DotAcc{}, [&](int i) {
                const double val = chain.value(i);
                return DotAcc{val * psi[i], val * val};
            });
        const double inv = 1.0 / std::sqrt(acc.n2 * h);
        inv_norm.push_back(inv);
        c.push_back(acc.dot * h * inv);
        inside += std::norm(c.back());
        if (inside >= (1.0 - 1e-10) * psi_n2) {
            break;  // the band holds the whole state; higher c_n ~ 0
        }
    }
    const int m = static_cast<int>(c.size()) - 1;  // last projected level
    const double residual = std::max(0.0, 1.0 - inside / psi_n2);
    if (out_residual != nullptr) {
        *out_residual = residual;
    }
    // Exact coefficient action on the scanned band.
    const int nd = m + (up ? 2 : 1);
    std::vector<std::complex<double>> d(static_cast<std::size_t>(nd));
    double norm2 = 0.0;
    if (up) {
        for (int n = 0; n <= m; ++n) {
            d[static_cast<std::size_t>(n + 1)] =
                std::sqrt(n + 1.0) * c[static_cast<std::size_t>(n)];
            norm2 += (n + 1.0) * std::norm(c[static_cast<std::size_t>(n)]);
        }
    } else {
        for (int n = 0; n + 1 <= m; ++n) {
            d[static_cast<std::size_t>(n)] =
                std::sqrt(n + 1.0) * c[static_cast<std::size_t>(n + 1)];
            norm2 += (n + 1.0) * std::norm(c[static_cast<std::size_t>(n + 1)]);
        }
    }
    if (residual > 0.5 || norm2 < vanish_eps) {
        return norm2;  // outside the band, or annihilated: psi untouched
    }
    // Pass 2: resynthesize psi = sum d_n |n> off a fresh chain, reusing the
    // cached level norms (the up-shift's top level n = m + 1 was not normed
    // in pass 1; it is measured here the same way).
    std::vector<std::complex<double>> out(static_cast<std::size_t>(g.n));
    ladder_detail::ScaledChain synth{g, omega};
    for (int n = 0; n < nd; ++n) {
        synth.advance_to(n);
        if (d[static_cast<std::size_t>(n)] == std::complex<double>{}) {
            continue;
        }
        double inv = 0.0;
        if (n < static_cast<int>(inv_norm.size())) {
            inv = inv_norm[static_cast<std::size_t>(n)];
        } else {
            const double lvl_n2 = parallel_sum(g.n, 0.0, [&](int i) {
                const double val = synth.value(i);
                return val * val;
            });
            inv = 1.0 / std::sqrt(lvl_n2 * h);
        }
        const std::complex<double> w = d[static_cast<std::size_t>(n)] * inv;
        parallel_for(g.n, [&](int i) {
            out[static_cast<std::size_t>(i)] += w * synth.value(i);
        });
    }
    const double inv_total = 1.0 / std::sqrt(norm2);
    parallel_for(g.n, [&](int i) {
        psi[i] = inv_total * out[static_cast<std::size_t>(i)];
    });
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
