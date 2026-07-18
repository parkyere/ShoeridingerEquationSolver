module;
#include <complex>
#include <cmath>
#include <cstddef>
#include <vector>
#include <cstdint>
export module ses.decay;
export import ses.grid;
export import ses.field;
import ses.parallel;


// Spontaneous decay via quantum jumps (Monte-Carlo wavefunction).
// Einstein A = (4/3) alpha^3 omega^3 |<f|r|i>|^2 (atomic units); selection
// rules enter automatically (forbidden channel -> A = 0, e.g. metastable 2s).
// Jumps are Poisson, weighted by the CURRENT excited population; randomness
// is injected by the caller.


export namespace ses {

inline constexpr double kFineStructureConstant = 7.2973525693e-3;

struct DipoleMatrixElement {
    std::complex<double> x;
    std::complex<double> y;
    std::complex<double> z;
};

// <f| r |i> component-wise: sum conj(f) * r * i * dV. Parallelized with
// per-z-slab partial sums combined in a FIXED order, so the result is
// deterministic run to run (the ses.parallel discipline).
inline DipoleMatrixElement dipole_matrix_element(const Field3D& f, const Field3D& i) {
    const Grid3D& g = f.grid();
    const int nz = g.z.n;
    std::vector<std::complex<double>> px(static_cast<std::size_t>(nz));
    std::vector<std::complex<double>> py(static_cast<std::size_t>(nz));
    std::vector<std::complex<double>> pz(static_cast<std::size_t>(nz));
    parallel_for(nz, [&](int k) {
        std::complex<double> dx{};
        std::complex<double> dy{};
        std::complex<double> dz{};
        for (int j = 0; j < g.y.n; ++j) {
            for (int ii = 0; ii < g.x.n; ++ii) {
                const std::complex<double> t = std::conj(f(ii, j, k)) * i(ii, j, k);
                dx += g.x.coord(ii) * t;
                dy += g.y.coord(j) * t;
                dz += g.z.coord(k) * t;
            }
        }
        px[static_cast<std::size_t>(k)] = dx;
        py[static_cast<std::size_t>(k)] = dy;
        pz[static_cast<std::size_t>(k)] = dz;
    });
    std::complex<double> dx{};
    std::complex<double> dy{};
    std::complex<double> dz{};
    for (int k = 0; k < nz; ++k) {
        dx += px[static_cast<std::size_t>(k)];
        dy += py[static_cast<std::size_t>(k)];
        dz += pz[static_cast<std::size_t>(k)];
    }
    const double dv = g.cell_volume();
    return DipoleMatrixElement{dv * dx, dv * dy, dv * dz};
}

inline constexpr double dipole_strength_sq(const DipoleMatrixElement& d) noexcept {
    return std::norm(d.x) + std::norm(d.y) + std::norm(d.z);
}

// Einstein A coefficient (atomic units): the spontaneous decay rate.
inline constexpr double einstein_a(double omega, double dipole_sq) noexcept {
    const double a3 = kFineStructureConstant * kFineStructureConstant *
                      kFineStructureConstant;
    return (4.0 / 3.0) * a3 * omega * omega * omega * dipole_sq;
}

struct JumpResult {
    bool jumped{};
    double p_jump{};
};

// One Poisson decay trial over dt: p = 1 - exp(-gamma * P_e * dt),
// P_e = |<excited|psi>|^2; jump -> psi = ground. On survival psi is untouched
// HERE -- callers add the MCWF no-jump H_eff damping separately (the app's
// GPU path does). u in [0,1) is the caller's uniform draw.
inline JumpResult quantum_jump(Field3D& psi, const Field3D& excited, const Field3D& ground,
                               double gamma, double dt, double u) {
    const double p_e = std::norm(inner_product(excited, psi));
    const double p = 1.0 - std::exp(-gamma * p_e * dt);
    if (u < p) {
        psi = ground;
        return JumpResult{true, p};
    }
    return JumpResult{false, p};
}

// MCWF NO-JUMP over dt: c_n *= exp(-gamma_n * dt / 2), then renormalize
// (non-Hermitian H_eff + renorm) -- the visible "breathe-out" between jumps.
// Single source of truth the app's GPU apply_mcwf_damping mirrors in the
// {|n>} basis.
inline std::vector<std::complex<double>> nojump_damped_amplitudes(
    const std::vector<std::complex<double>>& c, const std::vector<double>& gamma,
    double dt) {
    std::vector<std::complex<double>> out(c.size());
    double n2 = 0.0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const double f = std::exp(-0.5 * gamma[i] * dt);
        out[i] = f * c[i];
        n2 += std::norm(out[i]);
    }
    if (n2 > 0.0) {
        const double inv = 1.0 / std::sqrt(n2);
        for (std::complex<double>& z : out) {
            z = inv * z;
        }
    }
    return out;
}

// Running bound-survival multiplier for one real-time interval. The state
// norm fell from `baseline` to `post_norm` because of (a) absorbed flux at
// the walls (= ionization) AND (b) any KNOWN non-absorber loss such as the
// MCWF H_eff damping above. Only (a) is ionization, so add the known loss
// back before ratioing. bound_survival *= this; 1 - bound_survival is the
// cumulative ionized fraction. Returns 1 (nothing escaped) if baseline<=0.
inline double bound_survival_ratio(double post_norm, double known_loss,
                                   double baseline) {
    return baseline > 0.0 ? (post_norm + known_loss) / baseline : 1.0;
}

// ---- multi-channel jumps ----------------------------------------------------
//
// Channels compete as independent Poisson processes: channel m fires at rate
// gamma_m * P_m, P_m = |<from_m|psi>|^2; total escape p = 1 - exp(-sum dt);
// which channel fires is proportional to its rate. A jump collapses psi onto
// the channel's |to>. Caller injects u1 (jump?) and u2 (which channel?).

struct DecayChannel {
    const Field3D* from;  // excited eigenstate
    const Field3D* to;    // destination eigenstate
    double gamma;         // Einstein A rate (0 = forbidden channel)
};

struct MultiJumpResult {
    int channel{-1};   // fired channel index, or -1 for no jump
    double p_total{};  // total jump probability over this interval
};

struct ChannelPick {
    int channel{-1};
    double p_total{};
};

// Pure selection arithmetic over precomputed rates r_m = gamma_m * P_m
// (reused by the GPU shell on GPU-reduced populations). Zero-rate channels
// occupy zero measure and can never be picked; the final fallthrough guards
// u2 rounding at the top of the last stratum.
inline ChannelPick pick_decay_channel(const std::vector<double>& rates, double dt,
                                      double u1, double u2) noexcept {
    double total = 0.0;
    for (const double r : rates) {
        total += r;
    }
    if (total <= 0.0) {
        return ChannelPick{-1, 0.0};
    }
    const double p = 1.0 - std::exp(-total * dt);
    if (!(u1 < p)) {
        return ChannelPick{-1, p};
    }
    double acc = 0.0;
    int last_positive = -1;
    for (std::size_t m = 0; m < rates.size(); ++m) {
        if (rates[m] <= 0.0) {
            continue;
        }
        last_positive = static_cast<int>(m);
        acc += rates[m] / total;
        if (u2 < acc) {
            return ChannelPick{static_cast<int>(m), p};
        }
    }
    return ChannelPick{last_positive, p};
}

inline MultiJumpResult multi_quantum_jump(Field3D& psi,
                                          const std::vector<DecayChannel>& channels,
                                          double dt, double u1, double u2) {
    std::vector<double> rates(channels.size());
    for (std::size_t m = 0; m < channels.size(); ++m) {
        rates[m] = channels[m].gamma * std::norm(inner_product(*channels[m].from, psi));
    }
    const ChannelPick pick = pick_decay_channel(rates, dt, u1, u2);
    if (pick.channel >= 0) {
        psi = *channels[static_cast<std::size_t>(pick.channel)].to;
    }
    return MultiJumpResult{pick.channel, pick.p_total};
}

}  // namespace ses
