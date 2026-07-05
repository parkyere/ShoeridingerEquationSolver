#pragma once

// Spontaneous decay via quantum jumps (transitions arc T4, the Monte-Carlo-
// wavefunction picture). The Schrodinger equation carries no lifetimes; the
// decay RATE follows from the computed spectrum through the Einstein A
// coefficient (atomic units):
//     A = (4/3) alpha^3 omega^3 |<f|r|i>|^2
// so selection rules enter automatically (forbidden channel -> strength 0 ->
// A = 0, e.g. the metastable 2s). Jumps are a Poisson process weighted by
// the CURRENT excited population; randomness is injected by the caller.

#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

inline constexpr double kFineStructureConstant = 7.2973525693e-3;

struct DipoleMatrixElement {
    Complex<double> x;
    Complex<double> y;
    Complex<double> z;
};

// <f| r |i> component-wise: sum conj(f) * r * i * dV.
inline DipoleMatrixElement dipole_matrix_element(const Field3D& f, const Field3D& i) {
    const Grid3D& g = f.grid();
    Complex<double> dx{};
    Complex<double> dy{};
    Complex<double> dz{};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int ii = 0; ii < g.x.n; ++ii) {
                const Complex<double> t = std::conj(f(ii, j, k)) * i(ii, j, k);
                dx += g.x.coord(ii) * t;
                dy += g.y.coord(j) * t;
                dz += g.z.coord(k) * t;
            }
        }
    }
    const double dv = g.cell_volume();
    return DipoleMatrixElement{dv * dx, dv * dy, dv * dz};
}

inline double dipole_strength_sq(const DipoleMatrixElement& d) {
    return std::norm(d.x) + std::norm(d.y) + std::norm(d.z);
}

// Einstein A coefficient (atomic units): the spontaneous decay rate.
inline double einstein_a(double omega, double dipole_sq) {
    const double a3 = kFineStructureConstant * kFineStructureConstant *
                      kFineStructureConstant;
    return (4.0 / 3.0) * a3 * omega * omega * omega * dipole_sq;
}

struct JumpResult {
    bool jumped{};
    double p_jump{};
};

// One Poisson decay trial over an interval dt: jump probability
// p = 1 - exp(-gamma * P_e * dt) with P_e = |<excited|psi>|^2. On a jump the
// state collapses onto the ground state (the photon carries the rest away);
// on survival psi is untouched. u in [0,1) is the caller's uniform draw.
inline JumpResult quantum_jump(Field3D& psi, const Field3D& excited, const Field3D& ground,
                               double gamma, double dt, double u) {
    const double p_e = norm_sq(inner_product(excited, psi));
    const double p = 1.0 - std::exp(-gamma * p_e * dt);
    if (u < p) {
        psi = ground;
        return JumpResult{true, p};
    }
    return JumpResult{false, p};
}

// ---- multi-channel jumps (transitions arc T5) ------------------------------
//
// Every tracked orbital gets its lifetime: decay channels compete as
// independent Poisson processes. Channel m fires at rate gamma_m * P_m with
// P_m = |<from_m|psi>|^2; the total escape probability over dt is
//     p = 1 - exp(-sum_m gamma_m P_m dt),
// and WHICH channel fires is distributed proportionally to its rate. On a
// jump the MCWF jump operator |to><from| collapses psi onto the channel's
// destination eigenstate. The caller injects u1 (does a jump happen?) and
// u2 (which channel?), keeping randomness out of core.

struct DecayChannel {
    const Field3D* from;  // excited eigenstate
    const Field3D* to;    // destination eigenstate
    double gamma;         // Einstein A rate (0 = forbidden channel)
};

struct MultiJumpResult {
    int channel{-1};   // fired channel index, or -1 for no jump
    double p_total{};  // total jump probability over this interval
};

inline MultiJumpResult multi_quantum_jump(Field3D& psi,
                                          const std::vector<DecayChannel>& channels,
                                          double dt, double u1, double u2) {
    std::vector<double> rate(channels.size());
    double total = 0.0;
    for (std::size_t m = 0; m < channels.size(); ++m) {
        const double p_m = norm_sq(inner_product(*channels[m].from, psi));
        rate[m] = channels[m].gamma * p_m;
        total += rate[m];
    }
    if (total <= 0.0) {
        return MultiJumpResult{-1, 0.0};
    }
    const double p = 1.0 - std::exp(-total * dt);
    if (!(u1 < p)) {
        return MultiJumpResult{-1, p};
    }
    // Channel selection: u2 against the cumulative rate fractions. Zero-rate
    // channels occupy zero measure and can never be picked; the final
    // fallthrough guards u2 rounding at the top of the last stratum.
    double acc = 0.0;
    int last_positive = -1;
    for (std::size_t m = 0; m < channels.size(); ++m) {
        if (rate[m] <= 0.0) {
            continue;
        }
        last_positive = static_cast<int>(m);
        acc += rate[m] / total;
        if (u2 < acc) {
            psi = *channels[m].to;
            return MultiJumpResult{static_cast<int>(m), p};
        }
    }
    psi = *channels[static_cast<std::size_t>(last_positive)].to;
    return MultiJumpResult{last_positive, p};
}

}  // namespace ses
