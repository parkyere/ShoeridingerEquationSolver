// RED: the MCWF NO-JUMP evolution -- the piece decay.hpp deliberately left to
// the caller (quantum_jump only handles the jump; on survival psi was
// untouched). The app drains excited amplitudes by exp(-gamma_n dt/2) then
// renormalizes between jumps (the visible "breathe-out"). That amplitude
// transform had NO net: every decay/cascade/rabi arc passed with the whole
// branch disabled. This is the missing pure-logic net; the director's GPU
// apply_mcwf_damping mirrors it.

#include <complex>
#include <core/decay.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {


double pop(const std::vector<std::complex<double>>& c, int i) {
    return std::norm(c[static_cast<std::size_t>(i)]);
}

TEST(NoJumpDamping, IsIdentityOnAPureEigenstate) {
    // One occupied state: damping scales it, renorm restores it -- unchanged.
    const std::vector<std::complex<double>> c{{0.0, 0.0}, {0.6, -0.8}, {0.0, 0.0}};
    const std::vector<double> gamma{0.0, 0.13, 0.0};
    const std::vector<std::complex<double>> out =
        ses::nojump_damped_amplitudes(c, gamma, 2.0);
    EXPECT_NEAR(out[1].real(), 0.6, 1e-12);
    EXPECT_NEAR(out[1].imag(), -0.8, 1e-12);
    EXPECT_NEAR(pop(out, 0), 0.0, 1e-12);
    EXPECT_NEAR(pop(out, 2), 0.0, 1e-12);
}

TEST(NoJumpDamping, DrainsTheFasterDecayingComponentAndConservesNorm) {
    // Equal 50/50 of a stable (gamma=0) and a decaying state: after the
    // no-jump interval the stable one carries MORE than half, the decaying
    // one LESS, and the total is still 1.
    const double s = 1.0 / std::sqrt(2.0);
    const std::vector<std::complex<double>> c{{s, 0.0}, {s, 0.0}};
    const std::vector<double> gamma{0.0, 0.4};
    const std::vector<std::complex<double>> out =
        ses::nojump_damped_amplitudes(c, gamma, 1.0);
    EXPECT_GT(pop(out, 0), 0.5);
    EXPECT_LT(pop(out, 1), 0.5);
    EXPECT_NEAR(pop(out, 0) + pop(out, 1), 1.0, 1e-12);
    // Analytic: p0 = 1/(1+e^{-gamma dt}) with gamma dt = 0.4.
    EXPECT_NEAR(pop(out, 0), 1.0 / (1.0 + std::exp(-0.4)), 1e-12);
}

TEST(NoJumpDamping, GroundGrowsMonotonicallyTowardOne) {
    // Repeated application (the app between jumps): the ground fraction rises
    // every interval and approaches 1 as the excited amplitude breathes out.
    std::vector<std::complex<double>> c{{0.5, 0.0}, {std::sqrt(0.75), 0.0}};
    const std::vector<double> gamma{0.0, 0.25};
    double prev = pop(c, 0);
    for (int step = 0; step < 40; ++step) {
        c = ses::nojump_damped_amplitudes(c, gamma, 1.0);
        EXPECT_GT(pop(c, 0), prev - 1e-15);  // never decreases
        prev = pop(c, 0);
    }
    EXPECT_GT(pop(c, 0), 0.99);  // excited fraction has drained away
}

TEST(NoJumpDamping, DegenerateShellKeepsRelativeWeights) {
    // Two states sharing one gamma (a degenerate manifold): both scale by the
    // same factor, so their ratio survives -- damping drains the shell as a
    // whole, it does not repartition within it.
    const std::vector<std::complex<double>> c{{0.6, 0.0}, {0.0, 0.8}};
    const std::vector<double> gamma{0.3, 0.3};
    const std::vector<std::complex<double>> out =
        ses::nojump_damped_amplitudes(c, gamma, 1.5);
    EXPECT_NEAR(pop(out, 0) / pop(out, 1), 0.36 / 0.64, 1e-12);
    EXPECT_NEAR(pop(out, 0) + pop(out, 1), 1.0, 1e-12);
}

// ---- ionization tally accounting (bound_survival_ratio) -------------------
// The app reports an "ionized %" from a running bound-survival product; the
// accounting had NO net (ionized_fraction() was not even Shell-reachable) and
// carries a sign/back-out risk: absorbed flux IS ionization, but MCWF H_eff
// damping is NOT and must be backed out. These pin exactly that.

TEST(IonizationTally, NoLossLeavesSurvivalUnchanged) {
    EXPECT_NEAR(ses::bound_survival_ratio(1.0, 0.0, 1.0), 1.0, 1e-15);
}

TEST(IonizationTally, AbsorbedFluxCountsAsIonization) {
    // 10% of the norm left through the absorber this interval.
    EXPECT_NEAR(ses::bound_survival_ratio(0.9, 0.0, 1.0), 0.9, 1e-15);
}

TEST(IonizationTally, HeffDampingIsBackedOutNotCountedAsIonization) {
    // The no-jump damping removed L=0.05 of the norm; with no absorption the
    // survival must stay 1 (draining excited amplitude is not escape).
    const double L = 0.05;
    EXPECT_NEAR(ses::bound_survival_ratio(1.0 - L, L, 1.0), 1.0, 1e-15);
}

TEST(IonizationTally, IsolatesAbsorptionWhenBothActed) {
    // Absorber leaves 95%, then H_eff damping removes another L=0.05 of the
    // ORIGINAL norm -> post = 0.95 - 0.05 = 0.90; backing out L recovers the
    // pure absorbed survival 0.95.
    const double L = 0.05;
    EXPECT_NEAR(ses::bound_survival_ratio(0.95 - L, L, 1.0), 0.95, 1e-15);
}

}  // namespace
