// RED: 1D harmonic-oscillator ladder operators over Field1D (atomic units,
// m = hbar = 1):
//
//     a    = sqrt(omega/2) x + 1/sqrt(2 omega) d/dx
//     adag = sqrt(omega/2) x - 1/sqrt(2 omega) d/dx
//
// acting on eigenstates: adag|n> = sqrt(n+1)|n+1>, a|n> = sqrt(n)|n-1>,
// a|0> = 0. The load-bearing contracts:
//
//  - ladder_raise(psi, omega): applies adag, returns ||adag psi||^2 BEFORE
//    the internal renormalization (== n+1 on |n>), leaves psi normalized.
//    Never vanishes: <psi|a adag|psi> = <N> + 1 >= 1 on any normalized state.
//  - ladder_lower(psi, omega, vanish_eps): applies a, returns ||a psi||^2
//    (== n on |n>, == <N> in general). When the result norm falls below
//    vanish_eps the state was annihilated (a|0> = 0): psi is left UNCHANGED
//    and the return value is the caller's forbidden-transition signal.
//  - The derivative is spectral (FFT), consistent with the split-operator
//    periodic grid, so eigenstate chains stay clean to near machine epsilon.
//
// Ground state input: gaussian_wavepacket with sigma = 1/sqrt(2 omega) IS the
// exact HO ground state (E = omega/2), so the whole Fock chain is reachable
// by repeated raises with no diagonalization anywhere.

#include <gtest/gtest.h>

#include <complex>
#include <vector>

import ses.field;
import ses.grid;
import ses.ladder;
import ses.observables;
import ses.potential;
import ses.wavepacket;

namespace {

constexpr double kOmega = 0.25;
const ses::Grid1D kGrid{-20.0, 20.0, 1024};

ses::Field1D ho_ground() {
    // sigma = 1/sqrt(2 omega) makes the Gaussian the exact HO ground state.
    const double sigma = 1.0 / std::sqrt(2.0 * kOmega);
    return ses::gaussian_wavepacket(kGrid, 0.0, sigma, 0.0);
}

// Discrete overlap |<a|b>|^2 with grid weight (h cancels in normalized pairs).
double overlap_sq(const ses::Field1D& a, const ses::Field1D& b) {
    std::complex<double> acc{};
    for (int i = 0; i < a.size(); ++i) {
        acc += std::conj(a[i]) * b[i];
    }
    acc *= a.grid().spacing();
    return std::norm(acc);
}

TEST(Ladder, GroundStateSetupHasEnergyHalfOmega) {
    const ses::Field1D psi = ho_ground();
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    EXPECT_NEAR(ses::mean_energy(psi, v), 0.5 * kOmega, 1e-10);
}

TEST(Ladder, RaiseNormSqCountsNPlusOneAndEnergyClimbsTheFockChain) {
    ses::Field1D psi = ho_ground();
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    for (int n = 0; n < 8; ++n) {
        const double norm2 = ses::ladder_raise(psi, kOmega);
        EXPECT_NEAR(norm2, static_cast<double>(n + 1), 1e-9)
            << "raise from n = " << n;
        EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12) << "renormalized after raise";
        EXPECT_NEAR(ses::mean_energy(psi, v), (n + 1 + 0.5) * kOmega, 1e-9)
            << "energy after raise to n = " << n + 1;
    }
}

TEST(Ladder, LowerNormSqCountsNAndStepsBackDown) {
    ses::Field1D psi = ho_ground();
    for (int n = 0; n < 3; ++n) {
        ses::ladder_raise(psi, kOmega);
    }
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    for (int n = 3; n > 0; --n) {
        const double norm2 = ses::ladder_lower(psi, kOmega);
        EXPECT_NEAR(norm2, static_cast<double>(n), 1e-9) << "lower from n = " << n;
        EXPECT_NEAR(ses::mean_energy(psi, v), (n - 1 + 0.5) * kOmega, 1e-9)
            << "energy after lower to n = " << n - 1;
    }
}

TEST(Ladder, LowerOnGroundIsAnnihilationAndLeavesPsiUntouched) {
    ses::Field1D psi = ho_ground();
    const ses::Field1D before = psi;
    const double norm2 = ses::ladder_lower(psi, kOmega);
    EXPECT_LT(norm2, 1e-9) << "a|0> = 0 up to discretization";
    for (int i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(psi[i], before[i]) << "annihilated apply must not write psi";
    }
}

TEST(Ladder, RaiseThenLowerRoundTripsToTheGroundState) {
    ses::Field1D psi = ho_ground();
    const ses::Field1D ground = psi;
    ses::ladder_raise(psi, kOmega);
    ses::ladder_lower(psi, kOmega);
    EXPECT_NEAR(overlap_sq(psi, ground), 1.0, 1e-10);
}

TEST(Ladder, LowerOnSuperpositionKeepsTheReachablePart) {
    // psi = (|0> + |1>)/sqrt(2): a psi = (1/sqrt(2))|0>, so the apply is NOT
    // annihilated (norm^2 = 1/2 >> vanish_eps) and the result is pure ground.
    ses::Field1D ground = ho_ground();
    ses::Field1D excited = ground;
    ses::ladder_raise(excited, kOmega);
    ses::Field1D psi = ground;
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = (ground[i] + excited[i]);
    }
    ses::normalize(psi);
    const double norm2 = ses::ladder_lower(psi, kOmega);
    EXPECT_NEAR(norm2, 0.5, 1e-9);
    EXPECT_NEAR(overlap_sq(psi, ground), 1.0, 1e-9);
}

}  // namespace
