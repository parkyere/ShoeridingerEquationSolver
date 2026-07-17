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

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

import ses.field;
import ses.grid;
import ses.ladder;
import ses.observables;
import ses.potential;
import ses.wavepacket;

namespace {

constexpr double kOmega = 0.25;
// Nyquist matched to the physics band: the spectral derivative in a-dag
// amplifies round-off noise at k_max = pi/h by k_max/sqrt(2 omega) per
// raise, so an oversampled grid (k_max >> the state's k content) walks the
// noise floor up the Fock chain exponentially. 256 points over +-20 gives
// k_max ~ 20 vs psi_8 k content ~ 2.5: clean to ~1e-13 through n = 8, while
// 1024 points (k_max ~ 80) already loses the chain at n = 8.
const ses::Grid1D kGrid{-20.0, 20.0, 256};

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

// Count interior sign changes of a real field (the eigenstate node count).
int node_count(const ses::Field1D& f) {
    int nodes = 0;
    double prev = 0.0;
    const double tiny = 1e-6;
    for (int i = 0; i < f.size(); ++i) {
        const double re = f[i].real();
        if (std::abs(re) < tiny) {
            continue;  // skip the amplitude-zero tails / node itself
        }
        if (prev != 0.0 && (re > 0.0) != (prev > 0.0)) {
            ++nodes;
        }
        prev = re;
    }
    return nodes;
}

// RED: ho_eigenstate(grid, omega, n) -- the exact HO eigenstate built
// DIRECTLY from the normalized Hermite-Gauss recurrence in x-space (no
// derivative, so NO spectral round-off amplification). This is the ground-
// truth oracle the ladder chain is measured against, and lets the scene
// display any grid-representable level without the ladder noise cap.
TEST(HoEigenstate, IsNormalizedWithExactEnergyAndZeroVariance) {
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    for (int n = 0; n <= 12; ++n) {
        const ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, n);
        EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-10) << "norm at n = " << n;
        EXPECT_NEAR(ses::mean_energy(psi, v), (n + 0.5) * kOmega, 1e-6)
            << "energy at n = " << n;
        EXPECT_LT(ses::energy_variance(psi, v), 1e-8) << "variance at n = " << n;
    }
}

TEST(HoEigenstate, HasNNodes) {
    for (int n = 0; n <= 8; ++n) {
        EXPECT_EQ(node_count(ses::ho_eigenstate(kGrid, kOmega, n)), n)
            << "node count at n = " << n;
    }
}

TEST(HoEigenstate, IsOrthonormalAcrossLevels) {
    for (int m = 0; m <= 6; ++m) {
        for (int n = 0; n <= 6; ++n) {
            const double ov = overlap_sq(ses::ho_eigenstate(kGrid, kOmega, m),
                                         ses::ho_eigenstate(kGrid, kOmega, n));
            EXPECT_NEAR(ov, m == n ? 1.0 : 0.0, 1e-9)
                << "<" << m << "|" << n << ">^2";
        }
    }
}

TEST(HoEigenstate, MatchesTheCleanLadderChainAtLowLevels) {
    // Where the ladder is still clean (n <= 4 at omega = 0.25), the direct
    // oracle and the ladder-built state are the same physical state (equal
    // up to a global phase, so overlap^2 = 1).
    ses::Field1D psi = ho_ground();
    for (int n = 1; n <= 4; ++n) {
        ses::ladder_raise(psi, kOmega);
        EXPECT_NEAR(overlap_sq(psi, ses::ho_eigenstate(kGrid, kOmega, n)), 1.0,
                    1e-7)
            << "ladder vs oracle at n = " << n;
    }
}

TEST(HoEigenstate, IsOmegaGeneric) {
    // A stiffer well (narrower ground): still exact.
    const double w = 0.75;
    const std::vector<double> v = ses::harmonic_potential(kGrid, w);
    for (int n = 0; n <= 6; ++n) {
        const ses::Field1D psi = ses::ho_eigenstate(kGrid, w, n);
        EXPECT_NEAR(ses::mean_energy(psi, v), (n + 0.5) * w, 1e-6)
            << "energy at n = " << n;
        EXPECT_LT(ses::energy_variance(psi, v), 1e-8);
    }
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

TEST(Ladder, ChainIsOmegaGeneric) {
    // The operators take omega as a parameter: a stiffer well (omega = 0.75)
    // must climb just as cleanly (its ground is a narrower Gaussian and its
    // noise gain k_max/sqrt(2 omega) is LOWER than at 0.25).
    const double w = 0.75;
    ses::Field1D psi =
        ses::gaussian_wavepacket(kGrid, 0.0, 1.0 / std::sqrt(2.0 * w), 0.0);
    const std::vector<double> v = ses::harmonic_potential(kGrid, w);
    for (int n = 0; n < 4; ++n) {
        EXPECT_NEAR(ses::ladder_raise(psi, w), n + 1.0, 1e-9);
        EXPECT_NEAR(ses::mean_energy(psi, v), (n + 1.5) * w, 1e-9);
    }
}

TEST(LadderCap, TracksTheSpectralNoiseModel) {
    // ladder_cap(omega, k_max): the largest Fock level the FFT ladder
    // reaches before the k_max round-off floor, amplified by
    // g/sqrt(n+1) per raise (g = k_max/sqrt(2 omega)), would become
    // display-visible. Anchors pin the tuned model at the scene grid
    // (k_max = pi/h, h = 40/256); monotone: stiffer well (larger omega)
    // -> lower gain -> higher cap, finer grid -> higher k_max -> lower cap.
    const double k_max = std::numbers::pi / (40.0 / 256.0);
    EXPECT_EQ(ses::ladder_cap(0.25, k_max), 10);  // the fixed cap it replaces
    EXPECT_EQ(ses::ladder_cap(1.0, k_max), 15);
    EXPECT_EQ(ses::ladder_cap(0.05, k_max), 7);
    EXPECT_LE(ses::ladder_cap(0.25, 2.0 * k_max), 10);
    EXPECT_GE(ses::ladder_cap(0.5, k_max), ses::ladder_cap(0.25, k_max));
    EXPECT_GE(ses::ladder_cap(0.05, k_max), 1);
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
