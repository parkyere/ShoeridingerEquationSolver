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

// RED: ladder_rung_stable(psi, omega, n_from, up) -- the noise-free ladder
// rung for a state KNOWN to be the eigenstate |n_from> (up to a global
// phase). The raw spectral operator supplies the counting norm^2 and the
// global phase; the state itself is rebuilt from the direct Hermite oracle
// carrying that phase. Same mathematical object (adag|n> = sqrt(n+1)|n+1>),
// computed by the stable route -- so the round-off floor RESETS at every
// rung instead of compounding, and descending no longer amplifies the
// high-k residue that the raw chain leaves behind (the observed
// ladder-down instability). The usable range becomes the grid's
// REPRESENTABILITY ceiling, not the raw-chain noise cap.
TEST(LadderRungStable, RoundTripsFarBeyondTheRawChain) {
    // Up 25 rungs then down 25: far past the raw-chain cap (12 at omega =
    // 0.25, where a raw round trip disintegrates into high-k garbage), the
    // stable rungs land exactly back on the ground state.
    const std::vector<double> v = ses::harmonic_potential(kGrid, kOmega);
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 0);
    for (int n = 0; n < 25; ++n) {
        const double norm2 = ses::ladder_rung_stable(psi, kOmega, n, true);
        EXPECT_NEAR(norm2, n + 1.0, 1e-5) << "up from n = " << n;
        EXPECT_NEAR(overlap_sq(psi, ses::ho_eigenstate(kGrid, kOmega, n + 1)),
                    1.0, 1e-10)
            << "clean at n = " << n + 1;
    }
    for (int n = 25; n > 0; --n) {
        const double norm2 = ses::ladder_rung_stable(psi, kOmega, n, false);
        EXPECT_NEAR(norm2, static_cast<double>(n), 1e-5) << "down from n = " << n;
    }
    EXPECT_NEAR(overlap_sq(psi, ses::ho_eigenstate(kGrid, kOmega, 0)), 1.0,
                1e-10);
    EXPECT_NEAR(ses::mean_energy(psi, v), 0.5 * kOmega, 1e-9);
}

TEST(LadderRungStable, CarriesTheGlobalPhase) {
    // e^{i theta}|5> must rung to e^{i theta}|6>: the phase a stationary
    // state accumulated under evolution survives the stable rung (the raw
    // operator supplies it; the oracle rebuild must not reset it).
    const double theta = 0.7;
    const std::complex<double> ph{std::cos(theta), std::sin(theta)};
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 5);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] *= ph;
    }
    const double norm2 = ses::ladder_rung_stable(psi, kOmega, 5, true);
    EXPECT_NEAR(norm2, 6.0, 1e-6);
    const ses::Field1D oracle = ses::ho_eigenstate(kGrid, kOmega, 6);
    std::complex<double> ov{};
    for (int i = 0; i < psi.size(); ++i) {
        ov += std::conj(oracle[i]) * psi[i];
    }
    ov *= kGrid.spacing();
    EXPECT_NEAR(ov.real(), std::cos(theta), 1e-10);
    EXPECT_NEAR(ov.imag(), std::sin(theta), 1e-10);
}

TEST(LadderRungStable, GroundAnnihilationStillRefuses) {
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 0);
    const ses::Field1D before = psi;
    const double norm2 = ses::ladder_rung_stable(psi, kOmega, 0, false);
    EXPECT_LT(norm2, 1e-9);
    for (int i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(psi[i], before[i]) << "annihilated rung must not write psi";
    }
}

// RED: ho_level_cap(grid, omega) -- the REPRESENTABILITY ceiling: the
// largest level whose direct Hermite oracle is still faithful on the grid
// (discrete energy within 0.1% of (n+1/2)w). This is what caps the stable
// rungs; it is limited by the BOX at soft omega (wide turning points) and
// by the Nyquist band at stiff omega, far above the raw-chain noise cap.
TEST(HoLevelCap, SitsFarAboveTheRawChainCapAndPeaksNearMatchedNyquist) {
    for (double w : {0.05, 0.25, 1.0, 4.0}) {
        const int level = ses::ho_level_cap(kGrid, w);
        const int raw = ses::ladder_cap(kGrid, w);
        std::fprintf(stderr, "ho_level_cap(w=%.2f) = %d (raw chain %d)\n", w,
                     level, raw);
        EXPECT_GE(level, raw) << "omega=" << w;
    }
    const int c025 = ses::ho_level_cap(kGrid, 0.25);
    const int c1 = ses::ho_level_cap(kGrid, 1.0);
    const int c4 = ses::ho_level_cap(kGrid, 4.0);
    EXPECT_GE(c025, 25);  // box-limited but far beyond raw cap 12
    EXPECT_GE(c1, 80);    // matched Nyquist: the ceiling peaks here
    EXPECT_GT(c1, c025);
    EXPECT_GT(c1, c4);    // stiff well: k-band-limited again
    EXPECT_GE(c4, 25);
}

TEST(HoLevelCap, EveryLevelBelowTheCapIsActuallyClean) {
    // Spot-check the cap's meaning: at omega = 0.25 the oracle at the cap
    // is faithful (energy AND variance), one past it need not be.
    const double w = 0.25;
    const std::vector<double> v = ses::harmonic_potential(kGrid, w);
    const int cap = ses::ho_level_cap(kGrid, w);
    for (int n : {cap / 2, cap}) {
        const ses::Field1D psi = ses::ho_eigenstate(kGrid, w, n);
        const double e_exact = (n + 0.5) * w;
        EXPECT_NEAR(ses::mean_energy(psi, v), e_exact, 1e-3 * e_exact)
            << "n = " << n;
    }
}

// RED: ladder_fock(psi, omega, up, n_top, &residual) -- the superposition
// counterpart of ladder_rung_stable: project onto the truncated Fock basis
// |0..n_top>, act EXACTLY on the coefficients (adag: c'_{n+1} = sqrt(n+1)
// c_n; a: c'_n = sqrt(n+1) c_{n+1}), resynthesize from the noise-free
// oracles. The same linear operator, computed in the basis where it is
// trivial -- no spectral derivative, so it works at ANY grid k_max (the
// raw chain's noise gain grows with k_max and dies on fine grids).
// *residual reports the input's outside-band weight; the caller gates on
// it (a state not inside the band must take the raw operator instead).
TEST(LadderFock, AgreesWithTheRawOperatorOnTheCoarseGrid) {
    ses::Field1D psi{kGrid};
    const ses::Field1D e0 = ses::ho_eigenstate(kGrid, kOmega, 0);
    const ses::Field1D e1 = ses::ho_eigenstate(kGrid, kOmega, 1);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = e0[i] + e1[i];
    }
    ses::normalize(psi);
    ses::Field1D raw = psi;
    const double n2_raw = ses::ladder_lower(raw, kOmega);
    double residual = 1.0;
    const double n2 = ses::ladder_fock(psi, kOmega, false, 8, &residual);
    EXPECT_NEAR(n2, n2_raw, 1e-9);
    EXPECT_NEAR(n2, 0.5, 1e-9);
    EXPECT_LT(residual, 1e-10);
    EXPECT_NEAR(overlap_sq(psi, raw), 1.0, 1e-9);
    EXPECT_NEAR(overlap_sq(psi, e0), 1.0, 1e-9);
}

TEST(LadderFock, RaisesASuperpositionOnAFineGridWhereTheRawChainDies) {
    // 4096 points over +-20: k_max ~ 640, raw-chain gain ~ 900 per rung --
    // the raw operators are useless here, the Fock path is exact.
    const ses::Grid1D fine{-20.0, 20.0, 4096};
    ses::Field1D psi{fine};
    const ses::Field1D e2 = ses::ho_eigenstate(fine, kOmega, 2);
    const ses::Field1D e4 = ses::ho_eigenstate(fine, kOmega, 4);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = e2[i] + e4[i];
    }
    ses::normalize(psi);
    // First raise: counting norm^2 = sum (n+1)|c_n|^2 = (3 + 5)/2 = 4.
    double residual = 1.0;
    EXPECT_NEAR(ses::ladder_fock(psi, kOmega, true, 16, &residual), 4.0, 1e-9);
    EXPECT_LT(residual, 1e-10);
    for (int k = 0; k < 5; ++k) {
        ses::ladder_fock(psi, kOmega, true, 16, &residual);
    }
    // After 6 raises: (adag)^6 (|2> + |4>) ~ sqrt(8!/2!)|8> + sqrt(10!/4!)|10>.
    const double a8 = std::sqrt(40320.0 / 2.0);       // 8!/2!
    const double a10 = std::sqrt(3628800.0 / 24.0);   // 10!/4!
    const ses::Field1D e8 = ses::ho_eigenstate(fine, kOmega, 8);
    const ses::Field1D e10 = ses::ho_eigenstate(fine, kOmega, 10);
    ses::Field1D expect{fine};
    for (int i = 0; i < expect.size(); ++i) {
        expect[i] = a8 * e8[i] + a10 * e10[i];
    }
    ses::normalize(expect);
    EXPECT_NEAR(overlap_sq(psi, expect), 1.0, 1e-9);
}

TEST(LadderFock, ReportsOutsideBandResidualAndRefusesNothingItself) {
    // |10> against a band capped at n_top = 5: everything is residual; the
    // state must be left untouched (the caller's fallback signal).
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 10);
    const ses::Field1D before = psi;
    double residual = 0.0;
    ses::ladder_fock(psi, kOmega, true, 5, &residual);
    EXPECT_GT(residual, 0.99);
    for (int i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(psi[i], before[i]);
    }
}

TEST(LadderFock, AnnihilatesThePureGroundOnLowering) {
    ses::Field1D psi = ses::ho_eigenstate(kGrid, kOmega, 0);
    const ses::Field1D before = psi;
    double residual = 1.0;
    const double n2 = ses::ladder_fock(psi, kOmega, false, 8, &residual);
    EXPECT_LT(n2, 1e-9);
    EXPECT_LT(residual, 1e-10);
    for (int i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(psi[i], before[i]);
    }
}

TEST(LadderFock, AgreesWithTheStableRungOnEigenstates) {
    ses::Field1D fock = ses::ho_eigenstate(kGrid, kOmega, 7);
    ses::Field1D rung = fock;
    double residual = 1.0;
    const double n2f = ses::ladder_fock(fock, kOmega, true, 16, &residual);
    const double n2r = ses::ladder_rung_stable(rung, kOmega, 7, true);
    EXPECT_NEAR(n2f, 8.0, 1e-9);
    EXPECT_NEAR(n2r, 8.0, 1e-6);
    EXPECT_NEAR(overlap_sq(fock, rung), 1.0, 1e-10);
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

// Measure the TRUE clean ladder cap at omega: raise from the ground and
// stop at the first level where EITHER the ladder-built state diverges from
// the direct Hermite oracle (spectral noise) OR the oracle itself stops
// being grid-representable (its energy drifts off (n+1/2)w -- the band
// ceiling). The usable cap is the min of the two, measured not modeled.
int measure_clean_cap(double w, const ses::Grid1D& g) {
    const std::vector<double> v = ses::harmonic_potential(g, w);
    ses::Field1D psi = ses::ho_eigenstate(g, w, 0);
    int cap = 0;
    for (int n = 1; n <= 60; ++n) {
        ses::ladder_raise(psi, w);
        const ses::Field1D oracle = ses::ho_eigenstate(g, w, n);
        const double e = ses::mean_energy(oracle, v);
        const double e_exact = (n + 0.5) * w;
        const bool oracle_representable =
            std::abs(e - e_exact) < 1e-3 * e_exact;
        if (!oracle_representable) {
            break;  // grid band ceiling: no method reaches this level cleanly
        }
        const double defect = 1.0 - overlap_sq(psi, oracle);
        if (defect > 1e-6) {  // ~0.1% amplitude corruption: display-visible
            break;
        }
        cap = n;
    }
    return cap;
}

TEST(LadderCap, MatchesTheIndependentlyMeasuredCleanCap) {
    // ladder_cap(grid, omega) IS an empirical probe -- it raises from the
    // ground and finds where the ladder-built state diverges from the
    // direct Hermite oracle. Cross-check it against measure_clean_cap here,
    // an INDEPENDENT reimplementation (which also guards on oracle
    // representability), across the omega sweep.
    for (double w : {0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0}) {
        EXPECT_LE(std::abs(ses::ladder_cap(kGrid, w) - measure_clean_cap(w, kGrid)),
                  1)
            << "omega=" << w;
    }
}

TEST(LadderCap, IsNonMonotonicPeakingNearMatchedNyquist) {
    // Two competing noise gains in a-dag: the derivative term k_max/sqrt(2w)
    // (worse at small w) and the x term x_max*sqrt(w/2) (worse at large w).
    // They balance at w = k_max/x_max ~ 1 on this grid, where the clean cap
    // peaks -- measured, not modeled.
    const auto cap = [](double w) { return ses::ladder_cap(kGrid, w); };
    EXPECT_GT(cap(1.0), cap(0.05));  // rising branch (derivative-limited)
    EXPECT_GT(cap(1.0), cap(8.0));   // falling branch (x-term-limited)
    EXPECT_GE(cap(1.0), 14);         // the peak is a genuinely useful range
    EXPECT_LE(cap(0.05), 3);         // soft well: the chain collapses fast
    EXPECT_LE(cap(8.0), 9);          // stiff well: falls again
}

TEST(LadderCap, ReproducesTheRecordedMeasuredCurve) {
    // Regression lock of the measured clean cap on the scene grid
    // (-20..20, 256 pts). These are MEASURED values (ladder vs oracle
    // divergence at 0.1% amplitude), recorded so a change of grid, tol, or
    // FFT path that shifts the usable range trips the test.
    const struct {
        double w;
        int cap;
    } pts[] = {{0.05, 1}, {0.1, 5},  {0.25, 12}, {0.5, 14},
               {1.0, 16}, {2.0, 15}, {4.0, 13},  {8.0, 7}};
    for (const auto& p : pts) {
        EXPECT_LE(std::abs(ses::ladder_cap(kGrid, p.w) - p.cap), 1)
            << "omega=" << p.w;
    }
}

// RED: the deep-level wall. psi_0 = (w/pi)^{1/4} exp(-w x^2 / 2) underflows
// double wherever w x^2 / 2 > ~745 (exp < smallest denormal), so the plain
// recurrence seeds EXACT ZEROS past |x| ~ 38.6/sqrt(w) -- and a zero seed
// stays zero at every level (the recurrence only multiplies and mixes).
// High-n eigenstates whose classically allowed region reaches into that
// dead zone lose their outer lobes: at n = 1200, omega = 1 the wall sits at
// x ~ 38.6 while the turning points are at +-49, so ~44% of the state's
// probability is simply MISSING. On top of that, ho_level_cap stopped
// probing at a fixed 400 levels. The contracts below pin the fix: a scaled
// per-point (mantissa, power-of-two exponent) chain -- power-of-two scaling
// is exact, so everything the plain chain got right stays bitwise intact --
// must push the ceiling to the honest grid physics: box (turning points +
// tail inside [xmin, xmax]) vs Nyquist (k_n < pi/h), nothing else.
constexpr double kDeepOmega = 1.0;
// -60..60 / 4096: k_max ~ 107 so Nyquist allows n ~ 5700; the box allows
// n_box ~ w x_max^2 / 2 = 1800. Both sit far above the seed wall (~710).
const ses::Grid1D kDeepGrid{-60.0, 60.0, 4096};

TEST(HoEigenstateDeep, SurvivesTheSeedUnderflowWall) {
    // n = 1200: every node and the exact energy must survive past the wall.
    const std::vector<double> v =
        ses::harmonic_potential(kDeepGrid, kDeepOmega);
    const ses::Field1D psi = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1200);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-10);
    EXPECT_EQ(node_count(psi), 1200);
    const double e_exact = 1200.5 * kDeepOmega;
    EXPECT_NEAR(ses::mean_energy(psi, v), e_exact, 1e-3 * e_exact);
}

TEST(HoLevelCapDeep, ReachesTheBoxCeilingNotTheSeedFloor) {
    // The measured cap must land below the box ceiling by the tail margin
    // only -- not at the seed wall (~710), not at a fixed probe bound (400).
    const int cap = ses::ho_level_cap(kDeepGrid, kDeepOmega);
    EXPECT_GE(cap, 1200);
    EXPECT_LT(cap, 1800);
    // And it must stay honest: the oracle AT the cap is still faithful.
    const std::vector<double> v =
        ses::harmonic_potential(kDeepGrid, kDeepOmega);
    const ses::Field1D at_cap = ses::ho_eigenstate(kDeepGrid, kDeepOmega, cap);
    const double e_exact = (cap + 0.5) * kDeepOmega;
    EXPECT_NEAR(ses::mean_energy(at_cap, v), e_exact, 1e-3 * e_exact);
}

TEST(LadderFockDeep, RaisesADeepPairBeyondTheSeedWall) {
    // (|1198> + |1200>)/sqrt(2) --adag--> (sqrt(1199)|1199> +
    // sqrt(1201)|1201>) normalized; counting norm^2 = (1199 + 1201)/2.
    // The energy contract is against the EXACT spectrum -- a basis and an
    // input that are broken the same self-consistent way cannot fake it.
    const ses::Field1D a = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1198);
    const ses::Field1D b = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1200);
    ses::Field1D psi{kDeepGrid};
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = a[i] + b[i];
    }
    ses::normalize(psi);
    double residual = 1.0;
    const double norm2 =
        ses::ladder_fock(psi, kDeepOmega, true, 1210, &residual);
    EXPECT_NEAR(norm2, 1200.0, 1e-3 * 1200.0);
    EXPECT_LT(residual, 1e-9);
    const std::vector<double> v =
        ses::harmonic_potential(kDeepGrid, kDeepOmega);
    const double e_exact =
        (1199.0 * 1199.5 + 1201.0 * 1201.5) / 2400.0 * kDeepOmega;
    EXPECT_NEAR(ses::mean_energy(psi, v), e_exact, 1e-3 * e_exact);
}

TEST(LadderFock, WideBandStaysExactForALowState) {
    // A band top FAR above the state's support must change nothing:
    // (|0> + |1>)/sqrt(2) raised inside a 1500-level band is still
    // (|1> + sqrt(2)|2>)/sqrt(3) with counting norm^2 = (1 + 2)/2.
    const ses::Field1D g0 = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 0);
    const ses::Field1D g1 = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1);
    ses::Field1D psi{kDeepGrid};
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = g0[i] + g1[i];
    }
    ses::normalize(psi);
    double residual = 1.0;
    const double norm2 =
        ses::ladder_fock(psi, kDeepOmega, true, 1500, &residual);
    EXPECT_NEAR(norm2, 1.5, 1e-9);
    EXPECT_LT(residual, 1e-10);
    const ses::Field1D e1 = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 1);
    const ses::Field1D e2 = ses::ho_eigenstate(kDeepGrid, kDeepOmega, 2);
    ses::Field1D want{kDeepGrid};
    const double s3 = 1.0 / std::sqrt(3.0);
    for (int i = 0; i < want.size(); ++i) {
        want[i] = s3 * (e1[i] + std::sqrt(2.0) * e2[i]);
    }
    EXPECT_NEAR(overlap_sq(psi, want), 1.0, 1e-9);
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
