// RED: the 3D isotropic harmonic trap is a CENTRAL potential, so the whole
// tracked-manifold pipeline (1D radial solve x constexpr tesseral E1 factors
// x Einstein A) must apply to it UNCHANGED -- this is the physics license
// for giving the trap scene the same measurement/decay machinery as the
// atom. The oracles are the HO ladder algebra (exact, not fitted):
//  - E_{k,l} = (2k + l + 3/2) w from the radial solver;
//  - r is a ladder operator: it connects N' = N +- 1 ONLY, so the radial
//    dipole integral for any |dN| != 1 pair vanishes;
//  - |<0s| z |1p_z>|^2 = 1/(2w)  =>  R^2 = 3/(2w), and the factorized
//    A(1p -> 0s) = (2/3) alpha^3 w^2;
//  - every N = 2 state decays at exactly 2 A10 (SU(3): the shell shares one
//    rate), pinning the radial x angular composition across l.

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

import ses.radial;
import ses.harmonics;
import ses.decay;

namespace {

using ses::RadialGrid;
using ses::RadialState;

constexpr double kOmega = 0.25;  // the trap scene's kTrapOmega

const RadialGrid& rg() {
    static const RadialGrid g{20.0, 3999};
    return g;
}

RadialState solve(int l, int k) {
    std::vector<double> vr(static_cast<std::size_t>(rg().n));
    for (int i = 0; i < rg().n; ++i) {
        const double r = rg().r(i);
        vr[static_cast<std::size_t>(i)] = 0.5 * kOmega * kOmega * r * r;
    }
    return ses::radial_eigenstate(
        rg(), ses::radial_hamiltonian(rg(), vr, l), k);
}

TEST(TrapLadder, RadialSolverReproducesTheEquallySpacedSpectrum) {
    // E = (2k + l + 3/2) w for every level the trap manifold tracks.
    struct Level {
        int l, k;
    };
    const Level levels[] = {{0, 0}, {1, 0}, {0, 1}, {2, 0}, {1, 1}, {3, 0}};
    for (const Level& lv : levels) {
        const double expect = (2 * lv.k + lv.l + 1.5) * kOmega;
        EXPECT_NEAR(solve(lv.l, lv.k).energy, expect, 1e-3)
            << "l=" << lv.l << " k=" << lv.k;
    }
}

TEST(TrapLadder, DipoleIsALadderOperatorAcrossN) {
    // Allowed: N=0 (0s) <-> N=1 (1p): R^2 = 3/(2w) (so that the z channel
    // carries the textbook 1/(2w)). Forbidden despite dl = 1: N=1 (1p) <->
    // N=0's partner across TWO rungs, e.g. 0s <-> 2p (N=3) -- the ladder
    // algebra zeroes it, and the FD solver must reproduce that.
    const RadialState s0 = solve(0, 0);   // N=0
    const RadialState p1 = solve(1, 0);   // N=1
    const RadialState p3 = solve(1, 1);   // N=3 (2k+l=3)
    const double r_allowed = ses::radial_dipole_integral(rg(), s0.u, p1.u);
    const double r_forbidden = ses::radial_dipole_integral(rg(), s0.u, p3.u);
    EXPECT_NEAR(r_allowed * r_allowed, 3.0 / (2.0 * kOmega), 0.01);
    EXPECT_LT(std::abs(r_forbidden), 1e-3 * std::abs(r_allowed));
}

TEST(TrapLadder, FactorizedEinsteinAReproducesTheTextbookLadderRate) {
    // A(1p_z -> 0s) = (4/3) alpha^3 w^3 * R^2 * |A_z|^2 with |A_z|^2 = 1/3
    // collapses to the closed form (2/3) alpha^3 w^2.
    const RadialState s0 = solve(0, 0);
    const RadialState p1 = solve(1, 0);
    const double r = ses::radial_dipole_integral(rg(), s0.u, p1.u);
    const double gap = p1.energy - s0.energy;
    const double a10 = ses::einstein_a(
        gap, r * r * ses::tesseral_e1_sq(0, 0, 1, 0));
    const double a3 = std::pow(ses::kFineStructureConstant, 3.0);
    EXPECT_NEAR(a10, (2.0 / 3.0) * a3 * kOmega * kOmega, 2e-3 * a10);
}

TEST(TrapLadder, WholeN2ShellDecaysAtTwiceTheLadderRate) {
    // SU(3): every N = 2 state (2s AND each 1d_m) carries total rate 2 A10 --
    // summed over all N = 1 destinations and polarizations. Pins the radial
    // x tesseral composition across different l with one exact number.
    const RadialState s0 = solve(0, 0);
    const RadialState p1 = solve(1, 0);
    const RadialState s2 = solve(0, 1);  // N=2, l=0
    const RadialState d2 = solve(2, 0);  // N=2, l=2
    const double a3 = std::pow(ses::kFineStructureConstant, 3.0);
    const double a10 = (2.0 / 3.0) * a3 * kOmega * kOmega;

    const double r_s2p = ses::radial_dipole_integral(rg(), p1.u, s2.u);
    const double gap_s2p = s2.energy - p1.energy;
    double total_2s = 0.0;
    for (int m = -1; m <= 1; ++m) {
        total_2s += ses::einstein_a(
            gap_s2p, r_s2p * r_s2p * ses::tesseral_e1_sq(1, m, 0, 0));
    }
    EXPECT_NEAR(total_2s, 2.0 * a10, 5e-3 * total_2s);

    const double r_d2p = ses::radial_dipole_integral(rg(), p1.u, d2.u);
    const double gap_d2p = d2.energy - p1.energy;
    for (int md : {0, 1, -2}) {  // spot-check three 1d orientations
        double total_d = 0.0;
        for (int m = -1; m <= 1; ++m) {
            total_d += ses::einstein_a(
                gap_d2p, r_d2p * r_d2p * ses::tesseral_e1_sq(1, m, 2, md));
        }
        EXPECT_NEAR(total_d, 2.0 * a10, 5e-3 * total_d) << "1d m=" << md;
    }
}

}  // namespace
