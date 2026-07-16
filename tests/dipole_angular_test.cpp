// RED: specification for the factorized E1 dipole channel table.
//
// Every tracked state is (u_nl(r)/r) Y_lm with a REAL (tesseral) Y_lm, so the
// 3D dipole integral factorizes exactly:
//     <f| r_q |i> = R_{n'l',nl} * A_q(l'm'; lm),
//     R = integral u_{n'l'}(r) r u_{nl}(r) dr   (ses::radial_dipole_integral)
// and the SQUARED angular factors |A_q|^2 are exact rationals -- computable
// at compile time (constexpr, integer arithmetic, no sqrt). The channel
// build then needs ~40 cheap 1D radial integrals instead of ~700 GPU
// synthesis+reduction passes.
//
// Contract for ses.harmonics:
//   tesseral_e1_axis_sq(axis, l_to, m_to, l_from, m_from) -> |A_q|^2
//   tesseral_e1_sq(l_to, m_to, l_from, m_from)            -> sum over q
// Oracles:
//   (1) hand values: every 2p -> 1s orientation carries angular 1/3;
//   (2) the sum rule Sum_{m',q} |A_q|^2 = max(l,l')/(2l+1) for all l <= 5
//       (ties the m-resolved table to the level-averaged einstein_a_level);
//   (3) exact selection zeros (the analytic dl/dm filter, now provable);
//   (4) numeric 3D factorization on an artificial radial shell (compact
//       support: pure angular check, no tail truncation) for l up to 5;
//   (5) end-to-end: factorized A(2p_z -> 1s) reproduces the textbook
//       Einstein A = 1.5155e-8 /au from our own radial solve.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

import ses.harmonics;
import ses.radial;
import ses.decay;
import ses.grid;
import ses.vec;
import ses.field;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::RadialGrid;

TEST(TesseralE1, TwoPToOneSCarriesOneThirdInEveryOrientation) {
    // z couples 2p_z, x couples 2p_x, y couples 2p_y -- each with the same
    // exact angular strength 1/3 (cubic symmetry made exact, not fp-lucky).
    EXPECT_DOUBLE_EQ(ses::tesseral_e1_axis_sq(2, 0, 0, 1, 0), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(ses::tesseral_e1_axis_sq(0, 0, 0, 1, 1), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(ses::tesseral_e1_axis_sq(1, 0, 0, 1, -1), 1.0 / 3.0);
    EXPECT_DOUBLE_EQ(ses::tesseral_e1_sq(0, 0, 1, 0), 1.0 / 3.0);
    // The wrong axes are EXACT zeros.
    EXPECT_EQ(ses::tesseral_e1_axis_sq(0, 0, 0, 1, 0), 0.0);
    EXPECT_EQ(ses::tesseral_e1_axis_sq(1, 0, 0, 1, 0), 0.0);
}

TEST(TesseralE1, SumRuleMatchesLevelAveragedFactor) {
    // Sum over the destination shell (all m') and polarizations of
    // |A_q|^2 = max(l, l')/(2l+1) -- the factor einstein_a_level uses. Holds
    // for every tracked (l, m) in both directions; pinned to 1e-14 (exact
    // rational arithmetic, only the summation rounds).
    for (int l = 0; l <= 5; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (const int lp : {l - 1, l + 1}) {
                if (lp < 0 || lp > 5) {
                    continue;
                }
                double sum = 0.0;
                for (int mp = -lp; mp <= lp; ++mp) {
                    sum += ses::tesseral_e1_sq(lp, mp, l, m);
                }
                const double expect =
                    static_cast<double>(std::max(l, lp)) / (2.0 * l + 1.0);
                EXPECT_NEAR(sum, expect, 1e-14) << "l=" << l << " m=" << m
                                                << " l'=" << lp;
            }
        }
    }
}

TEST(TesseralE1, SelectionRulesAreExactZeros) {
    for (int l = 0; l <= 5; ++l) {
        for (int m = -l; m <= l; ++m) {
            for (int lp = 0; lp <= 5; ++lp) {
                for (int mp = -lp; mp <= lp; ++mp) {
                    const bool dl_ok = std::abs(lp - l) == 1;
                    const bool dm_ok =
                        mp == m || std::abs(std::abs(mp) - std::abs(m)) == 1;
                    if (!dl_ok || !dm_ok) {
                        EXPECT_EQ(ses::tesseral_e1_sq(lp, mp, l, m), 0.0)
                            << "l=" << l << " m=" << m << " l'=" << lp
                            << " m'=" << mp;
                    }
                    // m' == -m (m != 0) is inside the |dm| rule yet exactly
                    // forbidden in the tesseral basis -- the subtle case the
                    // channel filter documents.
                    if (dl_ok && m != 0 && mp == -m) {
                        EXPECT_EQ(ses::tesseral_e1_sq(lp, mp, l, m), 0.0);
                    }
                }
            }
        }
    }
}

// Numeric 3D check of the factorization itself, decoupled from hydrogen
// tails: a compact radial shell u(r) = r exp(-2 (r-3)^2) fits ±8 with room,
// so the 3D integral differs from R * A only by angular grid error.
TEST(TesseralE1, FactorizationMatchesNumeric3DIntegral) {
    const Grid3D g{Grid1D{-8.0, 8.0, 64}, Grid1D{-8.0, 8.0, 64},
                   Grid1D{-8.0, 8.0, 64}};
    const RadialGrid rg{8.0, 1599};
    std::vector<double> u(static_cast<std::size_t>(rg.n));
    double n2 = 0.0;
    for (int i = 0; i < rg.n; ++i) {
        const double r = rg.r(i);
        u[static_cast<std::size_t>(i)] = r * std::exp(-2.0 * (r - 3.0) * (r - 3.0));
        n2 += u[static_cast<std::size_t>(i)] * u[static_cast<std::size_t>(i)] * rg.h();
    }
    for (double& v : u) {
        v /= std::sqrt(n2);
    }
    const double rint = ses::radial_dipole_integral(rg, u, u);

    struct Pair {
        int l_to, m_to, l_from, m_from;
    };
    const Pair pairs[] = {{0, 0, 1, 0},   {1, 1, 2, 2},  {2, 0, 3, 1},
                          {3, -2, 4, -3}, {4, 4, 5, 5},  {4, 0, 5, -1},
                          {2, -1, 3, 0}};
    for (const Pair& p : pairs) {
        const Field3D to = ses::synthesize_orbital(g, rg, u, p.l_to, p.m_to);
        const Field3D from =
            ses::synthesize_orbital(g, rg, u, p.l_from, p.m_from);
        for (int axis = 0; axis < 3; ++axis) {
            double num = 0.0;
            for (int k = 0; k < g.z.n; ++k) {
                for (int j = 0; j < g.y.n; ++j) {
                    for (int i = 0; i < g.x.n; ++i) {
                        const double q = axis == 0   ? g.x.coord(i)
                                         : axis == 1 ? g.y.coord(j)
                                                     : g.z.coord(k);
                        num += to(i, j, k).real() * q * from(i, j, k).real();
                    }
                }
            }
            num *= g.cell_volume();
            const double numeric_sq = num * num;
            const double predicted =
                rint * rint *
                ses::tesseral_e1_axis_sq(axis, p.l_to, p.m_to, p.l_from,
                                         p.m_from);
            if (predicted > 0.0) {
                EXPECT_NEAR(numeric_sq / predicted, 1.0, 0.02)
                    << "axis=" << axis << " pair " << p.l_from << ","
                    << p.m_from << " -> " << p.l_to << "," << p.m_to;
            } else {
                EXPECT_LT(numeric_sq, 1e-4 * rint * rint)
                    << "axis=" << axis << " pair " << p.l_from << ","
                    << p.m_from << " -> " << p.l_to << "," << p.m_to;
            }
        }
    }
}

TEST(TesseralE1, FactorizedEinsteinAReproducesTextbook2pLifetime) {
    // Bare -1/r radial solve (the app's table source): A(2p_z -> 1s) =
    // (4/3) alpha^3 omega^3 R^2 |A_z|^2 with omega = 3/8 Ha must land on the
    // textbook 1.5155e-8 /au (tau = 1.6 ns) -- the whole factorized pipeline
    // against literature, computed from OUR solver.
    const RadialGrid rg{20.0, 3999};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = -1.0 / rg.r(i);
    }
    const ses::RadialState s1 =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 0), 0);
    const ses::RadialState p2 =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 1), 0);
    const double rint = ses::radial_dipole_integral(rg, s1.u, p2.u);
    const double omega = p2.energy - s1.energy;
    const double a = ses::einstein_a(
        omega, rint * rint * ses::tesseral_e1_sq(0, 0, 1, 0));
    EXPECT_NEAR(a, 1.5155e-8, 2e-11);
}

}  // namespace
