// RED: multi-channel quantum jumps (transitions arc T5) -- ALL tracked
// orbitals get their lifetimes, not just one. Decay channels compete as
// independent Poisson processes: channel m fires at rate gamma_m * P_m with
// P_m = |<from_m|psi>|^2, the total escape probability over dt is
//     p = 1 - exp(-sum_m gamma_m P_m dt),
// and WHICH channel fires is distributed proportionally to its rate. On a
// jump the MCWF jump operator |to><from| collapses psi onto the channel's
// destination eigenstate.
//
// Randomness stays OUT of core: callers inject u1 (does a jump happen?) and
// u2 (which channel?), so stratified draws give EXACT statistics.
//
// Oracles:
//  - no-jump is a bitwise no-op and reports channel -1;
//  - p_total matches the closed form and the single-channel quantum_jump;
//  - exact stratified jump count for p = 1/4;
//  - channel selection counts match the exact strata arithmetic (rates
//    weighted by the CURRENT populations);
//  - a zero-rate (forbidden) channel is never selected, even when u2 lands
//    on its (empty) slot;
//  - all-forbidden channel lists never jump, even with u1 = 0.

#include <core/complex.hpp>
#include <core/decay.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using ses::Complex;
using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;

Grid3D cube(double lo, double hi, int n) {
    return Grid3D{Grid1D{lo, hi, n}, Grid1D{lo, hi, n}, Grid1D{lo, hi, n}};
}

// Orthonormal manifold on the centered grid: parity makes the overlaps
// vanish to machine precision -- even "ground", x-odd, y-odd "excited".
struct Manifold {
    Field3D ground;
    Field3D ex;
    Field3D ey;
};

Manifold make_manifold(const Grid3D& g) {
    Manifold m{Field3D{g}, Field3D{g}, Field3D{g}};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                const double env = std::exp(-0.5 * (x * x + y * y + z * z));
                m.ground(i, j, k) = Complex<double>{env, 0.0};
                m.ex(i, j, k) = Complex<double>{x * env, 0.0};
                m.ey(i, j, k) = Complex<double>{y * env, 0.0};
            }
        }
    }
    ses::normalize(m.ground);
    ses::normalize(m.ex);
    ses::normalize(m.ey);
    return m;
}

// psi = 0.6 |ex> + 0.8 |ey>  ->  P_x = 0.36, P_y = 0.64.
Field3D make_superposition(const Manifold& m) {
    Field3D psi{m.ground.grid()};
    for (std::size_t i = 0; i < psi.data().size(); ++i) {
        psi.data()[i] = 0.6 * m.ex.data()[i] + 0.8 * m.ey.data()[i];
    }
    return psi;
}

TEST(MultiQuantumJump, NoJumpIsBitwiseNoOpAndReportsClosedFormP) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const Manifold m = make_manifold(g);
    Field3D psi = make_superposition(m);
    const Field3D before = psi;

    const double g1 = 0.5;
    const double g2 = 0.25;
    const double dt = 0.7;
    const std::vector<ses::DecayChannel> channels{
        {&m.ex, &m.ground, g1},
        {&m.ey, &m.ground, g2},
    };
    const double p1 = ses::norm_sq(ses::inner_product(m.ex, psi));
    const double p2 = ses::norm_sq(ses::inner_product(m.ey, psi));
    const double expected_p = 1.0 - std::exp(-(g1 * p1 + g2 * p2) * dt);

    const ses::MultiJumpResult r = ses::multi_quantum_jump(psi, channels, dt, 0.999, 0.5);
    EXPECT_EQ(r.channel, -1);
    EXPECT_NEAR(r.p_total, expected_p, 1e-12);
    for (std::size_t i = 0; i < psi.data().size(); ++i) {
        EXPECT_EQ(psi.data()[i].real(), before.data()[i].real());
        EXPECT_EQ(psi.data()[i].imag(), before.data()[i].imag());
    }
}

TEST(MultiQuantumJump, SingleChannelMatchesQuantumJump) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const Manifold m = make_manifold(g);
    Field3D psi_multi = make_superposition(m);
    Field3D psi_single = psi_multi;

    const double gamma = 0.8;
    const double dt = 0.4;
    const ses::MultiJumpResult rm = ses::multi_quantum_jump(
        psi_multi, {{&m.ex, &m.ground, gamma}}, dt, 0.999, 0.0);
    const ses::JumpResult rs =
        ses::quantum_jump(psi_single, m.ex, m.ground, gamma, dt, 0.999);
    EXPECT_NEAR(rm.p_total, rs.p_jump, 1e-15);
}

TEST(MultiQuantumJump, JumpCollapsesOntoTheSelectedDestination) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const Manifold m = make_manifold(g);

    // u2 = 0 -> first channel; destination is ground, bitwise.
    {
        Field3D psi = make_superposition(m);
        const std::vector<ses::DecayChannel> channels{
            {&m.ex, &m.ground, 1.0},
            {&m.ey, &m.ex, 1.0},
        };
        const ses::MultiJumpResult r =
            ses::multi_quantum_jump(psi, channels, 5.0, 0.0, 0.0);
        EXPECT_EQ(r.channel, 0);
        for (std::size_t i = 0; i < psi.data().size(); ++i) {
            EXPECT_EQ(psi.data()[i].real(), m.ground.data()[i].real());
            EXPECT_EQ(psi.data()[i].imag(), m.ground.data()[i].imag());
        }
    }
    // u2 -> 1 lands in the last channel; destination ex, bitwise.
    {
        Field3D psi = make_superposition(m);
        const std::vector<ses::DecayChannel> channels{
            {&m.ex, &m.ground, 1.0},
            {&m.ey, &m.ex, 1.0},
        };
        const ses::MultiJumpResult r =
            ses::multi_quantum_jump(psi, channels, 5.0, 0.0, 0.999999);
        EXPECT_EQ(r.channel, 1);
        for (std::size_t i = 0; i < psi.data().size(); ++i) {
            EXPECT_EQ(psi.data()[i].real(), m.ex.data()[i].real());
            EXPECT_EQ(psi.data()[i].imag(), m.ex.data()[i].imag());
        }
    }
}

TEST(MultiQuantumJump, StratifiedJumpCountIsExact) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const Manifold m = make_manifold(g);

    // gamma * P_x * dt = -ln(3/4) with P_x = 1  ->  p = 1/4 exactly.
    const double gamma_dt = -std::log(0.75);
    const int kDraws = 1000;
    int jumps = 0;
    for (int k = 0; k < kDraws; ++k) {
        Field3D psi = m.ex;
        const double u1 = (k + 0.5) / kDraws;
        const ses::MultiJumpResult r = ses::multi_quantum_jump(
            psi, {{&m.ex, &m.ground, gamma_dt}}, 1.0, u1, 0.5);
        EXPECT_NEAR(r.p_total, 0.25, 1e-9);
        if (r.channel >= 0) {
            ++jumps;
        }
    }
    EXPECT_EQ(jumps, 250);
}

TEST(MultiQuantumJump, ChannelSelectionFollowsPopulationWeightedRates) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const Manifold m = make_manifold(g);

    // Rates: gamma_1 P_x : gamma_2 P_y = 2*0.36 : 1*0.64 = 0.72 : 0.64.
    const double g1 = 2.0;
    const double g2 = 1.0;
    const Field3D proto = make_superposition(m);
    const double p1 = ses::norm_sq(ses::inner_product(m.ex, proto));
    const double p2 = ses::norm_sq(ses::inner_product(m.ey, proto));
    const double frac1 = (g1 * p1) / (g1 * p1 + g2 * p2);

    const int kDraws = 1000;
    // Strata arithmetic: u2 = (k+0.5)/N picks channel 0 iff u2 < frac1.
    int expected0 = 0;
    for (int k = 0; k < kDraws; ++k) {
        if ((k + 0.5) / kDraws < frac1) {
            ++expected0;
        }
    }
    int chan0 = 0;
    int chan1 = 0;
    for (int k = 0; k < kDraws; ++k) {
        Field3D psi = proto;
        const double u2 = (k + 0.5) / kDraws;
        const ses::MultiJumpResult r = ses::multi_quantum_jump(
            psi,
            {{&m.ex, &m.ground, g1}, {&m.ey, &m.ground, g2}},
            50.0, 0.0, u2);  // u1 = 0: always jump
        ASSERT_GE(r.channel, 0);
        (r.channel == 0 ? chan0 : chan1) += 1;
    }
    EXPECT_EQ(chan0, expected0);
    EXPECT_EQ(chan1, kDraws - expected0);
}

TEST(MultiQuantumJump, ForbiddenChannelIsNeverSelected) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const Manifold m = make_manifold(g);
    Field3D psi = make_superposition(m);

    // Channel 0 is forbidden (gamma = 0): even u2 = 0 must fall through to
    // the allowed channel.
    const ses::MultiJumpResult r = ses::multi_quantum_jump(
        psi,
        {{&m.ex, &m.ground, 0.0}, {&m.ey, &m.ground, 1.0}},
        50.0, 0.0, 0.0);
    EXPECT_EQ(r.channel, 1);
    for (std::size_t i = 0; i < psi.data().size(); ++i) {
        EXPECT_EQ(psi.data()[i].real(), m.ground.data()[i].real());
        EXPECT_EQ(psi.data()[i].imag(), m.ground.data()[i].imag());
    }
}

TEST(MultiQuantumJump, AllForbiddenNeverJumps) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const Manifold m = make_manifold(g);
    Field3D psi = make_superposition(m);
    const Field3D before = psi;

    const ses::MultiJumpResult r = ses::multi_quantum_jump(
        psi,
        {{&m.ex, &m.ground, 0.0}, {&m.ey, &m.ground, 0.0}},
        50.0, 0.0, 0.5);  // u1 = 0 would jump if any rate were nonzero
    EXPECT_EQ(r.channel, -1);
    EXPECT_EQ(r.p_total, 0.0);
    for (std::size_t i = 0; i < psi.data().size(); ++i) {
        EXPECT_EQ(psi.data()[i].real(), before.data()[i].real());
        EXPECT_EQ(psi.data()[i].imag(), before.data()[i].imag());
    }
}

}  // namespace
