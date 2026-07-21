// RED: exact 4x4 Heisenberg contracts.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>

import ses.spinexact;

namespace {

ses::SpinLattice product_updown(int flip_site) {
    ses::SpinLattice l;
    l.nx = ses::kExactSide;
    l.ny = ses::kExactSide;
    l.s.assign(ses::kExactSites, ses::spinor_from_bloch(0.0, 0.0, 1.0));
    if (flip_site >= 0) {
        l.s[static_cast<std::size_t>(flip_site)] =
            ses::spinor_from_bloch(0.0, 0.0, -1.0);
    }
    return l;
}

double norm2(const ses::SpinState16& s) {
    double n = 0.0;
    for (const auto& z : s.c) {
        n += std::norm(z);
    }
    return n;
}

double total_sz(const ses::SpinState16& s) {
    double t = 0.0;
    for (int i = 0; i < ses::kExactSites; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ses::exact_site_bloch(s, i, &x, &y, &z);
        t += z;
    }
    return t;
}

ses::SpinState16 seed_tilted() {
    ses::SpinLattice l;
    l.nx = ses::kExactSide;
    l.ny = ses::kExactSide;
    l.s.resize(ses::kExactSites);
    for (int y = 0; y < ses::kExactSide; ++y) {
        for (int x = 0; x < ses::kExactSide; ++x) {
            const double sgn = ((x + y) & 1) != 0 ? -1.0 : 1.0;
            l.s[static_cast<std::size_t>(y * ses::kExactSide + x)] =
                ses::spinor_from_bloch(0.6, 0.0, sgn * 0.8);
        }
    }
    return ses::exact_from_product(l);
}

double max_amp_diff(const ses::SpinState16& a, const ses::SpinState16& b) {
    double m = 0.0;
    for (std::size_t i = 0; i < a.c.size(); ++i) {
        const double d = std::abs(a.c[i] - b.c[i]);
        if (d > m) {
            m = d;
        }
    }
    return m;
}

// Fusion Stage 1: apply_fused (gather/matmul/scatter) must reproduce the
// hand-written gates, and the fused-gate step must reproduce exact_step.
TEST(SpinFuse, SiteGateMatchesExactRotate) {
    ses::SpinState16 a = seed_tilted();
    ses::SpinState16 b = a;
    ses::exact_site_rotate(a, 5, 0.3, -0.4, 0.85, 0.21);
    ses::apply_fused(b, ses::site_fused_gate(5, 0.3, -0.4, 0.85, 0.21));
    EXPECT_LT(max_amp_diff(a, b), 1e-12);
}

TEST(SpinFuse, BondGateMatchesExactBond) {
    ses::SpinState16 a = seed_tilted();
    ses::SpinState16 b = a;
    ses::exact_bond_gate(a, 6, 7, 0.17);
    ses::apply_fused(b, ses::bond_fused_gate(6, 7, 0.17));
    EXPECT_LT(max_amp_diff(a, b), 1e-12);
}

TEST(SpinFuse, FusedStepMatchesExactStep) {
    const double bx = 0.1, by = -0.05, bz = 0.2, jj = 0.5, dt = 0.05;
    ses::SpinState16 a = seed_tilted();
    ses::SpinState16 b = a;
    const std::vector<ses::FusedGate> gs =
        ses::step_gates(bx, by, bz, jj, dt);
    for (int k = 0; k < 20; ++k) {
        ses::exact_step(a, bx, by, bz, jj, dt);
        ses::fused_step(b, gs);
    }
    EXPECT_LT(max_amp_diff(a, b), 1e-10);
}

TEST(SpinExact, ProductBootRoundTripsTheArrows) {
    ses::SpinLattice l;
    l.nx = ses::kExactSide;
    l.ny = ses::kExactSide;
    l.s.resize(ses::kExactSites);
    for (int i = 0; i < ses::kExactSites; ++i) {
        const double th = 0.3 + 0.11 * i;
        const double ph = 0.7 * i;
        l.s[static_cast<std::size_t>(i)] = ses::spinor_from_bloch(
            std::sin(th) * std::cos(ph), std::sin(th) * std::sin(ph),
            std::cos(th));
    }
    const ses::SpinState16 s = ses::exact_from_product(l);
    ASSERT_EQ(s.c.size(), ses::kExactDim);
    EXPECT_NEAR(norm2(s), 1.0, 1e-12);
    for (int i = 0; i < ses::kExactSites; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ses::exact_site_bloch(s, i, &x, &y, &z);
        double lx = 0.0;
        double ly = 0.0;
        double lz = 0.0;
        ses::bloch_vector(l.s[static_cast<std::size_t>(i)], &lx, &ly,
                          &lz);
        EXPECT_NEAR(x, lx, 1e-12);
        EXPECT_NEAR(y, ly, 1e-12);
        EXPECT_NEAR(z, lz, 1e-12);
    }
}

TEST(SpinExact, FerroIsStationaryAndMagnonHopsConservingSz) {
    ses::SpinState16 up = ses::exact_from_product(product_updown(-1));
    for (int k = 0; k < 200; ++k) {
        ses::exact_step(up, 0.0, 0.0, 0.4, 0.5, 0.01);
    }
    for (int i = 0; i < ses::kExactSites; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ses::exact_site_bloch(up, i, &x, &y, &z);
        EXPECT_NEAR(z, 1.0, 1e-9);
    }

    ses::SpinState16 mg = ses::exact_from_product(product_updown(5));
    const double sz0 = total_sz(mg);
    double z5_min = 1.0;
    double z_other_min = 1.0;
    for (int k = 0; k < 400; ++k) {
        ses::exact_step(mg, 0.0, 0.0, 0.0, 0.5, 0.01);
    }
    EXPECT_NEAR(total_sz(mg), sz0, 1e-9);
    double x = 0.0;
    double y = 0.0;
    double z5 = 0.0;
    ses::exact_site_bloch(mg, 5, &x, &y, &z5);
    EXPECT_GT(z5, -0.9);
    double zmin = 1.0;
    for (int i = 0; i < ses::kExactSites; ++i) {
        if (i == 5) {
            continue;
        }
        double zz = 0.0;
        ses::exact_site_bloch(mg, i, &x, &y, &zz);
        zmin = std::min(zmin, zz);
    }
    EXPECT_LT(zmin, 0.99);
    (void)z5_min;
    (void)z_other_min;
}

TEST(SpinExact, NeelEntanglesArrowsShrinkEnergyConserved) {
    ses::SpinLattice l;
    l.nx = ses::kExactSide;
    l.ny = ses::kExactSide;
    l.s.resize(ses::kExactSites);
    for (int yy = 0; yy < ses::kExactSide; ++yy) {
        for (int xx = 0; xx < ses::kExactSide; ++xx) {
            const double sgn = ((xx + yy) & 1) != 0 ? -1.0 : 1.0;
            l.s[static_cast<std::size_t>(yy * ses::kExactSide + xx)] =
                ses::spinor_from_bloch(0.0, 0.0, sgn);
        }
    }
    ses::SpinState16 s = ses::exact_from_product(l);
    const double e0 = ses::exact_energy(s, 0.0, 0.0, 0.1, 0.5);
    ASSERT_NE(e0, 0.0);
    for (int k = 0; k < 500; ++k) {
        ses::exact_step(s, 0.0, 0.0, 0.1, 0.5, 0.01);
    }
    EXPECT_NEAR(norm2(s), 1.0, 1e-10);
    EXPECT_NEAR(ses::exact_energy(s, 0.0, 0.0, 0.1, 0.5), e0,
                1e-3 * std::abs(e0) + 1e-3);
    double mean_len = 0.0;
    for (int i = 0; i < ses::kExactSites; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        ses::exact_site_bloch(s, i, &x, &y, &z);
        mean_len += std::sqrt(x * x + y * y + z * z);
    }
    mean_len /= ses::kExactSites;
    EXPECT_LT(mean_len, 0.8);
}

TEST(SpinExact, MeasurementCollapsesASite) {
    ses::SpinLattice l = product_updown(-1);
    l.s[3] = ses::spinor_from_bloch(1.0, 0.0, 0.0);
    ses::SpinState16 s = ses::exact_from_product(l);
    const int out = ses::exact_measure_z(s, 3, 0.4);
    EXPECT_NE(out, 0);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    ses::exact_site_bloch(s, 3, &x, &y, &z);
    EXPECT_NEAR(z, static_cast<double>(out), 1e-12);
    EXPECT_NEAR(norm2(s), 1.0, 1e-12);
    ses::exact_site_rotate(s, 3, 0.0, 1.0, 0.0,
                           3.14159265358979323846);
    ses::exact_site_bloch(s, 3, &x, &y, &z);
    EXPECT_NEAR(z, -static_cast<double>(out), 1e-12);
}

}  // namespace
