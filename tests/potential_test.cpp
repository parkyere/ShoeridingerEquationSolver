// RED: specification for the potential builders (real-valued V on the grid).
//
//  - harmonic_potential(g, omega, x0):    V(x) = 1/2 omega^2 (x - x0)^2
//  - soft_coulomb_potential(g, Z, a, x0): V(x) = -Z / sqrt((x-x0)^2 + a^2)
//
// The soft Coulomb regularization is load-bearing: the bare -Z/|x| diverges
// on a grid point at the nucleus (docs/ARCHITECTURE.md); softened, the
// deepest value is exactly -Z/a at the center and finite everywhere.
// Oracles: exact values at grid points chosen to be exact in binary.

import ses.grid;
#include <core/potential.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using ses::Grid1D;

// Integer coordinates: [0,8) with n=8 -> x_i = 0,1,...,7 exactly.
const Grid1D kGrid{0.0, 8.0, 8};

TEST(AbsorbingMask, OneInInteriorTapersToZeroAtWalls) {
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1: coords -8 .. 7
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> m = ses::absorbing_mask(g, 3.0);
    ASSERT_EQ(m.size(), static_cast<std::size_t>(g.size()));
    // Deep interior (coord 0 on every axis) is exactly 1 -- the atom is safe.
    EXPECT_DOUBLE_EQ(m[static_cast<std::size_t>(g.flat(8, 8, 8))], 1.0);
    // The low corner (coords -8,-8,-8) sits on the wall: fully absorbed.
    EXPECT_NEAR(m[static_cast<std::size_t>(g.flat(0, 0, 0))], 0.0, 1e-12);
    // Every entry is a damping factor in [0, 1].
    for (double v : m) {
        EXPECT_GE(v, 0.0);
        EXPECT_LE(v, 1.0 + 1e-12);
    }
    // One step inside the low x-wall (coord -7): only the x-axis is in the
    // 3-wide layer (d = 1 -> sin^2(pi/6) = 0.25); y, z are interior (1).
    const double s = std::sin(0.5 * 3.14159265358979323846 * (1.0 / 3.0));
    EXPECT_NEAR(m[static_cast<std::size_t>(g.flat(1, 8, 8))], s * s, 1e-12);
}

TEST(BarrierPotential, SlabAlongXExactAndZeroElsewhere) {
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1: coords -8 .. 7
    const ses::Grid3D g{ax, ax, ax};
    // Rectangular slab: V = 0.25 for x in [0, 3), 0 elsewhere; y/z-free.
    const std::vector<double> v = ses::barrier_potential(g, 0.25, 0.0, 3.0);
    ASSERT_EQ(v.size(), static_cast<std::size_t>(g.size()));
    EXPECT_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))], 0.25);    // x = 0
    EXPECT_EQ(v[static_cast<std::size_t>(g.flat(10, 2, 14))], 0.25);  // x = 2, any y/z
    EXPECT_EQ(v[static_cast<std::size_t>(g.flat(7, 8, 8))], 0.0);     // x = -1
    EXPECT_EQ(v[static_cast<std::size_t>(g.flat(11, 8, 8))], 0.0);    // x = 3: half-open
}

TEST(HarmonicPotential, ExactValuesAndMinimum) {
    // omega = 2, x0 = 1:  V(x) = 2 (x-1)^2
    const std::vector<double> v = ses::harmonic_potential(kGrid, 2.0, 1.0);
    ASSERT_EQ(v.size(), 8u);
    EXPECT_EQ(v[1], 0.0);   // minimum at the center
    EXPECT_EQ(v[3], 8.0);   // 2 * (3-1)^2
    EXPECT_EQ(v[0], 2.0);   // 2 * (0-1)^2
}

TEST(HarmonicPotential, IsSymmetricAboutCenter) {
    // x0 = 4 on integer coords: V(4+d) == V(4-d) exactly.
    const std::vector<double> v = ses::harmonic_potential(kGrid, 1.0, 4.0);
    EXPECT_EQ(v[1], v[7]);
    EXPECT_EQ(v[2], v[6]);
    EXPECT_EQ(v[3], v[5]);
}

TEST(SoftCoulombPotential, ExactValuesAndFiniteAtNucleus) {
    // Z = 1, a = 1, nucleus at x0 = 2.
    const std::vector<double> v = ses::soft_coulomb_potential(kGrid, 1.0, 1.0, 2.0);
    EXPECT_DOUBLE_EQ(v[2], -1.0);                    // center: -Z/a, FINITE
    EXPECT_DOUBLE_EQ(v[4], -1.0 / std::sqrt(5.0));   // dx=2: -1/sqrt(4+1)
    EXPECT_DOUBLE_EQ(v[0], -1.0 / std::sqrt(5.0));   // symmetric partner
}

TEST(SoftCoulombPotential, DeepestAtNucleusAndAttractive) {
    const std::vector<double> v = ses::soft_coulomb_potential(kGrid, 2.0, 0.5, 3.0);
    for (std::size_t i = 0; i < v.size(); ++i) {
        EXPECT_LT(v[i], 0.0);        // attractive everywhere
        EXPECT_GE(v[i], v[3]);       // nowhere deeper than the nucleus
    }
    EXPECT_DOUBLE_EQ(v[3], -4.0);    // -Z/a = -2/0.5
}

// Regularized true Coulomb (3D): the bare -Z/|r-c| everywhere EXCEPT the single
// grid cell coincident with the nucleus, which -- where -Z/r would be -infinity
// -- takes the analytic average of -Z/r over its cubic cell, -Z*C/h with
// C = integral of 1/r over the unit cube. Unlike soft-Coulomb (which rounds the
// WHOLE well and shifts the spectrum up), this keeps every non-nucleus cell exact.
TEST(RegularizedCoulombPotential, ExactAwayFromNucleusAndFiniteAtIt) {
    const ses::Grid1D ax{-8.0, 8.0, 16};  // h = 1, coords -8..7, nucleus point at index 8
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 1.0, ses::Vec3d{});
    ASSERT_EQ(v.size(), static_cast<std::size_t>(g.size()));
    // Nucleus cell (coord 0,0,0): the analytic cell average -Z*C/h (h = 1), FINITE.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))],
                     -ses::kCoulombCellAverage);
    // A cell one step along x (r = 1): EXACT bare Coulomb -Z/r = -1.
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))], -1.0);
    // Body-diagonal neighbor (1,1,1): -Z/sqrt(3).
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 9, 9))], -1.0 / std::sqrt(3.0));
    // Attractive and finite everywhere; nucleus cell is the deepest.
    const double center = v[static_cast<std::size_t>(g.flat(8, 8, 8))];
    for (double x : v) {
        EXPECT_LT(x, 0.0);
        EXPECT_TRUE(std::isfinite(x));
        EXPECT_GE(x, center);
    }
}

TEST(RegularizedCoulombPotential, NucleusCellScalesWithChargeAndInverseSpacing) {
    // The cell average -Z*C/h scales linearly with Z and with 1/h.
    const ses::Grid1D ax{-4.0, 4.0, 16};  // h = 0.5, nucleus point at index 8
    const ses::Grid3D g{ax, ax, ax};
    const std::vector<double> v = ses::regularized_coulomb_potential(g, 2.0, ses::Vec3d{});
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(8, 8, 8))],
                     -2.0 * ses::kCoulombCellAverage / 0.5);
    // Away from the nucleus, still exact -Z/r (r = 0.5 one step along x).
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(g.flat(9, 8, 8))], -2.0 / 0.5);
}

}  // namespace
