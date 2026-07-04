// RED: specification for the potential builders (real-valued V on the grid).
//
//  - harmonic_potential(g, omega, x0):    V(x) = 1/2 omega^2 (x - x0)^2
//  - soft_coulomb_potential(g, Z, a, x0): V(x) = -Z / sqrt((x-x0)^2 + a^2)
//
// The soft Coulomb regularization is load-bearing: the bare -Z/|x| diverges
// on a grid point at the nucleus (docs/ARCHITECTURE.md); softened, the
// deepest value is exactly -Z/a at the center and finite everywhere.
// Oracles: exact values at grid points chosen to be exact in binary.

#include <core/grid.hpp>
#include <core/potential.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using ses::Grid1D;

// Integer coordinates: [0,8) with n=8 -> x_i = 0,1,...,7 exactly.
const Grid1D kGrid{0.0, 8.0, 8};

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

}  // namespace
