// RED: read access to the imaginary-time weight tables, so the GPU relax
// path consumes the TESTED tables instead of
// re-deriving them -- the same pattern as the real-time phase tables.
//
// Oracles: entries equal the defining formulas evaluated through the same
// public building blocks (wavenumbers, potential), bitwise.

import ses.grid;
#include <core/imaginary_time.hpp>
#include <core/potential.hpp>
#include <core/spectral.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>

namespace {

using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

Grid3D cube4() {
    const Grid1D axis{0.0, 4.0, 4};
    return Grid3D{axis, axis, axis};
}

TEST(ImaginaryTables, HalfPotentialWeightMatchesFormula) {
    const Grid3D g = cube4();
    const std::vector<double> v = ses::harmonic_potential(g, 2.0, Vec3d{1.0, 2.0, 3.0});
    const double dtau = 0.05;
    const ses::ImaginaryTimePropagator3D relaxer{g, v, dtau};

    const std::vector<double>& half = relaxer.half_potential_weight();
    ASSERT_EQ(half.size(), v.size());
    for (const std::size_t idx : {std::size_t{0}, std::size_t{17}, v.size() - 1}) {
        EXPECT_DOUBLE_EQ(half[idx], std::exp(-0.5 * v[idx] * dtau));
    }
}

TEST(ImaginaryTables, KineticWeightMatchesFormula) {
    const Grid3D g = cube4();
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double dtau = 0.05;
    const ses::ImaginaryTimePropagator3D relaxer{g, v, dtau};

    const std::vector<double> kx = ses::wavenumbers(g.x);
    const std::vector<double> ky = ses::wavenumbers(g.y);
    const std::vector<double> kz = ses::wavenumbers(g.z);

    const std::vector<double>& kin = relaxer.kinetic_weight();
    ASSERT_EQ(kin.size(), static_cast<std::size_t>(g.size()));
    for (const auto [i, j, k] : {std::array<int, 3>{1, 0, 0}, std::array<int, 3>{3, 2, 1},
                                 std::array<int, 3>{0, 0, 2}}) {
        const double kxx = kx[static_cast<std::size_t>(i)];
        const double kyy = ky[static_cast<std::size_t>(j)];
        const double kzz = kz[static_cast<std::size_t>(k)];
        const std::size_t idx = static_cast<std::size_t>(g.flat(i, j, k));
        EXPECT_DOUBLE_EQ(kin[idx],
                         std::exp(-0.5 * (kxx * kxx + kyy * kyy + kzz * kzz) * dtau));
    }
}

}  // namespace
