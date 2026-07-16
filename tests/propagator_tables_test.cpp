// RED: read access to the split-operator phase tables, so the GPU engine
// consumes the TESTED tables instead of re-deriving them,
// plus dt access on the simulation session for the same reason.
//
// Oracles: table entries equal the defining formulas evaluated through the
// same public building blocks (wavenumbers, potential), bitwise.

import ses.grid;
#include <core/propagator.hpp>
#include <core/simulation.hpp>
import ses.spectral;
import ses.vec;

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <vector>
import ses.potential;

namespace {

using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

Grid3D cube4() {
    const Grid1D axis{0.0, 4.0, 4};
    return Grid3D{axis, axis, axis};
}

TEST(PropagatorTables, HalfPotentialPhaseMatchesFormula) {
    const Grid3D g = cube4();
    const std::vector<double> v = ses::harmonic_potential(g, 2.0, Vec3d{1.0, 2.0, 3.0});
    const double dt = 0.1;
    const ses::SplitOperator3D prop{g, v, dt};

    const auto& half = prop.half_potential_phase();
    ASSERT_EQ(half.size(), v.size());
    for (const std::size_t idx : {std::size_t{0}, std::size_t{17}, v.size() - 1}) {
        const double th = -0.5 * v[idx] * dt;
        EXPECT_DOUBLE_EQ(half[idx].real(), std::cos(th));
        EXPECT_DOUBLE_EQ(half[idx].imag(), std::sin(th));
    }
}

TEST(PropagatorTables, KineticPhaseMatchesFormula) {
    const Grid3D g = cube4();
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double dt = 0.1;
    const ses::SplitOperator3D prop{g, v, dt};

    const std::vector<double> kx = ses::wavenumbers(g.x);
    const std::vector<double> ky = ses::wavenumbers(g.y);
    const std::vector<double> kz = ses::wavenumbers(g.z);

    const auto& kin = prop.kinetic_phase();
    ASSERT_EQ(kin.size(), static_cast<std::size_t>(g.size()));
    for (const auto [i, j, k] : {std::array<int, 3>{1, 0, 0}, std::array<int, 3>{3, 2, 1},
                                 std::array<int, 3>{0, 0, 2}}) {
        const double kxx = kx[static_cast<std::size_t>(i)];
        const double kyy = ky[static_cast<std::size_t>(j)];
        const double kzz = kz[static_cast<std::size_t>(k)];
        const double th = -0.5 * (kxx * kxx + kyy * kyy + kzz * kzz) * dt;
        const std::size_t idx = static_cast<std::size_t>(g.flat(i, j, k));
        EXPECT_DOUBLE_EQ(kin[idx].real(), std::cos(th));
        EXPECT_DOUBLE_EQ(kin[idx].imag(), std::sin(th));
    }
}

TEST(SimulationAccessors, ExposesDt) {
    const Grid3D g = cube4();
    const ses::WavepacketSimulation sim{ses::WavepacketSimulation::Config{
        g,
        std::vector<double>(static_cast<std::size_t>(g.size()), 0.0),
        Vec3d{},
        Vec3d{1.0, 1.0, 1.0},
        Vec3d{},
        0.025,
    }};
    EXPECT_EQ(sim.dt(), 0.025);
}

}  // namespace
