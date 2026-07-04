// RED: WavepacketSimulation -- the tested orchestration layer between the
// physics core and the (untested) GL shell. Owns grid + potential +
// propagator + psi, advances real time, and hands the shell density frames.
// Keeping this in core means the shell stays logic-free.
//
// Also: marching_cubes_at_fraction -- isovalue as a fraction of the current
// density peak, so an animated (dispersing) cloud keeps a visible surface.

#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/marching_cubes.hpp>
#include <core/observables.hpp>
#include <core/potential.hpp>
#include <core/propagator.hpp>
#include <core/simulation.hpp>
#include <core/vec.hpp>
#include <core/wavepacket.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;
using ses::WavepacketSimulation;

Grid3D small_grid() {
    const Grid1D axis{-8.0, 8.0, 16};
    return Grid3D{axis, axis, axis};
}

WavepacketSimulation::Config demo_config() {
    const Grid3D g = small_grid();
    return WavepacketSimulation::Config{
        g,
        ses::harmonic_potential(g, 1.0, Vec3d{}),
        Vec3d{2.0, 0.0, 0.0},   // r0
        Vec3d{1.0, 1.0, 1.0},   // sigma
        Vec3d{0.0, 0.5, 0.0},   // k0
        0.02,                   // dt
    };
}

TEST(WavepacketSimulation, TimeAdvancesByStepsTimesDt) {
    WavepacketSimulation sim{demo_config()};
    EXPECT_EQ(sim.time(), 0.0);
    sim.advance(3);
    EXPECT_DOUBLE_EQ(sim.time(), 3 * 0.02);
    sim.advance(2);
    EXPECT_DOUBLE_EQ(sim.time(), 5 * 0.02);
}

TEST(WavepacketSimulation, ConservesNormWhileAdvancing) {
    WavepacketSimulation sim{demo_config()};
    sim.advance(50);
    EXPECT_NEAR(ses::norm_sq(sim.psi()), 1.0, 1e-12);
}

TEST(WavepacketSimulation, MatchesManualPropagation) {
    // The sim must be exactly "gaussian_wavepacket, then SplitOperator3D
    // steps" -- no hidden extras.
    const WavepacketSimulation::Config cfg = demo_config();
    WavepacketSimulation sim{cfg};
    sim.advance(5);

    Field3D manual = ses::gaussian_wavepacket(cfg.grid, cfg.r0, cfg.sigma, cfg.k0);
    const ses::SplitOperator3D prop{cfg.grid, cfg.potential, cfg.dt};
    prop.step(manual, 5);

    const Field3D& s = sim.psi();
    double max_diff = 0.0;
    for (std::size_t i = 0; i < s.data().size(); ++i) {
        max_diff = std::max(max_diff, std::abs(s.data()[i].re - manual.data()[i].re));
        max_diff = std::max(max_diff, std::abs(s.data()[i].im - manual.data()[i].im));
    }
    EXPECT_LT(max_diff, 1e-14);
}

TEST(WavepacketSimulation, DensityMatchesProbabilityDensity) {
    WavepacketSimulation sim{demo_config()};
    sim.advance(10);
    const std::vector<double> rho = sim.density();
    const std::vector<double> expected = ses::probability_density(sim.psi());
    ASSERT_EQ(rho.size(), expected.size());
    for (std::size_t i = 0; i < rho.size(); ++i) {
        EXPECT_EQ(rho[i], expected[i]);
    }
}

TEST(MarchingCubesAtFraction, EqualsExplicitIsovalue) {
    const Grid3D g = small_grid();
    WavepacketSimulation sim{demo_config()};
    const std::vector<double> rho = sim.density();

    double peak = 0.0;
    for (double v : rho) {
        peak = std::max(peak, v);
    }
    const ses::Mesh a = ses::marching_cubes_at_fraction(rho, g, 0.25);
    const ses::Mesh b = ses::marching_cubes(rho, g, 0.25 * peak);
    ASSERT_EQ(a.vertices.size(), b.vertices.size());
    ASSERT_GT(a.vertices.size(), 0u);
    for (std::size_t i = 0; i < a.vertices.size(); ++i) {
        EXPECT_EQ(a.vertices[i].x, b.vertices[i].x);
        EXPECT_EQ(a.vertices[i].y, b.vertices[i].y);
        EXPECT_EQ(a.vertices[i].z, b.vertices[i].z);
    }
}

}  // namespace
