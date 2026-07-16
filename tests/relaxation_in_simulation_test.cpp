// RED: imaginary-time relaxation inside WavepacketSimulation, so the shell
// can show the cloud settling into the ground state live and then hand the
// relaxed state straight back to real-time evolution.
//
// Oracles:
//  - relax() lowers <H> monotonically toward the known ground-state energy
//    (3D harmonic: E0 = 3 omega / 2), keeps unit norm, and does NOT advance
//    real time;
//  - the punchline of the whole feature: a relaxed state is STATIONARY --
//    subsequent real-time evolution must leave the density where it is
//    (only the invisible global phase turns).

#include <core/field.hpp>
import ses.grid;
#include <core/observables.hpp>
#include <core/simulation.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.potential;

namespace {

using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;
using ses::WavepacketSimulation;

WavepacketSimulation::Config harmonic_config() {
    const Grid1D axis{-8.0, 8.0, 16};
    const Grid3D g{axis, axis, axis};
    return WavepacketSimulation::Config{
        g,
        ses::harmonic_potential(g, 1.0, Vec3d{}),
        Vec3d{1.5, 0.0, 0.0},  // displaced on purpose
        Vec3d{1.0, 1.0, 1.0},
        Vec3d{},
        0.02,
    };
}

TEST(SimulationRelax, PotentialAccessorReturnsConfigPotential) {
    const WavepacketSimulation::Config cfg = harmonic_config();
    const WavepacketSimulation sim{cfg};
    const std::vector<double>& v = sim.potential();
    ASSERT_EQ(v.size(), cfg.potential.size());
    EXPECT_EQ(v[0], cfg.potential[0]);
    EXPECT_EQ(v[v.size() / 2], cfg.potential[v.size() / 2]);
}

TEST(SimulationRelax, KeepsUnitNormAndFreezesRealTime) {
    WavepacketSimulation sim{harmonic_config()};
    sim.advance(5);
    const double t_before = sim.time();
    sim.relax(100, 0.05);
    EXPECT_EQ(sim.time(), t_before);  // imaginary time is not real time
    EXPECT_NEAR(ses::norm_sq(sim.psi()), 1.0, 1e-12);
}

TEST(SimulationRelax, LowersEnergyTowardGroundState) {
    WavepacketSimulation sim{harmonic_config()};
    const double e0 = ses::mean_energy(sim.psi(), sim.potential());
    sim.relax(100, 0.05);
    const double e1 = ses::mean_energy(sim.psi(), sim.potential());
    EXPECT_LT(e1, e0);
    sim.relax(300, 0.05);  // tau = 20 total
    const double e2 = ses::mean_energy(sim.psi(), sim.potential());
    EXPECT_LE(e2, e1 + 1e-12);
    EXPECT_NEAR(e2, 1.5, 0.02);  // 3D harmonic: E0 = 3 omega / 2
}

TEST(SimulationRelax, RelaxedStateIsStationaryUnderRealTime) {
    WavepacketSimulation sim{harmonic_config()};
    sim.relax(400, 0.05);
    const Vec3d r_before = ses::mean_position(sim.psi());
    const Vec3d s_before = ses::sigma_position(sim.psi());

    sim.advance(50);  // t = 1: a non-stationary packet would swing ~1 Bohr

    const Vec3d r_after = ses::mean_position(sim.psi());
    const Vec3d s_after = ses::sigma_position(sim.psi());
    EXPECT_NEAR(r_after.x, r_before.x, 0.01);
    EXPECT_NEAR(r_after.y, r_before.y, 0.01);
    EXPECT_NEAR(r_after.z, r_before.z, 0.01);
    EXPECT_NEAR(s_after.x, s_before.x, 0.01);
    EXPECT_NEAR(s_after.y, s_before.y, 0.01);
    EXPECT_NEAR(s_after.z, s_before.z, 0.01);
    EXPECT_NEAR(ses::norm_sq(sim.psi()), 1.0, 1e-12);
}

}  // namespace
