// RED: state injection and propagator access on the simulation session --
// the two hooks the GPU/CPU sync choreography needs:
// set_psi() lets the shell hand a GPU-evolved state back to the CPU session,
// and propagator() exposes the tested phase tables without rebuilding them.

#include <core/field.hpp>
import ses.grid;
#include <core/observables.hpp>
#include <core/propagator.hpp>
#include <core/simulation.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.wavepacket;
import ses.potential;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;
using ses::WavepacketSimulation;

WavepacketSimulation::Config config() {
    const Grid1D axis{-8.0, 8.0, 16};
    const Grid3D g{axis, axis, axis};
    return WavepacketSimulation::Config{
        g,
        ses::harmonic_potential(g, 1.0, Vec3d{}),
        Vec3d{1.0, 0.0, 0.0},
        Vec3d{1.0, 1.0, 1.0},
        Vec3d{},
        0.02,
    };
}

TEST(SimulationState, SetPsiReplacesTheState) {
    const WavepacketSimulation::Config cfg = config();
    WavepacketSimulation sim{cfg};

    const Field3D replacement =
        ses::gaussian_wavepacket(cfg.grid, Vec3d{-2.0, 1.0, 0.5}, Vec3d{0.8, 0.9, 1.1},
                                 Vec3d{0.3, 0.0, 0.0});
    sim.set_psi(replacement);

    const Vec3d r = ses::mean_position(sim.psi());
    EXPECT_NEAR(r.x, -2.0, 1e-6);
    EXPECT_NEAR(r.y, 1.0, 1e-6);
    EXPECT_NEAR(r.z, 0.5, 1e-6);
}

TEST(SimulationState, AdvanceEvolvesTheInjectedState) {
    const WavepacketSimulation::Config cfg = config();
    WavepacketSimulation sim{cfg};
    const Field3D replacement =
        ses::gaussian_wavepacket(cfg.grid, Vec3d{-1.0, 0.0, 0.0}, Vec3d{1.0, 1.0, 1.0},
                                 Vec3d{});
    sim.set_psi(replacement);
    sim.advance(5);

    Field3D manual = replacement;
    const ses::SplitOperator3D prop{cfg.grid, cfg.potential, cfg.dt};
    prop.step(manual, 5);

    double max_diff = 0.0;
    for (std::size_t i = 0; i < manual.data().size(); ++i) {
        max_diff = std::max(max_diff,
                            std::abs(sim.psi().data()[i].real() - manual.data()[i].real()));
        max_diff = std::max(max_diff,
                            std::abs(sim.psi().data()[i].imag() - manual.data()[i].imag()));
    }
    EXPECT_LT(max_diff, 1e-14);
}

TEST(SimulationState, PropagatorExposesTheTables) {
    const WavepacketSimulation::Config cfg = config();
    const WavepacketSimulation sim{cfg};
    const ses::SplitOperator3D reference{cfg.grid, cfg.potential, cfg.dt};

    const auto& half = sim.propagator().half_potential_phase();
    const auto& ref_half = reference.half_potential_phase();
    ASSERT_EQ(half.size(), ref_half.size());
    EXPECT_EQ(half[17].real(), ref_half[17].real());
    EXPECT_EQ(half[17].imag(), ref_half[17].imag());

    const auto& kin = sim.propagator().kinetic_phase();
    const auto& ref_kin = reference.kinetic_phase();
    ASSERT_EQ(kin.size(), ref_kin.size());
    EXPECT_EQ(kin[123].real(), ref_kin[123].real());
    EXPECT_EQ(kin[123].imag(), ref_kin[123].imag());
}

}  // namespace
