// RED: specification for the split-operator (Fourier) TDSE propagator.
//
// One step of exp(-i H dt), H = -1/2 d^2/dx^2 + V(x), via Strang splitting:
//     psi <- e^{-i V dt/2} . IFFT . e^{-i k^2 dt / 2} . FFT . e^{-i V dt/2} psi
// (atomic units; docs/ARCHITECTURE.md).
//
// Free particle (V = 0) is the perfect first oracle: the kinetic factor is
// applied EXACTLY in k-space, so there is no time-splitting error at all and
// the analytic Gaussian dispersion must be reproduced to spectral accuracy:
//     <x>(t)    = x0 + k0 t
//     sigma(t)  = sigma0 sqrt(1 + (t / (2 sigma0^2))^2)
//     <p>(t)    = k0
//     ||psi||   = 1
// The potential path (e^{-i V dt/2}) is pinned by the harmonic-oscillator
// coherent-state spec (harmonic_dynamics_test).


#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>
import ses.propagator;
import ses.observables;
import ses.grid;
import ses.field;
import ses.wavepacket;
import ses.potential;
import ses.imaginary_time;

namespace {

using ses::Field1D;
using ses::Grid1D;
using ses::SplitOperator1D;

// Wide box so the dispersed packet never touches the periodic walls.
const Grid1D kGrid{-40.0, 40.0, 1024};

constexpr double kX0 = -15.0;
constexpr double kSigma0 = 1.0;
constexpr double kK0 = 2.0;
constexpr double kDt = 0.05;
constexpr int kSteps = 200;  // T = 10

std::vector<double> free_potential() {
    return std::vector<double>(static_cast<std::size_t>(kGrid.n), 0.0);
}

Field1D propagated_packet() {
    Field1D psi = ses::gaussian_wavepacket(kGrid, kX0, kSigma0, kK0);
    SplitOperator1D prop{kGrid, free_potential(), kDt};
    prop.step(psi, kSteps);
    return psi;
}

TEST(SplitOperator, ZeroStepsIsIdentity) {
    Field1D psi = ses::gaussian_wavepacket(kGrid, kX0, kSigma0, kK0);
    const double x_before = ses::mean_position(psi);
    SplitOperator1D prop{kGrid, free_potential(), kDt};
    prop.step(psi, 0);
    EXPECT_DOUBLE_EQ(ses::mean_position(psi), x_before);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(SplitOperator, ConservesNorm) {
    const Field1D psi = propagated_packet();
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(SplitOperator, CenterMovesAtGroupVelocity) {
    const Field1D psi = propagated_packet();
    const double t = kDt * kSteps;
    EXPECT_NEAR(ses::mean_position(psi), kX0 + kK0 * t, 1e-6);
}

TEST(SplitOperator, PacketDispersesAnalytically) {
    const Field1D psi = propagated_packet();
    const double t = kDt * kSteps;
    const double ratio = t / (2.0 * kSigma0 * kSigma0);
    const double expected = kSigma0 * std::sqrt(1.0 + ratio * ratio);
    EXPECT_NEAR(ses::sigma_x(psi), expected, 1e-6);
}

TEST(SplitOperator, ConservesMeanMomentum) {
    const Field1D psi = propagated_packet();
    EXPECT_NEAR(ses::mean_momentum(psi), kK0, 1e-8);
}

}  // namespace

// RED: the mass parameter -- the Cu(111) surface-state electron (corral
// scene) propagates with m* != 1, so the split-operator pair and the energy
// readout take a trailing mass (default 1 = every existing caller, bitwise).
//   - kinetic phase: exp(-i k^2/(2 m) dt)  ->  free packet drifts at k0/m
//   - ITP weight:    exp(-k^2/(2 m) dtau)  ->  harmonic ground E0 = w/(2 sqrt(m))
//   - mean_energy(psi, v, m): kinetic term <k^2>/(2 m)
TEST(MassParameter, FreePacketDriftsAtK0OverM) {
    const ses::Grid3D g{ses::Grid1D{-32.0, 32.0, 128}, ses::Grid1D{-1.0, 1.0, 1},
                        ses::Grid1D{-1.0, 1.0, 1}};
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double k0 = 1.0;
    const double mass = 2.0;
    const double t_total = 4.0;
    const int nsteps = 200;
    ses::Field3D psi = ses::gaussian_wavepacket(
        g, ses::Vec3d{-8.0, 0.0, 0.0}, ses::Vec3d{2.0, 0.5, 0.5},
        ses::Vec3d{k0, 0.0, 0.0});
    const ses::SplitOperator3D prop{g, v, t_total / nsteps, mass};
    prop.step(psi, nsteps);
    // Half the mass-1 group velocity: <x> = -8 + (k0/m) t = -6.
    EXPECT_NEAR(ses::mean_position(psi).x, -8.0 + (k0 / mass) * t_total, 0.05);
}

TEST(MassParameter, DefaultMassIsBitwiseTheLegacyTables) {
    const ses::Grid3D g{ses::Grid1D{-8.0, 8.0, 16}, ses::Grid1D{-8.0, 8.0, 16},
                        ses::Grid1D{-1.0, 1.0, 1}};
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for (int i = 0; i < g.size(); ++i) {
        v[static_cast<std::size_t>(i)] = 0.01 * i;
    }
    const ses::SplitOperator3D legacy{g, v, 0.05};
    const ses::SplitOperator3D unit{g, v, 0.05, 1.0};
    for (std::size_t i = 0; i < legacy.kinetic_phase().size(); ++i) {
        ASSERT_EQ(legacy.kinetic_phase()[i], unit.kinetic_phase()[i]);
    }
    const ses::ImaginaryTimePropagator3D ilegacy{g, v, 0.05};
    const ses::ImaginaryTimePropagator3D iunit{g, v, 0.05, 1.0};
    for (std::size_t i = 0; i < ilegacy.kinetic_weight().size(); ++i) {
        ASSERT_EQ(ilegacy.kinetic_weight()[i], iunit.kinetic_weight()[i]);
    }
}

TEST(MassParameter, HarmonicGroundUnderMassRelaxesToOmegaOverTwoRootM) {
    // Flat axes pinned AT the origin (coord(0) = xmin), so the isotropic
    // well contributes no constant y^2/z^2 offset.
    const ses::Grid3D g{ses::Grid1D{-12.0, 12.0, 64}, ses::Grid1D{0.0, 2.0, 1},
                        ses::Grid1D{0.0, 2.0, 1}};
    const double w = 1.0;
    const double mass = 4.0;  // Omega = w/sqrt(m) = 0.5 -> E0 = 0.25
    const std::vector<double> v =
        ses::harmonic_potential(g, w, ses::Vec3d{0.0, 0.0, 0.0});
    ses::Field3D psi = ses::gaussian_wavepacket(
        g, ses::Vec3d{1.0, 0.0, 0.0}, ses::Vec3d{2.0, 0.5, 0.5},
        ses::Vec3d{0.0, 0.0, 0.0});
    const ses::ImaginaryTimePropagator3D itp{g, v, 0.02, mass};
    itp.relax(psi, 4000);
    EXPECT_NEAR(ses::mean_energy(psi, v, mass), 0.5 * w / std::sqrt(mass), 2e-3);
    // The mass-aware readout differs from the m = 1 readout (kinetic /m).
    EXPECT_GT(ses::mean_energy(psi, v), ses::mean_energy(psi, v, mass));
}
