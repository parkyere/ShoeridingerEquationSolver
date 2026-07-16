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

#include <core/field.hpp>
import ses.grid;
#include <core/observables.hpp>
#include <core/propagator.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.wavepacket;

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
