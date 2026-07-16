// RED: specification for imaginary-time relaxation to the ground state.
//
// Substituting t -> -i tau in the TDSE turns the evolution operator into
// e^{-H tau}, which damps every eigencomponent as e^{-E_n tau}; after
// renormalization the state converges to the ground state (as long as the
// initial guess overlaps it). Same Strang splitting as real time, but the
// phase factors become real decay factors e^{-V dtau/2}, e^{-k^2 dtau/2},
// and the state must be renormalized every step (the flow is not unitary).
//
// Oracles:
//  - harmonic well: E0 = omega/2 exactly, sigma_x = 1/sqrt(2 omega), <x> = 0,
//    starting from a deliberately WRONG initial packet (offset + wrong width);
//  - soft Coulomb (Z=1, a=1): bound (E0 < 0) and E0 ~= -0.6698 (literature
//    value for V = -1/sqrt(x^2+1)) -- the precursor of the hydrogen 1s cloud.

#include <core/field.hpp>
import ses.grid;
#include <core/imaginary_time.hpp>
#include <core/observables.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.wavepacket;
import ses.potential;

namespace {

using ses::Field1D;
using ses::Grid1D;
using ses::ImaginaryTimePropagator1D;

const Grid1D kGrid{-16.0, 16.0, 512};

TEST(ImaginaryTime, RelaxedStateIsNormalized) {
    const std::vector<double> v = ses::harmonic_potential(kGrid, 1.0, 0.0);
    Field1D psi = ses::gaussian_wavepacket(kGrid, 1.0, 2.0, 0.0);
    ImaginaryTimePropagator1D relaxer{kGrid, v, 0.005};
    relaxer.relax(psi, 100);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(ImaginaryTime, FindsHarmonicGroundState) {
    // omega = 1: E0 = 0.5, sigma_gs = 1/sqrt(2). Start displaced (x0 = 1)
    // and too wide (sigma = 2) -- relaxation must forget both.
    const double omega = 1.0;
    const std::vector<double> v = ses::harmonic_potential(kGrid, omega, 0.0);
    Field1D psi = ses::gaussian_wavepacket(kGrid, 1.0, 2.0, 0.0);
    ImaginaryTimePropagator1D relaxer{kGrid, v, 0.005};
    relaxer.relax(psi, 6000);  // tau = 30 >> 1/(E1-E0) = 1
    EXPECT_NEAR(ses::mean_energy(psi, v), 0.5 * omega, 1e-5);
    EXPECT_NEAR(ses::sigma_x(psi), 1.0 / std::sqrt(2.0 * omega), 1e-5);
    EXPECT_NEAR(ses::mean_position(psi), 0.0, 1e-8);
}

TEST(ImaginaryTime, FindsSoftCoulombBoundState) {
    // V = -1/sqrt(x^2+1): the 1D hydrogen stand-in. Ground state energy
    // ~= -0.6698 (literature).
    const std::vector<double> v = ses::soft_coulomb_potential(kGrid, 1.0, 1.0, 0.0);
    Field1D psi = ses::gaussian_wavepacket(kGrid, 0.5, 1.5, 0.0);
    ImaginaryTimePropagator1D relaxer{kGrid, v, 0.005};
    relaxer.relax(psi, 10000);  // tau = 50
    const double e0 = ses::mean_energy(psi, v);
    EXPECT_LT(e0, -0.5);              // clearly bound
    EXPECT_NEAR(e0, -0.6698, 1e-3);   // literature value
    EXPECT_NEAR(ses::mean_position(psi), 0.0, 1e-6);  // symmetric about nucleus
}

}  // namespace
