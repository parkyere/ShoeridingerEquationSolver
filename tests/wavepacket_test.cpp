// RED: specification for the Gaussian wavepacket factory and the observables
// used to interrogate it.
//
// Physics (atomic units, docs/ARCHITECTURE.md):
//     psi(x) = (2 pi s^2)^(-1/4) exp(-(x-x0)^2 / (4 s^2)) exp(i k0 x)
// - |psi|^2 is Gaussian with mean x0 and standard deviation s;
// - the exp(i k0 x) phase carries mean momentum <p> = k0 (in the momentum
//   representation the packet is centered at k0);
// - the state is normalized.
//
// Observables must be scale-invariant (they divide by the norm), so they work
// on unnormalized fields too.

#include <complex>
#include <core/field.hpp>
import ses.grid;
#include <core/observables.hpp>
#include <core/wavepacket.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace {

using ses::Field1D;
using ses::Grid1D;

const Grid1D kGrid{-16.0, 16.0, 256};  // h = 0.125, tails ~ e^-36 at the walls

TEST(GaussianWavepacket, IsNormalized) {
    const Field1D psi = ses::gaussian_wavepacket(kGrid, 2.0, 1.5, 3.0);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(GaussianWavepacket, PhaseCarriesThePlaneWave) {
    // psi(x) * e^{-i k0 x} must be real and non-negative everywhere.
    const double k0 = 3.0;
    const Field1D psi = ses::gaussian_wavepacket(kGrid, 2.0, 1.5, k0);
    for (int i = 0; i < psi.size(); ++i) {
        const double x = kGrid.coord(i);
        const std::complex<double> unwound =
            psi[i] * std::complex<double>{std::cos(-k0 * x), std::sin(-k0 * x)};
        EXPECT_NEAR(unwound.imag(), 0.0, 1e-12);
        EXPECT_GE(unwound.real(), -1e-12);
    }
}

TEST(GaussianWavepacket, ZeroMomentumPacketIsReal) {
    const Field1D psi = ses::gaussian_wavepacket(kGrid, 0.0, 1.0, 0.0);
    for (int i = 0; i < psi.size(); ++i) {
        EXPECT_EQ(psi[i].imag(), 0.0);
    }
}

TEST(Observables, MeanPositionOfPacketIsCenter) {
    const Field1D psi = ses::gaussian_wavepacket(kGrid, 2.0, 1.5, 3.0);
    EXPECT_NEAR(ses::mean_position(psi), 2.0, 1e-10);
}

TEST(Observables, SigmaXOfPacketIsSigma) {
    const Field1D psi = ses::gaussian_wavepacket(kGrid, 2.0, 1.5, 3.0);
    EXPECT_NEAR(ses::sigma_x(psi), 1.5, 1e-10);
}

TEST(Observables, MeanMomentumOfPacketIsK0) {
    // Exercises the fft + wavenumbers pipeline the propagator relies on.
    const Field1D psi = ses::gaussian_wavepacket(kGrid, 2.0, 1.5, 3.0);
    EXPECT_NEAR(ses::mean_momentum(psi), 3.0, 1e-8);
}

TEST(Observables, AreScaleInvariant) {
    Field1D psi = ses::gaussian_wavepacket(kGrid, -1.0, 2.0, 1.0);
    for (int i = 0; i < psi.size(); ++i) {
        psi[i] = 7.0 * psi[i];  // break normalization on purpose
    }
    EXPECT_NEAR(ses::mean_position(psi), -1.0, 1e-10);
    EXPECT_NEAR(ses::sigma_x(psi), 2.0, 1e-10);
    EXPECT_NEAR(ses::mean_momentum(psi), 1.0, 1e-8);
}

}  // namespace
