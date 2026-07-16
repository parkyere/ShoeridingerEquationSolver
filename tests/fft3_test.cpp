// RED: specification for the 3D FFT over a Field3D (1D transforms applied
// along each axis; same e^{-} kernel and 1/N-on-inverse convention as 1D).
//
// Axis sizes and spike positions are all DISTINCT (8 x 16 x 4, spike at
// (3,5,1)) so that any axis mix-up or stride bug lands the spike in the
// wrong bin and fails loudly.

#include <complex>
#include <core/fft.hpp>
#include <core/field.hpp>
import ses.grid;

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;

constexpr double kTol = 1e-12;
constexpr double kTwoPi = 2.0 * std::numbers::pi;

Grid3D make_grid(int nx, int ny, int nz) {
    return Grid3D{Grid1D{0.0, 1.0, nx}, Grid1D{0.0, 1.0, ny}, Grid1D{0.0, 1.0, nz}};
}

TEST(Fft3, DcConcentratesAtOriginBin) {
    Field3D f{make_grid(8, 4, 2)};
    for (std::complex<double>& v : f.data()) {
        v = std::complex<double>{1.0, 0.0};
    }
    ses::fft(f);
    EXPECT_NEAR(f(0, 0, 0).real(), 64.0, kTol);  // N = 8*4*2
    EXPECT_NEAR(f(0, 0, 0).imag(), 0.0, kTol);
    EXPECT_NEAR(f(1, 0, 0).real(), 0.0, kTol);
    EXPECT_NEAR(f(0, 1, 0).real(), 0.0, kTol);
    EXPECT_NEAR(f(0, 0, 1).real(), 0.0, kTol);
}

TEST(Fft3, PlaneWaveSpikesAtItsBin) {
    // x = e^{+i 2 pi (3 i/8 + 5 j/16 + 1 k/4)} -> single spike at (3,5,1)
    // with value N = 8*16*4 = 512.
    const int nx = 8, ny = 16, nz = 4;
    Field3D f{make_grid(nx, ny, nz)};
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const double th = kTwoPi * (3.0 * i / nx + 5.0 * j / ny + 1.0 * k / nz);
                f(i, j, k) = std::complex<double>{std::cos(th), std::sin(th)};
            }
        }
    }
    ses::fft(f);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const double expected = (i == 3 && j == 5 && k == 1) ? 512.0 : 0.0;
                EXPECT_NEAR(f(i, j, k).real(), expected, 1e-9)
                    << "at (" << i << "," << j << "," << k << ")";
                EXPECT_NEAR(f(i, j, k).imag(), 0.0, 1e-9);
            }
        }
    }
}

TEST(Fft3, InverseRestoresInput) {
    Field3D f{make_grid(8, 4, 2)};
    for (int idx = 0; idx < f.size(); ++idx) {
        const double t = static_cast<double>(idx);
        f.data()[static_cast<std::size_t>(idx)] =
            std::complex<double>{std::sin(1.3 * t) + 0.1 * t, std::cos(2.7 * t)};
    }
    const Field3D original = f;
    ses::fft(f);
    ses::ifft(f);
    for (int idx = 0; idx < f.size(); ++idx) {
        const std::size_t s = static_cast<std::size_t>(idx);
        EXPECT_NEAR(f.data()[s].real(), original.data()[s].real(), kTol);
        EXPECT_NEAR(f.data()[s].imag(), original.data()[s].imag(), kTol);
    }
}

TEST(Fft3, ParsevalEnergyIdentity) {
    Field3D f{make_grid(4, 8, 2)};
    for (int idx = 0; idx < f.size(); ++idx) {
        const double t = static_cast<double>(idx);
        f.data()[static_cast<std::size_t>(idx)] =
            std::complex<double>{std::cos(0.9 * t), 0.2 * std::sin(1.7 * t)};
    }
    double time_energy = 0.0;
    for (const std::complex<double>& v : f.data()) {
        time_energy += std::norm(v);
    }
    ses::fft(f);
    double freq_energy = 0.0;
    for (const std::complex<double>& v : f.data()) {
        freq_energy += std::norm(v);
    }
    EXPECT_NEAR(time_energy, freq_energy / 64.0, 1e-10);
}

}  // namespace
