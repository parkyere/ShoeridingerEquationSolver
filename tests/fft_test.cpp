// RED: specification for the hand-written radix-2 FFT (the project centerpiece;
// the split-operator propagator applies the kinetic factor in k-space with it).
//
// Convention pinned here (matches FFTW/NumPy):
//     forward:  X_k = sum_n x_n e^{-2 pi i k n / N}   (unnormalized)
//     inverse:  x_n = (1/N) sum_k X_k e^{+2 pi i k n / N}
// so ifft(fft(x)) == x. Sizes are powers of two (radix-2).
//
// Oracles: DC -> bin 0; delta -> flat spectrum; e^{+2 pi i k0 n/N} -> a spike
// at k0 ONLY (this is the one test that detects a wrong kernel sign -- the
// spike would land at N-k0); cosine -> conjugate bin pair; round-trip;
// linearity; Parseval.

#include <core/complex.hpp>
#include <core/fft.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

namespace {

using Cd = ses::Complex<double>;
using CVec = std::vector<Cd>;

constexpr double kTol = 1e-12;
constexpr double kTwoPi = 2.0 * std::numbers::pi;

void expect_near(const Cd& z, double re, double im, double tol = kTol) {
    EXPECT_NEAR(z.re, re, tol);
    EXPECT_NEAR(z.im, im, tol);
}

TEST(Fft, SizeOneIsIdentity) {
    CVec a{Cd{3.0, -4.0}};
    ses::fft(a);
    expect_near(a[0], 3.0, -4.0);
}

TEST(Fft, DcSignalConcentratesAtBinZero) {
    CVec a(8, Cd{1.0, 0.0});
    ses::fft(a);
    expect_near(a[0], 8.0, 0.0);  // X_0 = N
    for (std::size_t k = 1; k < a.size(); ++k) {
        expect_near(a[k], 0.0, 0.0);
    }
}

TEST(Fft, DeltaHasFlatSpectrum) {
    CVec a(8, Cd{});
    a[0] = Cd{1.0, 0.0};
    ses::fft(a);
    for (std::size_t k = 0; k < a.size(); ++k) {
        expect_near(a[k], 1.0, 0.0);
    }
}

TEST(Fft, ComplexExponentialPinsKernelSign) {
    // x_n = e^{+2 pi i k0 n / N}, k0 = 3, N = 16.
    // With the e^{-} forward kernel the spike is at k = 3 with value N.
    // A wrong (e^{+}) kernel would put it at k = 13 -- this test forbids that.
    const std::size_t n = 16;
    const std::size_t k0 = 3;
    CVec a(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double th = kTwoPi * static_cast<double>(k0 * i) / static_cast<double>(n);
        a[i] = Cd{std::cos(th), std::sin(th)};
    }
    ses::fft(a);
    for (std::size_t k = 0; k < n; ++k) {
        if (k == k0) {
            expect_near(a[k], 16.0, 0.0);
        } else {
            expect_near(a[k], 0.0, 0.0);
        }
    }
}

TEST(Fft, RealCosineSplitsIntoConjugateBinPair) {
    // x_n = cos(2 pi k0 n / N), k0 = 3, N = 16  ->  X_3 = X_13 = N/2.
    const std::size_t n = 16;
    const std::size_t k0 = 3;
    CVec a(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double th = kTwoPi * static_cast<double>(k0 * i) / static_cast<double>(n);
        a[i] = Cd{std::cos(th), 0.0};
    }
    ses::fft(a);
    for (std::size_t k = 0; k < n; ++k) {
        if (k == k0 || k == n - k0) {
            expect_near(a[k], 8.0, 0.0);
        } else {
            expect_near(a[k], 0.0, 0.0);
        }
    }
}

CVec deterministic_signal(std::size_t n) {
    // Non-symmetric, non-sparse, reproducible test data (no RNG per repo rules).
    CVec a(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(i);
        a[i] = Cd{std::sin(1.3 * x) + 0.1 * x, std::cos(2.7 * x) - 0.05 * x};
    }
    return a;
}

TEST(Fft, InverseRestoresInput) {
    const CVec original = deterministic_signal(16);
    CVec a = original;
    ses::fft(a);
    ses::ifft(a);
    for (std::size_t i = 0; i < a.size(); ++i) {
        expect_near(a[i], original[i].re, original[i].im);
    }
}

TEST(Fft, IsLinear) {
    // fft(2x + y) == 2 fft(x) + fft(y)
    const std::size_t n = 8;
    CVec x = deterministic_signal(n);
    CVec y(n);
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = Cd{0.5 * static_cast<double>(i), -1.0};
    }
    CVec combo(n);
    for (std::size_t i = 0; i < n; ++i) {
        combo[i] = 2.0 * x[i] + y[i];
    }
    ses::fft(x);
    ses::fft(y);
    ses::fft(combo);
    for (std::size_t i = 0; i < n; ++i) {
        const Cd expected = 2.0 * x[i] + y[i];
        expect_near(combo[i], expected.re, expected.im);
    }
}

TEST(Fft, ParsevalEnergyIdentity) {
    // sum |x_n|^2 == (1/N) sum |X_k|^2
    CVec a = deterministic_signal(16);
    double time_energy = 0.0;
    for (const Cd& z : a) {
        time_energy += ses::norm_sq(z);
    }
    ses::fft(a);
    double freq_energy = 0.0;
    for (const Cd& z : a) {
        freq_energy += ses::norm_sq(z);
    }
    EXPECT_NEAR(time_energy, freq_energy / 16.0, 1e-10);
}

}  // namespace
