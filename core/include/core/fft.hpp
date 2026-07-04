#pragma once

// Hand-written radix-2 Cooley-Tukey FFT (purist reinvention boundary: no FFTW).
//
// Convention (matches FFTW/NumPy, pinned by tests/fft_test.cpp):
//     forward:  X_k = sum_n x_n e^{-2 pi i k n / N}   (unnormalized)
//     inverse:  x_n = (1/N) sum_k X_k e^{+2 pi i k n / N}
// Sizes must be powers of two.

#include <core/complex.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <utility>
#include <vector>

namespace ses {

// In-place forward transform. Iterative: bit-reversal permutation, then
// butterfly passes of doubling length.
inline void fft(std::vector<Complex<double>>& a) {
    const std::size_t n = a.size();
    assert((n & (n - 1)) == 0 && "fft size must be a power of two");
    if (n < 2) {
        return;
    }

    // Bit-reversal permutation.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(a[i], a[j]);
        }
    }

    // Butterfly passes: len = 2, 4, ..., n.
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const double ang = -2.0 * std::numbers::pi / static_cast<double>(len);
        const Complex<double> wlen{std::cos(ang), std::sin(ang)};
        for (std::size_t i = 0; i < n; i += len) {
            Complex<double> w{1.0, 0.0};
            for (std::size_t j = 0; j < len / 2; ++j) {
                const Complex<double> u = a[i + j];
                const Complex<double> v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w = w * wlen;
            }
        }
    }
}

// In-place inverse transform via the conjugation identity:
//     ifft(X) = conj(fft(conj(X))) / N
inline void ifft(std::vector<Complex<double>>& a) {
    for (Complex<double>& z : a) {
        z = conj(z);
    }
    fft(a);
    const double inv = 1.0 / static_cast<double>(a.size());
    for (Complex<double>& z : a) {
        z = inv * conj(z);
    }
}

}  // namespace ses
