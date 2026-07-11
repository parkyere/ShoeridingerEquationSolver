#pragma once

// Hand-written radix-2 Cooley-Tukey FFT (purist reinvention boundary: no FFTW).
//
// Convention (matches FFTW/NumPy, pinned by tests/fft_test.cpp):
//     forward:  X_k = sum_n x_n e^{-2 pi i k n / N}   (unnormalized)
//     inverse:  x_n = (1/N) sum_k X_k e^{+2 pi i k n / N}
// Sizes must be powers of two.

#include <core/complex.hpp>
#include <core/field.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <utility>
#include <vector>

namespace ses {

// In-place forward transform of a contiguous line of length n.
// Iterative: bit-reversal permutation, then butterfly passes of doubling
// length.
inline void fft(Complex<double>* a, std::size_t n) {
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

    // Twiddle table w[j] = e^{-2 pi i j / n}, j = 0 .. n/2-1, computed directly
    // so every factor sits within ~1 ulp of the unit circle. (The recurrence
    // w *= wlen drifts off the circle and systematically damps the norm --
    // caught by SplitOperator.ConservesNorm over 200 steps.)
    std::vector<Complex<double>> w(n / 2);
    const double ang = -2.0 * std::numbers::pi / static_cast<double>(n);
    for (std::size_t j = 0; j < n / 2; ++j) {
        const double th = ang * static_cast<double>(j);
        w[j] = Complex<double>{std::cos(th), std::sin(th)};
    }

    // Butterfly passes: len = 2, 4, ..., n. Stage len uses w_len^j = w[j * n/len].
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t stride = n / len;
        for (std::size_t i = 0; i < n; i += len) {
            for (std::size_t j = 0; j < len / 2; ++j) {
                const Complex<double> u = a[i + j];
                const Complex<double> v = a[i + j + len / 2] * w[j * stride];
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
            }
        }
    }
}

inline void fft(std::vector<Complex<double>>& a) { fft(a.data(), a.size()); }

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

// 3D forward transform: 1D FFT per axis (x-lines contiguous in the x-fastest
// layout; y/z lines gathered into per-thread scratch). Each line is owned by
// exactly one thread, so the threaded result is BITWISE IDENTICAL to serial.
inline void fft(Field3D& f) {
    std::vector<Complex<double>>& a = f.data();
    const Grid3D& g = f.grid();
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;

#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            fft(a.data() + g.flat(0, j, k), static_cast<std::size_t>(nx));
        }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Complex<double>> line(static_cast<std::size_t>(ny));
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
        for (int k = 0; k < nz; ++k) {
            for (int i = 0; i < nx; ++i) {
                for (int j = 0; j < ny; ++j) {
                    line[static_cast<std::size_t>(j)] =
                        a[static_cast<std::size_t>(g.flat(i, j, k))];
                }
                fft(line.data(), line.size());
                for (int j = 0; j < ny; ++j) {
                    a[static_cast<std::size_t>(g.flat(i, j, k))] =
                        line[static_cast<std::size_t>(j)];
                }
            }
        }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<Complex<double>> line(static_cast<std::size_t>(nz));
#ifdef _OPENMP
#pragma omp for collapse(2) schedule(static)
#endif
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                for (int k = 0; k < nz; ++k) {
                    line[static_cast<std::size_t>(k)] =
                        a[static_cast<std::size_t>(g.flat(i, j, k))];
                }
                fft(line.data(), line.size());
                for (int k = 0; k < nz; ++k) {
                    a[static_cast<std::size_t>(g.flat(i, j, k))] =
                        line[static_cast<std::size_t>(k)];
                }
            }
        }
    }
}

// 3D inverse: conjugation identity with N = nx*ny*nz. The elementwise loops
// are threaded (disjoint elements: bitwise identical to serial).
inline void ifft(Field3D& f) {
    std::vector<Complex<double>>& a = f.data();
    const std::int64_t n = static_cast<std::int64_t>(a.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < n; ++i) {
        a[static_cast<std::size_t>(i)] = conj(a[static_cast<std::size_t>(i)]);
    }
    fft(f);
    const double inv = 1.0 / static_cast<double>(f.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::int64_t i = 0; i < n; ++i) {
        a[static_cast<std::size_t>(i)] = inv * conj(a[static_cast<std::size_t>(i)]);
    }
}

}  // namespace ses
