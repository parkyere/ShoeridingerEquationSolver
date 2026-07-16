// RED: specification for the complex scalar field over a Grid1D.
//
// The field holds the wavefunction psi on the grid. The load-bearing physics
// here is the DISCRETE NORM: ||psi||^2 = sum_i |psi_i|^2 * h  (the grid weight
// h must be included, or every probability downstream is silently wrong).
//
// Oracles:
//  - constant field on an exact-binary grid -> exact norm value;
//  - the continuum-normalized Gaussian (2 pi s^2)^(-1/4) exp(-x^2/(4 s^2))
//    sampled on a wide, fine grid must have discrete norm ~= 1 (this catches
//    both a missing h and a wrong h);
//  - normalize() must produce unit norm for an arbitrary field.

#include <core/complex.hpp>
#include <core/field.hpp>
import ses.grid;

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

namespace {

using ses::Complex;
using ses::Field1D;
using ses::Grid1D;

TEST(Field1D, SizeMatchesGrid) {
    const Field1D f{Grid1D{0.0, 8.0, 16}};
    EXPECT_EQ(f.size(), 16);
    EXPECT_EQ(f.grid().n, 16);
}

TEST(Field1D, InitializesToZero) {
    const Field1D f{Grid1D{0.0, 8.0, 16}};
    for (int i = 0; i < f.size(); ++i) {
        EXPECT_EQ(f[i].real(), 0.0);
        EXPECT_EQ(f[i].imag(), 0.0);
    }
}

TEST(Field1D, ElementReadWrite) {
    Field1D f{Grid1D{0.0, 8.0, 16}};
    f[3] = Complex<double>{1.5, -2.5};
    EXPECT_EQ(f[3].real(), 1.5);
    EXPECT_EQ(f[3].imag(), -2.5);
}

TEST(Field1D, NormSqIncludesGridWeight) {
    // [0, 8) with n=16: h = 0.5 (exact in binary).
    // Constant psi = 1  ->  sum |1|^2 * h = 16 * 0.5 = 8 exactly.
    Field1D f{Grid1D{0.0, 8.0, 16}};
    for (int i = 0; i < f.size(); ++i) {
        f[i] = Complex<double>{1.0, 0.0};
    }
    EXPECT_EQ(norm_sq(f), 8.0);
}

TEST(Field1D, ContinuumNormalizedGaussianHasUnitDiscreteNorm) {
    // psi(x) = (2 pi s^2)^(-1/4) exp(-x^2/(4 s^2)), s = 1.
    // integral |psi|^2 dx = 1; on [-16,16) n=256 (h = 0.125, tails ~e^-128)
    // the Riemann sum of a smooth decaying function converges to machine
    // precision.
    const Grid1D g{-16.0, 16.0, 256};
    Field1D f{g};
    const double s = 1.0;
    const double amp = std::pow(2.0 * std::numbers::pi * s * s, -0.25);
    for (int i = 0; i < f.size(); ++i) {
        const double x = g.coord(i);
        f[i] = Complex<double>{amp * std::exp(-x * x / (4.0 * s * s)), 0.0};
    }
    EXPECT_NEAR(norm_sq(f), 1.0, 1e-12);
}

TEST(Field1D, NormalizeMakesUnitNorm) {
    Field1D f{Grid1D{-4.0, 4.0, 32}};
    for (int i = 0; i < f.size(); ++i) {
        f[i] = Complex<double>{0.3 * i, -0.7 * i};  // arbitrary non-trivial data
    }
    normalize(f);
    EXPECT_NEAR(norm_sq(f), 1.0, 1e-12);
}

}  // namespace
