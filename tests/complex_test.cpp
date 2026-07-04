// RED: specification for the hand-rolled complex number type ses::Complex<T>.
//
// This is the foundation of the whole numerical core: the TDSE state psi is a
// complex field, the FFT twiddle factors are complex, and the split-operator
// phase factors are complex. Per the purist reinvention boundary
// (docs/ARCHITECTURE.md) we hand-roll this instead of using std::complex.
//
// Oracle: exact arithmetic identities (i*i = -1, |3-4i| = 5, ...). All simple
// operations use values exactly representable in binary floating point, so
// EXPECT_EQ is legitimate there; sqrt/division results use EXPECT_DOUBLE_EQ.

#include <core/complex.hpp>

#include <gtest/gtest.h>

namespace {

using Cd = ses::Complex<double>;

TEST(Complex, DefaultConstructsToZero) {
    constexpr Cd z{};
    EXPECT_EQ(z.re, 0.0);
    EXPECT_EQ(z.im, 0.0);
}

TEST(Complex, AggregateConstruction) {
    constexpr Cd z{3.0, -4.0};
    EXPECT_EQ(z.re, 3.0);
    EXPECT_EQ(z.im, -4.0);
}

TEST(Complex, Addition) {
    constexpr Cd s = Cd{1.0, 2.0} + Cd{3.0, -5.0};
    EXPECT_EQ(s.re, 4.0);
    EXPECT_EQ(s.im, -3.0);
}

TEST(Complex, Subtraction) {
    constexpr Cd d = Cd{1.0, 2.0} - Cd{3.0, -5.0};
    EXPECT_EQ(d.re, -2.0);
    EXPECT_EQ(d.im, 7.0);
}

TEST(Complex, MultiplicationSatisfiesISquaredEqualsMinusOne) {
    constexpr Cd i{0.0, 1.0};
    constexpr Cd ii = i * i;
    EXPECT_EQ(ii.re, -1.0);
    EXPECT_EQ(ii.im, 0.0);
}

TEST(Complex, MultiplicationGeneralCase) {
    // (1+2i)(3+4i) = 3 + 4i + 6i + 8i^2 = -5 + 10i
    constexpr Cd p = Cd{1.0, 2.0} * Cd{3.0, 4.0};
    EXPECT_EQ(p.re, -5.0);
    EXPECT_EQ(p.im, 10.0);
}

TEST(Complex, ScalarMultiplicationFromBothSides) {
    constexpr Cd l = 2.0 * Cd{1.0, -3.0};
    constexpr Cd r = Cd{1.0, -3.0} * 2.0;
    EXPECT_EQ(l.re, 2.0);
    EXPECT_EQ(l.im, -6.0);
    EXPECT_EQ(r.re, 2.0);
    EXPECT_EQ(r.im, -6.0);
}

TEST(Complex, Conjugate) {
    constexpr Cd c = conj(Cd{3.0, -4.0});
    EXPECT_EQ(c.re, 3.0);
    EXPECT_EQ(c.im, 4.0);
}

TEST(Complex, NormSquaredIsSquaredMagnitude) {
    // |3-4i|^2 = 9 + 16 = 25 (this is the probability-density operation)
    constexpr double n = norm_sq(Cd{3.0, -4.0});
    EXPECT_EQ(n, 25.0);
}

TEST(Complex, AbsIsMagnitude) {
    EXPECT_DOUBLE_EQ(abs(Cd{3.0, -4.0}), 5.0);
}

TEST(Complex, DivisionByComplex) {
    // (-5+10i)/(3+4i) = (1+2i)  [inverse of the multiplication case above]
    const Cd q = Cd{-5.0, 10.0} / Cd{3.0, 4.0};
    EXPECT_DOUBLE_EQ(q.re, 1.0);
    EXPECT_DOUBLE_EQ(q.im, 2.0);
}

TEST(Complex, MultiplicationConjugateGivesRealNormSq) {
    // z * conj(z) must be purely real and equal |z|^2
    constexpr Cd z{3.0, -4.0};
    constexpr Cd zz = z * conj(z);
    EXPECT_EQ(zz.re, 25.0);
    EXPECT_EQ(zz.im, 0.0);
}

}  // namespace
