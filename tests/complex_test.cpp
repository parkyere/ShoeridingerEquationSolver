// Contract specification for std::complex<T>.
//
// The reinvention boundary excludes the C++ STANDARD LIBRARY (only third-
// party libraries are reinvented), so std::complex is used directly. These
// tests pin the arithmetic contract the whole core relies on -- construction,
// +,-,*,/ (built with -fcx-limited-range: the naive formulas, no Annex G NaN
// fixups), scalar multiply, conj (via ADL into std), abs, and std::norm == |z|^2.
//
// Oracle: exact arithmetic identities (i*i = -1, |3-4i| = 5, ...). All simple
// operations use values exactly representable in binary floating point, so
// EXPECT_EQ is legitimate there; sqrt/division results use EXPECT_DOUBLE_EQ.

#include <complex>

#include <gtest/gtest.h>

// C++20 makes std::complex ARITHMETIC constexpr, and MSVC and GCC constant-
// evaluate it. Clang (even 22) cannot constant-fold libstdc++'s
// __complex__-based compound assignments (a long-standing clang/libstdc++
// gap; libc++ is fine), so on that one combination the SAME expressions and
// value pins run at runtime only -- the arithmetic contract is unchanged,
// only the compile-time-evaluability pin is narrowed to where it holds.
// (Construction/conj/norm stay constexpr everywhere.)
#if defined(__clang__) && defined(__GLIBCXX__)
#define SES_COMPLEX_ARITH_CONSTEXPR const
#else
#define SES_COMPLEX_ARITH_CONSTEXPR constexpr
#endif

namespace {

using Cd = std::complex<double>;

TEST(Complex, DefaultConstructsToZero) {
    constexpr Cd z{};
    EXPECT_EQ(z.real(), 0.0);
    EXPECT_EQ(z.imag(), 0.0);
}

TEST(Complex, AggregateConstruction) {
    constexpr Cd z{3.0, -4.0};
    EXPECT_EQ(z.real(), 3.0);
    EXPECT_EQ(z.imag(), -4.0);
}

TEST(Complex, Addition) {
    SES_COMPLEX_ARITH_CONSTEXPR Cd s = Cd{1.0, 2.0} + Cd{3.0, -5.0};
    EXPECT_EQ(s.real(), 4.0);
    EXPECT_EQ(s.imag(), -3.0);
}

TEST(Complex, Subtraction) {
    SES_COMPLEX_ARITH_CONSTEXPR Cd d = Cd{1.0, 2.0} - Cd{3.0, -5.0};
    EXPECT_EQ(d.real(), -2.0);
    EXPECT_EQ(d.imag(), 7.0);
}

TEST(Complex, MultiplicationSatisfiesISquaredEqualsMinusOne) {
    constexpr Cd i{0.0, 1.0};
    SES_COMPLEX_ARITH_CONSTEXPR Cd ii = i * i;
    EXPECT_EQ(ii.real(), -1.0);
    EXPECT_EQ(ii.imag(), 0.0);
}

TEST(Complex, MultiplicationGeneralCase) {
    // (1+2i)(3+4i) = 3 + 4i + 6i + 8i^2 = -5 + 10i
    SES_COMPLEX_ARITH_CONSTEXPR Cd p = Cd{1.0, 2.0} * Cd{3.0, 4.0};
    EXPECT_EQ(p.real(), -5.0);
    EXPECT_EQ(p.imag(), 10.0);
}

TEST(Complex, ScalarMultiplicationFromBothSides) {
    SES_COMPLEX_ARITH_CONSTEXPR Cd l = 2.0 * Cd{1.0, -3.0};
    SES_COMPLEX_ARITH_CONSTEXPR Cd r = Cd{1.0, -3.0} * 2.0;
    EXPECT_EQ(l.real(), 2.0);
    EXPECT_EQ(l.imag(), -6.0);
    EXPECT_EQ(r.real(), 2.0);
    EXPECT_EQ(r.imag(), -6.0);
}

TEST(Complex, Conjugate) {
    constexpr Cd c = conj(Cd{3.0, -4.0});
    EXPECT_EQ(c.real(), 3.0);
    EXPECT_EQ(c.imag(), 4.0);
}

TEST(Complex, NormSquaredIsSquaredMagnitude) {
    // |3-4i|^2 = 9 + 16 = 25 -- std::norm is the probability-density operation.
    constexpr double n = std::norm(Cd{3.0, -4.0});
    EXPECT_EQ(n, 25.0);
}

TEST(Complex, AbsIsMagnitude) {
    EXPECT_DOUBLE_EQ(abs(Cd{3.0, -4.0}), 5.0);
}

TEST(Complex, DivisionByComplex) {
    // (-5+10i)/(3+4i) = (1+2i)  [inverse of the multiplication case above]
    const Cd q = Cd{-5.0, 10.0} / Cd{3.0, 4.0};
    EXPECT_DOUBLE_EQ(q.real(), 1.0);
    EXPECT_DOUBLE_EQ(q.imag(), 2.0);
}

TEST(Complex, MultiplicationConjugateGivesRealNormSq) {
    // z * conj(z) must be purely real and equal |z|^2
    constexpr Cd z{3.0, -4.0};
    SES_COMPLEX_ARITH_CONSTEXPR Cd zz = z * conj(z);
    EXPECT_EQ(zz.real(), 25.0);
    EXPECT_EQ(zz.imag(), 0.0);
}

}  // namespace
