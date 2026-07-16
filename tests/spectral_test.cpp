// RED: specification for the FFT-bin -> physical-wavenumber mapping.
//
// For a periodic Grid1D of n points over length L = xmax - xmin, the FFT bin j
// corresponds to wavenumber
//     k_j = 2 pi j / L          for j = 0 .. n/2 - 1   (non-negative branch)
//     k_j = 2 pi (j - n) / L    for j = n/2 .. n - 1   (negative branch)
// (the NumPy fftfreq layout). Getting this wrong is THE classic split-operator
// bug: the kinetic phase then scrambles high-frequency components.

import ses.grid;
#include <core/spectral.hpp>

#include <gtest/gtest.h>

#include <numbers>
#include <vector>

namespace {

using ses::Grid1D;

constexpr double kTwoPi = 2.0 * std::numbers::pi;

TEST(Wavenumbers, MatchesFftfreqLayout) {
    // L = 8, n = 8  ->  dk = 2 pi / 8.
    const Grid1D g{0.0, 8.0, 8};
    const std::vector<double> k = ses::wavenumbers(g);
    ASSERT_EQ(k.size(), 8u);
    const double dk = kTwoPi / 8.0;
    EXPECT_DOUBLE_EQ(k[0], 0.0);
    EXPECT_DOUBLE_EQ(k[1], dk);
    EXPECT_DOUBLE_EQ(k[2], 2.0 * dk);
    EXPECT_DOUBLE_EQ(k[3], 3.0 * dk);
    EXPECT_DOUBLE_EQ(k[4], -4.0 * dk);  // Nyquist bin: negative branch
    EXPECT_DOUBLE_EQ(k[5], -3.0 * dk);
    EXPECT_DOUBLE_EQ(k[6], -2.0 * dk);
    EXPECT_DOUBLE_EQ(k[7], -dk);
}

TEST(Wavenumbers, SpacingIsIndependentOfOffset) {
    // The mapping depends only on L and n, not on where the box sits.
    const Grid1D g{-13.0, -5.0, 8};  // same L = 8
    const std::vector<double> k = ses::wavenumbers(g);
    const double dk = kTwoPi / 8.0;
    EXPECT_DOUBLE_EQ(k[1], dk);
    EXPECT_DOUBLE_EQ(k[7], -dk);
}

}  // namespace
