// RED: specification for the 1D uniform grid.
//
// Convention (load-bearing for the whole project): the grid is PERIODIC, as
// required by the split-operator Fourier propagator. For extent [xmin, xmax)
// with n points:
//     h = (xmax - xmin) / n,     x_i = xmin + i*h,   i = 0 .. n-1.
// xmax itself is NOT a grid point -- it aliases to xmin under periodicity.
// (The endpoint-inclusive convention h = L/(n-1) would silently break the FFT
// wavenumber mapping later; this spec exists to forbid it.)

import ses.grid;

#include <gtest/gtest.h>

namespace {

using ses::Grid1D;

TEST(Grid1D, SizeIsPointCount) {
    constexpr Grid1D g{0.0, 10.0, 10};
    EXPECT_EQ(g.size(), 10);
}

TEST(Grid1D, SpacingIsExtentOverN) {
    // Periodic convention: h = L/n, NOT L/(n-1).
    constexpr Grid1D g{0.0, 10.0, 10};
    EXPECT_EQ(g.spacing(), 1.0);
}

TEST(Grid1D, CoordOfFirstPointIsXmin) {
    constexpr Grid1D g{-5.0, 5.0, 100};
    EXPECT_EQ(g.coord(0), -5.0);
}

TEST(Grid1D, LastPointStopsOneStepShortOfXmax) {
    // coord(n-1) = xmax - h; xmax aliases to xmin under periodicity.
    constexpr Grid1D g{0.0, 10.0, 10};
    EXPECT_EQ(g.coord(9), 9.0);
}

TEST(Grid1D, CenterPointOfSymmetricGrid) {
    // [-5, 5) with n=100: h=0.1, coord(50) = -5 + 5.0 = 0 exactly.
    constexpr Grid1D g{-5.0, 5.0, 100};
    EXPECT_DOUBLE_EQ(g.coord(50), 0.0);
}

TEST(Grid1D, ExtentAccessors) {
    constexpr Grid1D g{-2.0, 3.0, 50};
    EXPECT_EQ(g.xmin, -2.0);
    EXPECT_EQ(g.xmax, 3.0);
}

}  // namespace
