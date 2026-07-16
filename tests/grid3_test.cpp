// RED: specification for the 3D periodic grid, composed of three Grid1D axes.
//
// Memory-layout convention pinned here (load-bearing for the 3D FFT and for
// the GPU 3D-volume upload): X FASTEST --
//     flat(i, j, k) = i + nx * (j + ny * k)
// so x-lines are contiguous, y-stride is nx, z-stride is nx*ny.

import ses.grid;

#include <gtest/gtest.h>

namespace {

using ses::Grid1D;
using ses::Grid3D;

// Deliberately distinct axis sizes so any axis mix-up changes the answer.
const Grid3D kGrid{Grid1D{0.0, 8.0, 16}, Grid1D{0.0, 4.0, 8}, Grid1D{-1.0, 1.0, 4}};

TEST(Grid3D, SizeIsProductOfAxes) {
    EXPECT_EQ(kGrid.size(), 16 * 8 * 4);
}

TEST(Grid3D, FlatIndexIsXFastest) {
    EXPECT_EQ(kGrid.flat(0, 0, 0), 0);
    EXPECT_EQ(kGrid.flat(1, 0, 0) - kGrid.flat(0, 0, 0), 1);        // x-stride 1
    EXPECT_EQ(kGrid.flat(0, 1, 0) - kGrid.flat(0, 0, 0), 16);       // y-stride nx
    EXPECT_EQ(kGrid.flat(0, 0, 1) - kGrid.flat(0, 0, 0), 16 * 8);   // z-stride nx*ny
    EXPECT_EQ(kGrid.flat(15, 7, 3), 16 * 8 * 4 - 1);                // last cell
}

TEST(Grid3D, CellVolumeIsProductOfSpacings) {
    // h = (0.5, 0.5, 0.5) -> exact 0.125.
    EXPECT_EQ(kGrid.cell_volume(), 0.125);
}

TEST(Grid3D, AxisCoordsComeFromAxisGrids) {
    EXPECT_EQ(kGrid.x.coord(3), 1.5);
    EXPECT_EQ(kGrid.y.coord(2), 1.0);
    EXPECT_EQ(kGrid.z.coord(1), -0.5);
}

}  // namespace
