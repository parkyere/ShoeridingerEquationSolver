// RED: ses.heightfield contract. z = z_scale*|psi|^2/norm; phase -> per-vertex
// ses::phase_color; triangle soup (pos+normal, ses::Mesh); stride decimates
// the source lattice.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>

import ses.heightfield;
import ses.colormap;
import ses.field;
import ses.grid;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;

Grid3D small_grid() {
    return Grid3D{Grid1D{-1.0, 1.0, 4}, Grid1D{-1.0, 1.0, 4},
                  Grid1D{-1.0, 1.0, 1}};
}

TEST(Heightfield, SoupCountsAndSizes) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = 1.0;
        }
    }
    const ses::Heightfield hf = ses::heightfield_surface(psi, 3.0, 2.0, 1);
    // 3x3 cells, 2 triangles each, 3 soup vertices per triangle.
    EXPECT_EQ(hf.mesh.vertices.size(), 54u);
    EXPECT_EQ(hf.mesh.normals.size(), 54u);
    EXPECT_EQ(hf.colors.size(), 54u);
}

TEST(Heightfield, HeightIsScaledDensity) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = 1.0;
        }
    }
    psi(1, 2, 0) = std::complex<double>{0.0, 2.0};
    const double z_scale = 3.0;
    const double norm = 2.0;
    const ses::Heightfield hf = ses::heightfield_surface(psi, z_scale, norm, 1);
    EXPECT_NEAR(hf.mesh.vertices[0].x, g.x.coord(0), 1e-12);
    EXPECT_NEAR(hf.mesh.vertices[0].y, g.y.coord(0), 1e-12);
    EXPECT_NEAR(hf.mesh.vertices[0].z, z_scale * 1.0 / norm, 1e-12);
    const double bumped = z_scale * 4.0 / norm;
    bool found = false;
    for (std::size_t v = 0; v < hf.mesh.vertices.size(); ++v) {
        if (std::abs(hf.mesh.vertices[v].x - g.x.coord(1)) < 1e-12 &&
            std::abs(hf.mesh.vertices[v].y - g.y.coord(2)) < 1e-12) {
            EXPECT_NEAR(hf.mesh.vertices[v].z, bumped, 1e-12);
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST(Heightfield, PhaseColorsShareTheWheel) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = 1.0;
        }
    }
    psi(1, 2, 0) = std::complex<double>{0.0, 2.0};
    const ses::Heightfield hf = ses::heightfield_surface(psi, 1.0, 1.0, 1);
    const ses::Rgb flat = ses::phase_color(0.0);
    const ses::Rgb bumped = ses::phase_color(std::atan2(2.0, 0.0));
    EXPECT_NEAR(hf.colors[0].r, flat.r, 1e-12);
    EXPECT_NEAR(hf.colors[0].g, flat.g, 1e-12);
    EXPECT_NEAR(hf.colors[0].b, flat.b, 1e-12);
    bool found = false;
    for (std::size_t v = 0; v < hf.mesh.vertices.size(); ++v) {
        if (std::abs(hf.mesh.vertices[v].x - g.x.coord(1)) < 1e-12 &&
            std::abs(hf.mesh.vertices[v].y - g.y.coord(2)) < 1e-12) {
            EXPECT_NEAR(hf.colors[v].r, bumped.r, 1e-12);
            EXPECT_NEAR(hf.colors[v].g, bumped.g, 1e-12);
            EXPECT_NEAR(hf.colors[v].b, bumped.b, 1e-12);
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST(Heightfield, FlatFieldNormalsPointUp) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = std::complex<double>{0.6, 0.8};
        }
    }
    const ses::Heightfield hf = ses::heightfield_surface(psi, 5.0, 1.0, 1);
    for (const ses::Vec3d& n : hf.mesh.normals) {
        EXPECT_NEAR(n.x, 0.0, 1e-12);
        EXPECT_NEAR(n.y, 0.0, 1e-12);
        EXPECT_NEAR(n.z, 1.0, 1e-12);
    }
}

TEST(Heightfield, StrideDecimatesTheLattice) {
    // 8x8 plane, stride 2 -> samples 0,2,4,6 -> 4x4 vertex grid -> 54 soup.
    const Grid3D g{Grid1D{-2.0, 2.0, 8}, Grid1D{-2.0, 2.0, 8},
                   Grid1D{-1.0, 1.0, 1}};
    Field3D psi{g};
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            psi(i, j, 0) = 1.0;
        }
    }
    const ses::Heightfield hf = ses::heightfield_surface(psi, 1.0, 1.0, 2);
    EXPECT_EQ(hf.mesh.vertices.size(), 54u);
    // Decimated vertices sit on the SOURCE lattice coordinates.
    EXPECT_NEAR(hf.mesh.vertices[0].x, g.x.coord(0), 1e-12);
    bool has_stride_point = false;
    for (const ses::Vec3d& v : hf.mesh.vertices) {
        if (std::abs(v.x - g.x.coord(2)) < 1e-12) {
            has_stride_point = true;
        }
    }
    EXPECT_TRUE(has_stride_point);
}

TEST(Heightfield, AmpThresholdCropsToBoundingBox) {
    // 8x8 plane, single spike at (4,4). A positive amp_threshold meshes only
    // the spike's bounding box (+1-cell skirt), skipping the negligible rest.
    const Grid3D g{Grid1D{-2.0, 2.0, 8}, Grid1D{-2.0, 2.0, 8},
                   Grid1D{-1.0, 1.0, 1}};
    Field3D psi{g};
    psi(4, 4, 0) = 1.0;
    const double z_scale = 3.0;
    const double norm = 1.0;
    const ses::Heightfield hf =
        ses::heightfield_surface(psi, z_scale, norm, 1, 0.5);
    // cell (4,4) + 1-cell skirt -> samples [3..5]x[3..5] = 2x2 cells x 6.
    EXPECT_EQ(hf.mesh.vertices.size(), 24u);
    EXPECT_EQ(hf.mesh.normals.size(), 24u);
    EXPECT_EQ(hf.colors.size(), 24u);
    bool peak = false;
    for (const ses::Vec3d& v : hf.mesh.vertices) {
        EXPECT_GE(v.x, g.x.coord(3) - 1e-12);
        EXPECT_LE(v.x, g.x.coord(5) + 1e-12);
        EXPECT_GE(v.y, g.y.coord(3) - 1e-12);
        EXPECT_LE(v.y, g.y.coord(5) + 1e-12);
        if (std::abs(v.x - g.x.coord(4)) < 1e-12 &&
            std::abs(v.y - g.y.coord(4)) < 1e-12) {
            EXPECT_NEAR(v.z, z_scale * 1.0 / norm, 1e-12);
            peak = true;
        }
    }
    EXPECT_TRUE(peak);  // the spike survives at full height
}

TEST(Heightfield, ZeroAmpThresholdMeshesFullGrid) {
    // amp_threshold = 0 is the backward-compatible default: whole grid.
    const Grid3D g{Grid1D{-2.0, 2.0, 8}, Grid1D{-2.0, 2.0, 8},
                   Grid1D{-1.0, 1.0, 1}};
    Field3D psi{g};
    psi(4, 4, 0) = 1.0;
    const ses::Heightfield full = ses::heightfield_surface(psi, 3.0, 1.0, 1);
    const ses::Heightfield same =
        ses::heightfield_surface(psi, 3.0, 1.0, 1, 0.0);
    EXPECT_EQ(full.mesh.vertices.size(), 294u);  // 7x7 cells x 6
    EXPECT_EQ(same.mesh.vertices.size(), full.mesh.vertices.size());
}

TEST(Heightfield, FullyNegligibleFieldUnderThresholdIsEmpty) {
    const Grid3D g{Grid1D{-2.0, 2.0, 8}, Grid1D{-2.0, 2.0, 8},
                   Grid1D{-1.0, 1.0, 1}};
    Field3D psi{g};
    for (int j = 0; j < 8; ++j) {
        for (int i = 0; i < 8; ++i) {
            psi(i, j, 0) = 1e-3;
        }
    }
    const ses::Heightfield hf =
        ses::heightfield_surface(psi, 3.0, 1.0, 1, 0.5);
    EXPECT_EQ(hf.mesh.vertices.size(), 0u);
}

TEST(Heightfield, NonPositiveNormMeansFlatZero) {
    const Grid3D g = small_grid();
    Field3D psi{g};
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            psi(i, j, 0) = 7.0;
        }
    }
    const ses::Heightfield hf = ses::heightfield_surface(psi, 3.0, 0.0, 1);
    for (const ses::Vec3d& v : hf.mesh.vertices) {
        EXPECT_EQ(v.z, 0.0);
    }
}

}  // namespace
