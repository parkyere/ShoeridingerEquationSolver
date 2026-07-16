// RED: marching cubes -- isosurface extraction from a real scalar field on a
// Grid3D, plus the probability_density bridge from the wavefunction.
//
// This is pure geometry generation (no graphics API): the Humble-Object shell
// will only upload the resulting mesh. Oracles:
//  - analytic sphere: vertex radii, total area vs 4 pi R^2, outward normals;
//  - WATERTIGHTNESS (every undirected edge shared by exactly two triangles)
//    on the sphere and on a two-blob field -- a strong topological check on
//    the 256-case tables;
//  - single-hot-corner cube: exactly one triangle at the three edge midpoints.
//
// Convention: the surface encloses the region where field > isovalue (the
// "inside" of a density cloud), and normals point OUTWARD (down-gradient).

#include <core/field.hpp>
import ses.grid;
#include <core/marching_cubes.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <vector>

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Mesh;
using ses::Vec3d;

Grid3D cube_grid(double lo, double hi, int n) {
    return Grid3D{Grid1D{lo, hi, n}, Grid1D{lo, hi, n}, Grid1D{lo, hi, n}};
}

std::vector<double> sample(const Grid3D& g, double (*f)(double, double, double)) {
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                v[static_cast<std::size_t>(g.flat(i, j, k))] =
                    f(g.x.coord(i), g.y.coord(j), g.z.coord(k));
            }
        }
    }
    return v;
}

double sphere_field(double x, double y, double z) {
    return 1.44 - (x * x + y * y + z * z);  // R = 1.2, positive inside
}

double two_blob_field(double x, double y, double z) {
    const double d1 = (x + 1.25) * (x + 1.25) + y * y + z * z;
    const double d2 = (x - 1.25) * (x - 1.25) + (y - 0.3) * (y - 0.3) + z * z;
    return std::exp(-d1) + std::exp(-d2);  // two separate blobs at iso 0.5
}

double vec_len(const Vec3d& v) { return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z); }

double triangle_area(const Vec3d& a, const Vec3d& b, const Vec3d& c) {
    const Vec3d u{b.x - a.x, b.y - a.y, b.z - a.z};
    const Vec3d w{c.x - a.x, c.y - a.y, c.z - a.z};
    const Vec3d cr{u.y * w.z - u.z * w.y, u.z * w.x - u.x * w.z, u.x * w.y - u.y * w.x};
    return 0.5 * vec_len(cr);
}

// Every undirected edge of a closed triangle soup must appear exactly twice.
void expect_watertight(const Mesh& mesh) {
    ASSERT_EQ(mesh.vertices.size() % 3, 0u);
    auto key = [](const Vec3d& v) {
        return std::tuple<std::int64_t, std::int64_t, std::int64_t>{
            std::llround(v.x * 1e7), std::llround(v.y * 1e7), std::llround(v.z * 1e7)};
    };
    using VKey = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
    std::map<std::pair<VKey, VKey>, int> edge_count;
    for (std::size_t t = 0; t + 2 < mesh.vertices.size(); t += 3) {
        for (int e = 0; e < 3; ++e) {
            VKey a = key(mesh.vertices[t + static_cast<std::size_t>(e)]);
            VKey b = key(mesh.vertices[t + static_cast<std::size_t>((e + 1) % 3)]);
            if (b < a) {
                std::swap(a, b);
            }
            ++edge_count[{a, b}];
        }
    }
    for (const auto& [edge, count] : edge_count) {
        ASSERT_EQ(count, 2) << "non-manifold or open edge";
    }
}

TEST(ProbabilityDensity, IsNormSqPerCell) {
    const Grid3D g = cube_grid(0.0, 2.0, 2);
    Field3D psi{g};
    psi(0, 0, 0) = std::complex<double>{3.0, -4.0};
    psi(1, 1, 1) = std::complex<double>{0.0, 2.0};
    const std::vector<double> rho = ses::probability_density(psi);
    ASSERT_EQ(rho.size(), 8u);
    EXPECT_EQ(rho[static_cast<std::size_t>(g.flat(0, 0, 0))], 25.0);
    EXPECT_EQ(rho[static_cast<std::size_t>(g.flat(1, 1, 1))], 4.0);
    EXPECT_EQ(rho[static_cast<std::size_t>(g.flat(1, 0, 0))], 0.0);
}

TEST(MarchingCubes, EmptyAndFullFieldsYieldNoTriangles) {
    const Grid3D g = cube_grid(0.0, 4.0, 8);
    const std::vector<double> zeros(static_cast<std::size_t>(g.size()), 0.0);
    const std::vector<double> ones(static_cast<std::size_t>(g.size()), 1.0);
    EXPECT_TRUE(ses::marching_cubes(zeros, g, 0.5).vertices.empty());
    EXPECT_TRUE(ses::marching_cubes(ones, g, 0.5).vertices.empty());
}

TEST(MarchingCubes, SingleHotCornerYieldsOneTriangle) {
    // One cube (2x2x2 points over [0,1]^3 via [0,2) n=2), corner (0,0,0) hot.
    const Grid3D g = cube_grid(0.0, 2.0, 2);
    std::vector<double> f(8, 0.0);
    f[static_cast<std::size_t>(g.flat(0, 0, 0))] = 1.0;
    const Mesh mesh = ses::marching_cubes(f, g, 0.5);
    ASSERT_EQ(mesh.vertices.size(), 3u);
    ASSERT_EQ(mesh.normals.size(), 3u);
    // Vertices at the midpoints of the three edges leaving the hot corner.
    bool seen_x = false, seen_y = false, seen_z = false;
    for (const Vec3d& v : mesh.vertices) {
        if (std::abs(v.x - 0.5) < 1e-12 && std::abs(v.y) < 1e-12 && std::abs(v.z) < 1e-12) {
            seen_x = true;
        }
        if (std::abs(v.x) < 1e-12 && std::abs(v.y - 0.5) < 1e-12 && std::abs(v.z) < 1e-12) {
            seen_y = true;
        }
        if (std::abs(v.x) < 1e-12 && std::abs(v.y) < 1e-12 && std::abs(v.z - 0.5) < 1e-12) {
            seen_z = true;
        }
    }
    EXPECT_TRUE(seen_x && seen_y && seen_z);
}

TEST(MarchingCubes, SphereVerticesLieOnTheSphere) {
    const Grid3D g = cube_grid(-2.0, 2.0, 32);
    const Mesh mesh = ses::marching_cubes(sample(g, sphere_field), g, 0.0);
    ASSERT_GT(mesh.vertices.size(), 100u);
    for (const Vec3d& v : mesh.vertices) {
        EXPECT_NEAR(vec_len(v), 1.2, 0.02);
    }
}

TEST(MarchingCubes, SphereAreaMatchesAnalytic) {
    const Grid3D g = cube_grid(-2.0, 2.0, 32);
    const Mesh mesh = ses::marching_cubes(sample(g, sphere_field), g, 0.0);
    double area = 0.0;
    for (std::size_t t = 0; t + 2 < mesh.vertices.size(); t += 3) {
        area += triangle_area(mesh.vertices[t], mesh.vertices[t + 1], mesh.vertices[t + 2]);
    }
    const double analytic = 4.0 * 3.14159265358979323846 * 1.44;
    EXPECT_NEAR(area, analytic, 0.02 * analytic);
}

TEST(MarchingCubes, SphereIsWatertight) {
    const Grid3D g = cube_grid(-2.0, 2.0, 32);
    expect_watertight(ses::marching_cubes(sample(g, sphere_field), g, 0.0));
}

TEST(MarchingCubes, TwoBlobFieldIsWatertight) {
    const Grid3D g = cube_grid(-4.0, 4.0, 32);
    const Mesh mesh = ses::marching_cubes(sample(g, two_blob_field), g, 0.5);
    ASSERT_GT(mesh.vertices.size(), 100u);
    expect_watertight(mesh);
}

TEST(MarchingCubes, NormalsPointOutwardAndAreUnit) {
    const Grid3D g = cube_grid(-2.0, 2.0, 32);
    const Mesh mesh = ses::marching_cubes(sample(g, sphere_field), g, 0.0);
    ASSERT_EQ(mesh.normals.size(), mesh.vertices.size());
    for (std::size_t i = 0; i < mesh.vertices.size(); ++i) {
        const Vec3d& v = mesh.vertices[i];
        const Vec3d& n = mesh.normals[i];
        EXPECT_NEAR(vec_len(n), 1.0, 1e-9);
        const double r = vec_len(v);
        const double dot = (n.x * v.x + n.y * v.y + n.z * v.z) / r;
        EXPECT_GT(dot, 0.9);  // radially outward for a sphere
    }
}

}  // namespace
