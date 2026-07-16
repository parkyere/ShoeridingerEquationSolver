// RED: UV-sphere mesh generator -- the nucleus marker geometry. (A proton is
// ~1e-5 Bohr, invisible at scene scale; the app draws a symbolic sphere.)
//
// Oracles: every vertex sits exactly on the sphere, every normal is the unit
// outward radial, and the triangle soup is watertight (the same topological
// check that pinned marching cubes).

#include <core/marching_cubes.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
import ses.sphere;

namespace {

using ses::Mesh;
using ses::Vec3d;

const Vec3d kCenter{1.0, -2.0, 3.0};
constexpr double kRadius = 0.35;

Mesh make_sphere() { return ses::sphere_mesh(kCenter, kRadius, 12, 16); }

TEST(SphereMesh, IsANonEmptyTriangleSoup) {
    const Mesh m = make_sphere();
    ASSERT_GT(m.vertices.size(), 0u);
    EXPECT_EQ(m.vertices.size() % 3, 0u);
    EXPECT_EQ(m.normals.size(), m.vertices.size());
}

TEST(SphereMesh, VerticesLieExactlyOnTheSphere) {
    const Mesh m = make_sphere();
    for (const Vec3d& v : m.vertices) {
        const Vec3d d{v.x - kCenter.x, v.y - kCenter.y, v.z - kCenter.z};
        EXPECT_NEAR(std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z), kRadius, 1e-12);
    }
}

TEST(SphereMesh, NormalsAreUnitOutwardRadials) {
    const Mesh m = make_sphere();
    for (std::size_t i = 0; i < m.vertices.size(); ++i) {
        const Vec3d& v = m.vertices[i];
        const Vec3d& n = m.normals[i];
        EXPECT_NEAR(std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z), 1.0, 1e-12);
        const Vec3d r{(v.x - kCenter.x) / kRadius, (v.y - kCenter.y) / kRadius,
                      (v.z - kCenter.z) / kRadius};
        EXPECT_NEAR(n.x, r.x, 1e-12);
        EXPECT_NEAR(n.y, r.y, 1e-12);
        EXPECT_NEAR(n.z, r.z, 1e-12);
    }
}

TEST(SphereMesh, IsWatertight) {
    const Mesh m = make_sphere();
    auto key = [](const Vec3d& v) {
        return std::tuple<std::int64_t, std::int64_t, std::int64_t>{
            std::llround(v.x * 1e9), std::llround(v.y * 1e9), std::llround(v.z * 1e9)};
    };
    using VKey = std::tuple<std::int64_t, std::int64_t, std::int64_t>;
    std::map<std::pair<VKey, VKey>, int> edge_count;
    for (std::size_t t = 0; t + 2 < m.vertices.size(); t += 3) {
        for (int e = 0; e < 3; ++e) {
            VKey a = key(m.vertices[t + static_cast<std::size_t>(e)]);
            VKey b = key(m.vertices[t + static_cast<std::size_t>((e + 1) % 3)]);
            if (b < a) {
                std::swap(a, b);
            }
            ++edge_count[{a, b}];
        }
    }
    for (const auto& [edge, count] : edge_count) {
        ASSERT_EQ(count, 2) << "open or non-manifold edge";
    }
}

// ---- arrow_mesh: the orientation-gizmo geometry (cylinder shaft + cone) ----

TEST(ArrowMesh, IsANonEmptyTriangleSoupWithUnitNormals) {
    const Mesh a = ses::arrow_mesh(Vec3d{0.0, 0.0, 1.0}, 1.0, 0.05, 0.12, 0.3, 12);
    ASSERT_GT(a.vertices.size(), 0u);
    EXPECT_EQ(a.vertices.size() % 3, 0u);
    EXPECT_EQ(a.normals.size(), a.vertices.size());
    for (const Vec3d& n : a.normals) {
        EXPECT_NEAR(std::sqrt(n.x * n.x + n.y * n.y + n.z * n.z), 1.0, 1e-9);
    }
}

TEST(ArrowMesh, TipReachesTheLengthAndStaysWithinTheHead) {
    const double len = 1.0, head_r = 0.12, shaft_r = 0.05;
    const Mesh a = ses::arrow_mesh(Vec3d{0.0, 0.0, 1.0}, len, shaft_r, head_r, 0.3, 16);
    double max_z = -1e9, min_z = 1e9, max_r = 0.0;
    for (const Vec3d& v : a.vertices) {
        max_z = std::max(max_z, v.z);
        min_z = std::min(min_z, v.z);
        max_r = std::max(max_r, std::sqrt(v.x * v.x + v.y * v.y));
    }
    EXPECT_NEAR(max_z, len, 1e-9);    // apex sits at z = length
    EXPECT_NEAR(min_z, 0.0, 1e-9);    // base at the origin
    EXPECT_LE(max_r, head_r + 1e-9);  // never wider than the head
    EXPECT_GT(max_r, shaft_r);        // the head flares past the shaft
}

TEST(ArrowMesh, OrientsAlongAnArbitraryAxis) {
    const Mesh a = ses::arrow_mesh(Vec3d{1.0, 0.0, 0.0}, 2.0, 0.05, 0.12, 0.3, 8);
    double max_x = -1e9;
    for (const Vec3d& v : a.vertices) {
        max_x = std::max(max_x, v.x);
    }
    EXPECT_NEAR(max_x, 2.0, 1e-9);  // tip along +x at length 2
}

}  // namespace
