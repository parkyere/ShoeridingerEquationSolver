// RED: UV-sphere mesh generator -- the nucleus marker geometry. (A proton is
// ~1e-5 Bohr, invisible at scene scale; the app draws a symbolic sphere.)
//
// Oracles: every vertex sits exactly on the sphere, every normal is the unit
// outward radial, and the triangle soup is watertight (the same topological
// check that pinned marching cubes).

#include <core/marching_cubes.hpp>
#include <core/sphere.hpp>
#include <core/vec.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>

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

}  // namespace
