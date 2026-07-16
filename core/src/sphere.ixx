module;
#include <cmath>
#include <cstddef>
#include <numbers>
#include <vector>
#include <core/marching_cubes.hpp>
export module ses.sphere;
export import ses.vec;


// UV-sphere triangle soup for the nucleus marker. Lattice directions are
// computed once and shared by adjacent triangles (longitude wraps by index),
// so seams and poles weld bitwise -- watertight by construction. Normals are
// the exact unit radials.


export namespace ses {

inline Mesh sphere_mesh(Vec3d center, double radius, int rings, int segments) {
    // Unit directions on a (rings+1) x segments lattice; theta = 0 is the
    // +z pole, theta = pi the -z pole.
    std::vector<Vec3d> dir(static_cast<std::size_t>((rings + 1) * segments));
    for (int i = 0; i <= rings; ++i) {
        const double theta = std::numbers::pi * i / rings;
        const double st = std::sin(theta);
        const double ct = std::cos(theta);
        for (int j = 0; j < segments; ++j) {
            const double phi = 2.0 * std::numbers::pi * j / segments;
            dir[static_cast<std::size_t>(i * segments + j)] =
                Vec3d{st * std::cos(phi), st * std::sin(phi), ct};
        }
    }
    auto at = [&](int i, int j) {
        return dir[static_cast<std::size_t>(i * segments + (j % segments))];
    };

    Mesh m;
    auto emit = [&](Vec3d u) {
        m.vertices.push_back(center + radius * u);
        m.normals.push_back(u);
    };

    for (int j = 0; j < segments; ++j) {
        // top cap: pole row 0 collapses to a single point
        emit(at(0, j));
        emit(at(1, j));
        emit(at(1, j + 1));
        // middle bands
        for (int i = 1; i + 1 < rings; ++i) {
            const Vec3d a = at(i, j);
            const Vec3d b = at(i + 1, j);
            const Vec3d c = at(i + 1, j + 1);
            const Vec3d d = at(i, j + 1);
            emit(a);
            emit(b);
            emit(c);
            emit(a);
            emit(c);
            emit(d);
        }
        // bottom cap: pole row `rings` collapses to a single point
        emit(at(rings - 1, j));
        emit(at(rings, j));
        emit(at(rings - 1, j + 1));
    }
    return m;
}

// Solid arrow (cylinder shaft + cone head) from the origin along `dir`, total
// length `len`; triangle soup for the XYZ gizmo. Normals are unit but not
// necessarily outward: the mesh shader lights with abs(dot(n, view)).
inline Mesh arrow_mesh(Vec3d dir, double len, double shaft_r, double head_r,
                       double head_frac, int segments) {
    const Vec3d d = normalized(dir);
    // Any frame perpendicular to d (helper picked non-parallel to d).
    const Vec3d helper =
        (std::abs(d.x) < 0.9) ? Vec3d{1.0, 0.0, 0.0} : Vec3d{0.0, 1.0, 0.0};
    const Vec3d u = normalized(cross(helper, d));
    const Vec3d v = cross(d, u);  // unit (d, u orthonormal)
    const double shaft_len = len * (1.0 - head_frac);
    const double head_len = len * head_frac;
    const Vec3d neck = shaft_len * d;  // shaft top = cone base
    const Vec3d tip = len * d;
    const Vec3d back = -1.0 * d;

    Mesh m;
    auto emit = [&](Vec3d p, Vec3d n) {
        m.vertices.push_back(p);
        m.normals.push_back(n);
    };
    auto radial = [&](int k) {
        const double a = 2.0 * std::numbers::pi * k / segments;
        return std::cos(a) * u + std::sin(a) * v;
    };

    for (int k = 0; k < segments; ++k) {
        const Vec3d r0 = radial(k);
        const Vec3d r1 = radial(k + 1);
        // shaft side (radial normals)
        const Vec3d b0 = shaft_r * r0;
        const Vec3d b1 = shaft_r * r1;
        const Vec3d t0 = neck + shaft_r * r0;
        const Vec3d t1 = neck + shaft_r * r1;
        emit(b0, r0);
        emit(b1, r1);
        emit(t1, r1);
        emit(b0, r0);
        emit(t1, r1);
        emit(t0, r0);
        // shaft base cap (disk at the origin)
        emit(Vec3d{0.0, 0.0, 0.0}, back);
        emit(b1, back);
        emit(b0, back);
        // head base disk (radius head_r; closes the shaft top and cone bottom)
        const Vec3d h0 = neck + head_r * r0;
        const Vec3d h1 = neck + head_r * r1;
        emit(neck, back);
        emit(h1, back);
        emit(h0, back);
        // cone side (slanted normals)
        const Vec3d n0 = normalized(head_len * r0 + head_r * d);
        const Vec3d n1 = normalized(head_len * r1 + head_r * d);
        emit(h0, n0);
        emit(h1, n1);
        emit(tip, normalized(n0 + n1));
    }
    return m;
}

}  // namespace ses
