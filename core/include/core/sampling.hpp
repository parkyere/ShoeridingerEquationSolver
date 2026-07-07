#pragma once

// Trilinear sampling of the complex field at arbitrary positions, and the
// per-vertex phase colors that paint momentum stripes onto an isosurface.
//
// Interpolating the COMPLEX value (re and im independently) and then taking
// atan2 keeps constant-phase regions exactly constant: a real amplitude
// scales both components equally and cancels in the ratio.

#include <core/colormap.hpp>
#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/marching_cubes.hpp>
#include <core/vec.hpp>

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

namespace ses {

namespace sampling_detail {

// Cell index and fractional offset along one axis; the cell is clamped to
// [0, n-2] so the last grid point (t = 1 in the last cell) stays valid.
inline std::pair<int, double> cell_and_t(double u, const Grid1D& axis) {
    const double s = (u - axis.xmin) / axis.spacing();
    int i = static_cast<int>(std::floor(s));
    i = std::clamp(i, 0, axis.n - 2);
    return {i, s - i};
}

}  // namespace sampling_detail

inline Complex<double> sample_trilinear(const Field3D& f, Vec3d p) {
    const Grid3D& g = f.grid();
    const auto [i, tx] = sampling_detail::cell_and_t(p.x, g.x);
    const auto [j, ty] = sampling_detail::cell_and_t(p.y, g.y);
    const auto [k, tz] = sampling_detail::cell_and_t(p.z, g.z);

    auto lerp = [](Complex<double> a, Complex<double> b, double t) {
        return a + t * (b - a);  // component-wise: same math as before
    };

    const Complex<double> c00 = lerp(f(i, j, k), f(i + 1, j, k), tx);
    const Complex<double> c10 = lerp(f(i, j + 1, k), f(i + 1, j + 1, k), tx);
    const Complex<double> c01 = lerp(f(i, j, k + 1), f(i + 1, j, k + 1), tx);
    const Complex<double> c11 = lerp(f(i, j + 1, k + 1), f(i + 1, j + 1, k + 1), tx);
    return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
}

// Rigid rotation of the complex field by `angle` radians about a coordinate
// axis (0 = x, 1 = y, 2 = z): out(r) = f(R(-angle) r), trilinearly sampled.
// This is exactly exp(-i angle L_axis) f -- the Larmor precession of the cloud
// in a magnetic field along that axis (the central potential commutes with the
// rotation, so it factors cleanly out of the atomic evolution). Points that
// rotate outside the box clamp to the edge value (the cloud is ~0 there).
inline Field3D rotate_field(const Field3D& f, int axis, double angle) {
    const Grid3D& g = f.grid();
    Field3D out{g};
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                Vec3d src;
                if (axis == 0) {  // about x: rotate the (y, z) plane by -angle
                    src = Vec3d{x, y * c + z * s, -y * s + z * c};
                } else if (axis == 1) {  // about y: rotate the (z, x) plane
                    src = Vec3d{x * c - z * s, y, x * s + z * c};
                } else {  // about z: rotate the (x, y) plane
                    src = Vec3d{x * c + y * s, -x * s + y * c, z};
                }
                out(i, j, k) = sample_trilinear(f, src);
            }
        }
    }
    return out;
}

// One cyclic phase color per mesh vertex: arg(psi) sampled at the vertex.
inline std::vector<Rgb> phase_colors(const Mesh& mesh, const Field3D& psi) {
    std::vector<Rgb> colors;
    colors.reserve(mesh.vertices.size());
    for (const Vec3d& v : mesh.vertices) {
        const Complex<double> s = sample_trilinear(psi, v);
        colors.push_back(phase_color(std::atan2(s.imag(), s.real())));
    }
    return colors;
}

}  // namespace ses
