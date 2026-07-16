#pragma once

// Trilinear sampling of the complex field, and per-vertex isosurface phase
// colors. Interpolating the COMPLEX value and then taking atan2 keeps
// constant-phase regions exactly constant (amplitude cancels in the ratio).

#include <core/complex.hpp>
#include <core/field.hpp>
import ses.grid;
#include <core/marching_cubes.hpp>
import ses.vec;

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

import ses.colormap;

namespace ses {

namespace sampling_detail {

// Cell index and fractional offset along one axis; the cell is clamped to
// [0, n-2] so the last grid point (t = 1 in the last cell) stays valid.
inline std::pair<int, double> cell_and_t(double u, const Grid1D& axis) noexcept {
    const double s = (u - axis.xmin) / axis.spacing();
    int i = static_cast<int>(std::floor(s));
    i = std::clamp(i, 0, axis.n - 2);
    return {i, s - i};
}

}  // namespace sampling_detail

inline Complex<double> sample_trilinear(const Field3D& f, Vec3d p) noexcept {
    const Grid3D& g = f.grid();
    const auto [i, tx] = sampling_detail::cell_and_t(p.x, g.x);
    const auto [j, ty] = sampling_detail::cell_and_t(p.y, g.y);
    const auto [k, tz] = sampling_detail::cell_and_t(p.z, g.z);

    auto lerp = [](Complex<double> a, Complex<double> b, double t) {
        return a + t * (b - a);  // component-wise on re and im
    };

    const Complex<double> c00 = lerp(f(i, j, k), f(i + 1, j, k), tx);
    const Complex<double> c10 = lerp(f(i, j + 1, k), f(i + 1, j + 1, k), tx);
    const Complex<double> c01 = lerp(f(i, j, k + 1), f(i + 1, j, k + 1), tx);
    const Complex<double> c11 = lerp(f(i, j + 1, k + 1), f(i + 1, j + 1, k + 1), tx);
    return lerp(lerp(c00, c10, ty), lerp(c01, c11, ty), tz);
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
