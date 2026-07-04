#pragma once

// Volume-rendering math. The GLSL ray marcher mirrors these formulas
// line-by-line; correctness is pinned HERE (tests/volume_test.cpp) because
// shaders cannot be unit-tested.

#include <core/colormap.hpp>
#include <core/vec.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numbers>
#include <vector>

namespace ses {

struct RayHit {
    bool hit{};
    double t_near{};
    double t_far{};
};

// Slab-method ray/AABB intersection. Returns the RAW parameter interval:
// t_near < 0 means the ray starts inside; callers clamp with max(t_near, 0).
inline RayHit ray_box(Vec3d origin, Vec3d dir, Vec3d box_min, Vec3d box_max) {
    double t_near = -std::numeric_limits<double>::infinity();
    double t_far = std::numeric_limits<double>::infinity();

    const double o[3] = {origin.x, origin.y, origin.z};
    const double d[3] = {dir.x, dir.y, dir.z};
    const double lo[3] = {box_min.x, box_min.y, box_min.z};
    const double hi[3] = {box_max.x, box_max.y, box_max.z};

    for (int axis = 0; axis < 3; ++axis) {
        if (d[axis] == 0.0) {
            if (o[axis] < lo[axis] || o[axis] > hi[axis]) {
                return RayHit{false, 0.0, 0.0};  // parallel outside the slab
            }
            continue;  // parallel inside: no constraint from this slab
        }
        double t1 = (lo[axis] - o[axis]) / d[axis];
        double t2 = (hi[axis] - o[axis]) / d[axis];
        if (t1 > t2) {
            std::swap(t1, t2);
        }
        t_near = std::max(t_near, t1);
        t_far = std::min(t_far, t2);
        if (t_near > t_far) {
            return RayHit{false, 0.0, 0.0};
        }
    }
    return RayHit{true, t_near, t_far};
}

// Ray/sphere intersection (dir must be unit length). Raw [t_near, t_far]:
// t_near < 0 when the ray starts inside; tangent rays report t_near == t_far.
inline RayHit ray_sphere(Vec3d origin, Vec3d dir, Vec3d center, double radius) {
    const Vec3d oc = origin - center;
    const double b = dot(oc, dir);
    const double c = dot(oc, oc) - radius * radius;
    const double disc = b * b - c;
    if (disc < 0.0) {
        return RayHit{false, 0.0, 0.0};
    }
    const double s = std::sqrt(disc);
    return RayHit{true, -b - s, -b + s};
}

// Beer-Lambert emission-absorption opacity of one ray-march step.
inline double sample_alpha(double density01, double absorbance, double step) {
    return 1.0 - std::exp(-absorbance * density01 * step);
}

struct Rgba {
    double r{};
    double g{};
    double b{};
    double a{};
};

struct VolumeSample {
    Rgb color;
    double alpha{};
};

// Premultiplied front-to-back accumulation:
//     C += (1 - A) a c,   A += (1 - A) a.
// Early exit once A saturates is a pure optimization (weights vanish).
inline Rgba composite_front_to_back(const std::vector<VolumeSample>& samples) {
    Rgba out{};
    for (const VolumeSample& s : samples) {
        const double w = (1.0 - out.a) * s.alpha;
        out.r += w * s.color.r;
        out.g += w * s.color.g;
        out.b += w * s.color.b;
        out.a += w;
        if (out.a >= 0.999) {
            break;
        }
    }
    return out;
}

// The tested cyclic phase colormap baked into a lookup table (bin centers),
// so the GPU samples verified colors instead of a re-derived formula.
inline std::vector<Rgb> phase_lut(int n) {
    std::vector<Rgb> lut(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        const double theta =
            -std::numbers::pi + 2.0 * std::numbers::pi * (i + 0.5) / n;
        lut[static_cast<std::size_t>(i)] = phase_color(theta);
    }
    return lut;
}

}  // namespace ses
