#pragma once

// Hand-rolled renderer math (purist reinvention boundary: no GLM).
// Conventions (pinned by tests/camera_test.cpp):
//  - Mat4 is COLUMN-MAJOR: element(row r, col c) = m[c*4 + r].
//  - Right-handed view space, camera looks down -Z.
//  - NDC depth [-1, +1] (standard OpenGL clip conventions). The Vulkan
//    renderer applies its own y-flip/depth-remap clip correction.

#include <core/vec.hpp>

#include <cmath>

namespace ses {

struct Mat4 {
    double m[16]{};  // column-major

    static Mat4 identity() {
        Mat4 r;
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1.0;
        return r;
    }

    static Mat4 translation(Vec3d t) {
        Mat4 r = identity();
        r.m[12] = t.x;
        r.m[13] = t.y;
        r.m[14] = t.z;
        return r;
    }

    static Mat4 scale(double s) {
        Mat4 r = identity();
        r.m[0] = r.m[5] = r.m[10] = s;
        return r;
    }
};

inline Mat4 operator*(const Mat4& a, const Mat4& b) {
    Mat4 r;
    for (int c = 0; c < 4; ++c) {
        for (int row = 0; row < 4; ++row) {
            double acc = 0.0;
            for (int k = 0; k < 4; ++k) {
                acc += a.m[k * 4 + row] * b.m[c * 4 + k];
            }
            r.m[c * 4 + row] = acc;
        }
    }
    return r;
}

// Apply to a point (w = 1) with perspective divide.
inline Vec3d transform_point(const Mat4& a, Vec3d p) {
    const double x = a.m[0] * p.x + a.m[4] * p.y + a.m[8] * p.z + a.m[12];
    const double y = a.m[1] * p.x + a.m[5] * p.y + a.m[9] * p.z + a.m[13];
    const double z = a.m[2] * p.x + a.m[6] * p.y + a.m[10] * p.z + a.m[14];
    const double w = a.m[3] * p.x + a.m[7] * p.y + a.m[11] * p.z + a.m[15];
    return {x / w, y / w, z / w};
}

// gluLookAt: right-handed view matrix, forward = -Z.
inline Mat4 look_at(Vec3d eye, Vec3d center, Vec3d up) {
    const Vec3d f = normalized(center - eye);   // forward
    const Vec3d s = normalized(cross(f, up));   // right
    const Vec3d u = cross(s, f);                // true up

    Mat4 r = Mat4::identity();
    r.m[0] = s.x;
    r.m[4] = s.y;
    r.m[8] = s.z;
    r.m[1] = u.x;
    r.m[5] = u.y;
    r.m[9] = u.z;
    r.m[2] = -f.x;
    r.m[6] = -f.y;
    r.m[10] = -f.z;
    r.m[12] = -dot(s, eye);
    r.m[13] = -dot(u, eye);
    r.m[14] = dot(f, eye);
    return r;
}

// gluPerspective: fovy in radians, NDC depth [-1, +1].
inline Mat4 perspective(double fovy, double aspect, double znear, double zfar) {
    const double f = 1.0 / std::tan(fovy / 2.0);
    Mat4 r;
    r.m[0] = f / aspect;
    r.m[5] = f;
    r.m[10] = (zfar + znear) / (znear - zfar);
    r.m[11] = -1.0;
    r.m[14] = 2.0 * zfar * znear / (znear - zfar);
    return r;
}

// Orbit-camera eye position on a sphere around the target.
// azimuth 0, elevation 0 -> on the target's +Z axis; azimuth swings toward
// +X; elevation rises toward +Y.
inline Vec3d orbit_eye(double azimuth, double elevation, double distance, Vec3d target) {
    const double ce = std::cos(elevation);
    return target + distance * Vec3d{ce * std::sin(azimuth), std::sin(elevation),
                                     ce * std::cos(azimuth)};
}

}  // namespace ses