// RED: exact, unitary rotation of a 3D field about the z-axis.
// The paramagnetic term (B/2) L_z of the magnetic Hamiltonian is the
// generator of z-rotations, so evolving psi under it is a rigid rotation --
// but applied to the ACTUAL wavefunction, not just the display. Done by the
// three-shear Fourier method (each shear is a per-line Fourier phase ramp),
// which is information-preserving: exactly norm-conserving, no interpolation
// blur, valid for |theta| < pi.
//
// Oracles:
//  - norm is conserved to machine precision (each shear is unitary);
//  - an off-axis blob's mean position rotates: <x>,<y> -> R(theta);
//  - a z-symmetric field is invariant;
//  - composition R(a) . R(b) == R(a+b).

#include <core/field.hpp>
import ses.grid;
#include <core/observables.hpp>
#include <core/rotation.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <cmath>

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;

Grid3D cube(double half, int n) {
    const Grid1D a{-half, half, n};
    return Grid3D{a, a, a};
}

// A real Gaussian blob centered at (x0, y0, 0).
Field3D blob(const Grid3D& g, double x0, double y0, double sigma) {
    Field3D f{g};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i) - x0;
                const double y = g.y.coord(j) - y0;
                const double z = g.z.coord(k);
                const double e =
                    std::exp(-(x * x + y * y + z * z) / (2.0 * sigma * sigma));
                f(i, j, k) = std::complex<double>{e, 0.0};
            }
        }
    }
    ses::normalize(f);
    return f;
}

TEST(RotateZ, ConservesNormExactly) {
    const Grid3D g = cube(16.0, 64);
    Field3D f = blob(g, 4.0, 0.0, 1.5);
    const double n0 = ses::norm_sq(f);
    ses::rotate_z(f, 0.6);
    EXPECT_NEAR(ses::norm_sq(f), n0, 1e-11);
}

TEST(RotateZ, MeanPositionRotates) {
    const Grid3D g = cube(16.0, 64);
    Field3D f = blob(g, 4.0, 0.0, 1.5);
    const double theta = 0.6;
    ses::rotate_z(f, theta);
    const ses::Vec3d r = ses::mean_position(f);
    EXPECT_NEAR(r.x, 4.0 * std::cos(theta), 2e-2);
    EXPECT_NEAR(r.y, 4.0 * std::sin(theta), 2e-2);
    EXPECT_NEAR(r.z, 0.0, 1e-6);
}

TEST(RotateZ, ZSymmetricFieldIsInvariant) {
    const Grid3D g = cube(16.0, 64);
    // Depends only on rho and z -> invariant under z-rotation.
    Field3D f{g};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double rho2 =
                    g.x.coord(i) * g.x.coord(i) + g.y.coord(j) * g.y.coord(j);
                const double z = g.z.coord(k);
                f(i, j, k) =
                    std::complex<double>{std::exp(-0.1 * rho2 - 0.1 * z * z), 0.0};
            }
        }
    }
    const Field3D before = f;
    ses::rotate_z(f, 0.5);
    double max_diff = 0.0;
    for (std::size_t i = 0; i < f.data().size(); ++i) {
        max_diff = std::max(max_diff, std::abs(f.data()[i] - before.data()[i]));
    }
    EXPECT_LT(max_diff, 2e-3);
}

// rotate_axis generalizes to rotation about any coordinate axis (for a
// magnetic field along x or y, not just z): three-shear in the plane
// perpendicular to `axis`. rotate_axis(_, 2, _) must equal rotate_z.

TEST(RotateAxis, AgreesWithRotateZForAxisTwo) {
    const Grid3D g = cube(16.0, 64);
    Field3D a = blob(g, 3.0, 1.0, 1.6);
    Field3D b = a;
    ses::rotate_z(a, 0.5);
    ses::rotate_axis(b, 2, 0.5);
    double d = 0.0;
    for (std::size_t i = 0; i < a.data().size(); ++i) {
        d = std::max(d, std::abs(a.data()[i] - b.data()[i]));
    }
    EXPECT_LT(d, 1e-13);
}

TEST(RotateAxis, AboutXRotatesInTheYZPlane) {
    const Grid3D g = cube(16.0, 64);
    Field3D f = blob(g, 0.0, 4.0, 1.5);  // on the +y axis
    const double theta = 0.6;
    const double n0 = ses::norm_sq(f);
    ses::rotate_axis(f, 0, theta);
    EXPECT_NEAR(ses::norm_sq(f), n0, 1e-11);
    const ses::Vec3d r = ses::mean_position(f);
    EXPECT_NEAR(r.x, 0.0, 1e-6);
    EXPECT_NEAR(r.y, 4.0 * std::cos(theta), 2e-2);
    EXPECT_NEAR(r.z, 4.0 * std::sin(theta), 2e-2);
}

TEST(RotateAxis, AboutYConservesNormAndLeavesAYSymmetricFieldInvariant) {
    const Grid3D g = cube(16.0, 64);
    Field3D f{g};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                // depends only on the distance from the y-axis and on y
                const double s2 =
                    g.x.coord(i) * g.x.coord(i) + g.z.coord(k) * g.z.coord(k);
                const double y = g.y.coord(j);
                f(i, j, k) =
                    std::complex<double>{std::exp(-0.1 * s2 - 0.1 * y * y), 0.0};
            }
        }
    }
    const Field3D before = f;
    const double n0 = ses::norm_sq(f);
    ses::rotate_axis(f, 1, 0.5);
    EXPECT_NEAR(ses::norm_sq(f), n0, 1e-11);
    double max_diff = 0.0;
    for (std::size_t i = 0; i < f.data().size(); ++i) {
        max_diff = std::max(max_diff, std::abs(f.data()[i] - before.data()[i]));
    }
    EXPECT_LT(max_diff, 2e-3);
}

TEST(RotateZ, ComposesAdditively) {
    const Grid3D g = cube(16.0, 64);
    Field3D a = blob(g, 3.0, 1.0, 1.6);
    Field3D b = a;
    ses::rotate_z(a, 0.3);
    ses::rotate_z(a, 0.4);
    ses::rotate_z(b, 0.7);
    double max_diff = 0.0;
    for (std::size_t i = 0; i < a.data().size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a.data()[i] - b.data()[i]));
    }
    EXPECT_LT(max_diff, 2e-3);
}

}  // namespace
