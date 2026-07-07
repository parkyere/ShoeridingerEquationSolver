// RED: trilinear sampling of the complex field at arbitrary points, and
// per-vertex phase colors for a mesh.
//
// Marching-cubes vertices sit BETWEEN grid points, so painting phase on the
// isosurface needs interpolation of psi at arbitrary positions.
//
// Sharp oracles:
//  - trilinear interpolation reproduces any function linear in x,y,z EXACTLY;
//  - for psi = A(r) e^{i phi0} with real A > 0, the interpolated amplitude
//    cancels inside atan2, so every vertex phase is EXACTLY phi0;
//  - for a unit-amplitude plane wave e^{i k x} the interpolated phase tracks
//    k*x to O((kh)^2) between grid points (tolerance-checked).

#include <core/colormap.hpp>
#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/marching_cubes.hpp>
#include <core/observables.hpp>
#include <core/sampling.hpp>
#include <core/vec.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using ses::Complex;
using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Mesh;
using ses::Rgb;
using ses::Vec3d;

// A p-like test field: coordinate component `comp` times a Gaussian envelope,
// normalized. p_like(.,0,.) ~ p_x, (.,1,.) ~ p_y.
Field3D p_like(const Grid3D& g, int comp, double sigma) {
    Field3D f{g};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double c[3] = {g.x.coord(i), g.y.coord(j), g.z.coord(k)};
                const double r2 = c[0] * c[0] + c[1] * c[1] + c[2] * c[2];
                f(i, j, k) = Complex<double>{
                    c[comp] * std::exp(-r2 / (2.0 * sigma * sigma)), 0.0};
            }
        }
    }
    ses::normalize(f);
    return f;
}

TEST(RotateField, ByZeroIsIdentityInTheInterior) {
    const Grid1D ax{-8.0, 8.0, 48};
    const Grid3D g{ax, ax, ax};
    const Field3D f = p_like(g, 0, 2.0);
    const Field3D r = ses::rotate_field(f, 2, 0.0);
    for (int i = 4; i < g.x.n - 4; i += 9) {
        EXPECT_NEAR(r(i, 12, 20).real(), f(i, 12, 20).real(), 1e-12);
    }
}

TEST(RotateField, SphericalFieldIsInvariantUnderZRotation) {
    const Grid1D ax{-8.0, 8.0, 48};
    const Grid3D g{ax, ax, ax};
    Field3D f{g};  // spherically symmetric Gaussian
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i), y = g.y.coord(j), z = g.z.coord(k);
                f(i, j, k) = Complex<double>{std::exp(-(x * x + y * y + z * z) / 8.0), 0.0};
            }
        }
    }
    ses::normalize(f);
    const Field3D r = ses::rotate_field(f, 2, 0.7);
    EXPECT_GT(std::abs(ses::inner_product(f, r)), 0.99);  // rotation-invariant
}

TEST(RotateField, RotatesPxIntoPyAboutZ) {
    const Grid1D ax{-8.0, 8.0, 48};
    const Grid3D g{ax, ax, ax};
    const Field3D px = p_like(g, 0, 2.0);
    const Field3D py = p_like(g, 1, 2.0);
    Field3D r = ses::rotate_field(px, 2, 1.5707963267948966);  // +90 deg about z
    ses::normalize(r);
    EXPECT_GT(std::abs(ses::inner_product(py, r)), 0.9);   // aligned with p_y
    EXPECT_LT(std::abs(ses::inner_product(px, r)), 0.15);  // no longer p_x
}

Grid3D integer_grid() {
    const Grid1D axis{0.0, 8.0, 8};  // coords 0..7 exactly
    return Grid3D{axis, axis, axis};
}

TEST(TrilinearSample, ExactAtGridPoints) {
    Field3D f{integer_grid()};
    f(2, 3, 4) = Complex<double>{1.5, -2.5};
    const Complex<double> s = ses::sample_trilinear(f, Vec3d{2.0, 3.0, 4.0});
    EXPECT_EQ(s.real(), 1.5);
    EXPECT_EQ(s.imag(), -2.5);
}

TEST(TrilinearSample, MidpointOfEdgeAveragesEndpoints) {
    Field3D f{integer_grid()};
    f(2, 3, 4) = Complex<double>{1.0, 4.0};
    f(3, 3, 4) = Complex<double>{3.0, -2.0};
    const Complex<double> s = ses::sample_trilinear(f, Vec3d{2.5, 3.0, 4.0});
    EXPECT_DOUBLE_EQ(s.real(), 2.0);
    EXPECT_DOUBLE_EQ(s.imag(), 1.0);
}

TEST(TrilinearSample, ReproducesLinearFieldsExactly) {
    const Grid3D g = integer_grid();
    Field3D f{g};
    auto re = [](double x, double y, double z) { return 2.0 * x - 3.0 * y + z + 5.0; };
    auto im = [](double x, double y, double z) { return -x + 0.5 * y + 2.0 * z - 1.0; };
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i), y = g.y.coord(j), z = g.z.coord(k);
                f(i, j, k) = Complex<double>{re(x, y, z), im(x, y, z)};
            }
        }
    }
    const Vec3d probes[] = {{1.3, 4.7, 2.2}, {0.1, 0.9, 6.3}, {5.999, 3.5, 0.5}};
    for (const Vec3d& p : probes) {
        const Complex<double> s = ses::sample_trilinear(f, p);
        EXPECT_NEAR(s.real(), re(p.x, p.y, p.z), 1e-12);
        EXPECT_NEAR(s.imag(), im(p.x, p.y, p.z), 1e-12);
    }
}

TEST(TrilinearSample, WorksAtTheLastGridPoint) {
    Field3D f{integer_grid()};
    f(7, 7, 7) = Complex<double>{9.0, -9.0};
    const Complex<double> s = ses::sample_trilinear(f, Vec3d{7.0, 7.0, 7.0});
    EXPECT_DOUBLE_EQ(s.real(), 9.0);
    EXPECT_DOUBLE_EQ(s.imag(), -9.0);
}

TEST(PhaseColors, UniformPhaseGivesExactlyUniformColor) {
    // psi = A(r) e^{i 0.7}, A real positive Gaussian: interpolation scales
    // the amplitude only, so atan2 recovers 0.7 exactly at every vertex.
    const Grid1D axis{-8.0, 8.0, 16};
    const Grid3D g{axis, axis, axis};
    Field3D psi{g};
    const double phi0 = 0.7;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i), y = g.y.coord(j), z = g.z.coord(k);
                const double a = std::exp(-(x * x + y * y + z * z) / 8.0);
                psi(i, j, k) = Complex<double>{a * std::cos(phi0), a * std::sin(phi0)};
            }
        }
    }
    const Mesh mesh =
        ses::marching_cubes_at_fraction(ses::probability_density(psi), g, 0.3);
    ASSERT_GT(mesh.vertices.size(), 0u);

    const std::vector<Rgb> colors = ses::phase_colors(mesh, psi);
    ASSERT_EQ(colors.size(), mesh.vertices.size());
    const Rgb expected = ses::phase_color(phi0);
    for (const Rgb& c : colors) {
        EXPECT_NEAR(c.r, expected.r, 1e-9);
        EXPECT_NEAR(c.g, expected.g, 1e-9);
        EXPECT_NEAR(c.b, expected.b, 1e-9);
    }
}

TEST(PhaseColors, PlaneWavePhaseTracksKx) {
    // Unit-amplitude e^{i k x}: between grid points the interpolated phase
    // deviates from k*x only at O((kh)^2) (~1e-3 rad here).
    const Grid1D axis{-8.0, 8.0, 16};  // h = 1
    const Grid3D g{axis, axis, axis};
    Field3D psi{g};
    const double kx = 0.3;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                psi(i, j, k) = Complex<double>{std::cos(kx * x), std::sin(kx * x)};
            }
        }
    }
    // Hand-built probe mesh: interior vertices with known x (normals unused).
    Mesh mesh;
    mesh.vertices = {Vec3d{0.5, 0.25, -1.75}, Vec3d{-3.3, 2.0, 4.4}, Vec3d{5.6, -6.0, 0.0}};
    mesh.normals.resize(mesh.vertices.size());

    const std::vector<Rgb> colors = ses::phase_colors(mesh, psi);
    ASSERT_EQ(colors.size(), mesh.vertices.size());
    for (std::size_t i = 0; i < colors.size(); ++i) {
        const double phase = std::remainder(kx * mesh.vertices[i].x,
                                            2.0 * 3.14159265358979323846);
        const Rgb expected = ses::phase_color(phase);
        EXPECT_NEAR(colors[i].r, expected.r, 0.01);
        EXPECT_NEAR(colors[i].g, expected.g, 0.01);
        EXPECT_NEAR(colors[i].b, expected.b, 0.01);
    }
}

}  // namespace
