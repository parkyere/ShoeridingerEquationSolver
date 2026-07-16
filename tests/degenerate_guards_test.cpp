// Degenerate / adversarial-input guards (REVIEW_BACKLOG latent-correctness).
// These regimes the app does not currently produce, but the helpers must not
// emit NaN / deref past-the-end / spin when handed them.

#include <core/field.hpp>
#include <core/fft.hpp>
import ses.grid;
#include <core/marching_cubes.hpp>
#include <core/observables.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

using ses::Complex;
using ses::Field1D;
using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;

Grid3D cube(int n) {
    const Grid1D a{-4.0, 4.0, n};
    return Grid3D{a, a, a};
}

// normalize() of a zero field must not divide by zero -> NaN.
TEST(DegenerateGuards, NormalizeZeroFieldStaysFinite) {
    Field3D f{cube(8)};  // all zero
    ses::normalize(f);
    for (const Complex<double>& z : f.data()) {
        EXPECT_TRUE(std::isfinite(z.real()));
        EXPECT_TRUE(std::isfinite(z.imag()));
        EXPECT_EQ(z.real(), 0.0);
        EXPECT_EQ(z.imag(), 0.0);
    }
}

// Every observable divides by the total weight -> 0/0 on a zero field.
TEST(DegenerateGuards, Observables3ZeroFieldFinite) {
    const Grid3D g = cube(8);
    const Field3D f{g};  // all zero
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const ses::Vec3d mp = ses::mean_position(f);
    const ses::Vec3d sp = ses::sigma_position(f);
    const ses::Vec3d mk = ses::mean_momentum(f);
    const double e = ses::mean_energy(f, v);
    for (double c : {mp.x, mp.y, mp.z, sp.x, sp.y, sp.z, mk.x, mk.y, mk.z, e}) {
        EXPECT_TRUE(std::isfinite(c));
    }
}

TEST(DegenerateGuards, Observables1ZeroFieldFinite) {
    const Grid1D a{-4.0, 4.0, 16};
    const Field1D f{a};  // all zero
    const std::vector<double> v(static_cast<std::size_t>(a.n), 0.0);
    EXPECT_TRUE(std::isfinite(ses::mean_position(f)));
    EXPECT_TRUE(std::isfinite(ses::sigma_x(f)));
    EXPECT_TRUE(std::isfinite(ses::mean_momentum(f)));
    EXPECT_TRUE(std::isfinite(ses::mean_energy(f, v)));
}

// A single occupied cell: <x^2> - <x>^2 rounds to a tiny negative -> the
// variance clamp must keep sigma real (not NaN) and non-negative.
TEST(DegenerateGuards, SigmaPositionSingleCellNonNegative) {
    const Grid3D g = cube(8);
    Field3D f{g};
    f(5, 2, 6) = Complex<double>{1.0, 0.0};
    const ses::Vec3d sp = ses::sigma_position(f);
    for (double c : {sp.x, sp.y, sp.z}) {
        EXPECT_TRUE(std::isfinite(c));
        EXPECT_GE(c, 0.0);
    }
}

// marching_cubes_at_fraction: *max_element(end()) is UB on an empty field,
// and a non-positive peak has no isosurface.
TEST(DegenerateGuards, MarchingCubesEmptyReturnsEmpty) {
    const Grid3D g = cube(8);
    const ses::Mesh m = ses::marching_cubes_at_fraction({}, g, 0.25);
    EXPECT_TRUE(m.vertices.empty());
}

TEST(DegenerateGuards, MarchingCubesZeroFieldReturnsEmpty) {
    const Grid3D g = cube(8);
    const std::vector<double> rho(static_cast<std::size_t>(g.size()), 0.0);
    const ses::Mesh m = ses::marching_cubes_at_fraction(rho, g, 0.25);
    EXPECT_TRUE(m.vertices.empty());
}

// A non-power-of-two length must fail loudly (not spin / garbage under NDEBUG).
TEST(DegenerateGuards, FftNonPowerOfTwoThrows) {
    std::vector<Complex<double>> v48(48);
    EXPECT_THROW(ses::fft(v48), std::invalid_argument);
    std::vector<Complex<double>> v3(3);
    EXPECT_THROW(ses::fft(v3), std::invalid_argument);
    // Powers of two still fine.
    std::vector<Complex<double>> v16(16);
    EXPECT_NO_THROW(ses::fft(v16));
}

}  // namespace
