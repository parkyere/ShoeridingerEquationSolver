// RED: 3D dynamics -- the split-operator propagator and imaginary-time
// relaxation lifted to Field3D.
//
// Oracles:
//  - free ANISOTROPIC packet: per-axis center drift r0 + k0 t and per-axis
//    dispersion sigma_i(t) = sigma_i sqrt(1 + (t/(2 sigma_i^2))^2). V = 0
//    makes Fourier propagation time-exact, so dt can be large (2 steps).
//    Distinct per-axis sigma/k0 catch any axis transposition.
//  - 3D isotropic harmonic ground state via imaginary time:
//    E0 = 3 omega / 2, per-axis sigma = 1/sqrt(2 omega), from a deliberately
//    wrong (offset, isotropic-width) initial packet.

#include <core/field.hpp>
import ses.grid;
#include <core/imaginary_time.hpp>
#include <core/observables.hpp>
#include <core/potential.hpp>
#include <core/propagator.hpp>
import ses.vec;
#include <core/wavepacket.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

Grid3D cube(double lo, double hi, int n) {
    return Grid3D{Grid1D{lo, hi, n}, Grid1D{lo, hi, n}, Grid1D{lo, hi, n}};
}

TEST(SplitOperator3, FreePacketDispersesAnisotropically) {
    const Grid3D grid = cube(-12.0, 12.0, 64);
    const Vec3d r0{-3.0, 1.0, 0.0};
    const Vec3d sigma0{0.8, 1.0, 1.2};
    const Vec3d k0{1.5, 0.0, -1.0};
    Field3D psi = ses::gaussian_wavepacket(grid, r0, sigma0, k0);

    const std::vector<double> v(static_cast<std::size_t>(grid.size()), 0.0);
    const double dt = 1.0;  // large: V=0 is time-exact
    ses::SplitOperator3D prop{grid, v, dt};
    prop.step(psi, 2);  // T = 2

    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);

    const double t = 2.0;
    const Vec3d r = ses::mean_position(psi);
    EXPECT_NEAR(r.x, r0.x + k0.x * t, 1e-6);
    EXPECT_NEAR(r.y, r0.y + k0.y * t, 1e-6);
    EXPECT_NEAR(r.z, r0.z + k0.z * t, 1e-6);

    auto spread = [t](double s0) {
        const double q = t / (2.0 * s0 * s0);
        return s0 * std::sqrt(1.0 + q * q);
    };
    const Vec3d s = ses::sigma_position(psi);
    EXPECT_NEAR(s.x, spread(sigma0.x), 1e-6);
    EXPECT_NEAR(s.y, spread(sigma0.y), 1e-6);
    EXPECT_NEAR(s.z, spread(sigma0.z), 1e-6);

    const Vec3d p = ses::mean_momentum(psi);
    EXPECT_NEAR(p.x, k0.x, 1e-8);
    EXPECT_NEAR(p.y, k0.y, 1e-8);
    EXPECT_NEAR(p.z, k0.z, 1e-8);
}

TEST(ImaginaryTime3, FindsIsotropicHarmonicGroundState) {
    // omega = 2: E0 = 3 omega/2 = 3.0, per-axis sigma = 1/sqrt(2 omega) = 0.5.
    const double omega = 2.0;
    const Grid3D grid = cube(-4.0, 4.0, 32);
    const std::vector<double> v = ses::harmonic_potential(grid, omega, Vec3d{});

    Field3D psi = ses::gaussian_wavepacket(grid, Vec3d{0.5, -0.5, 0.3},
                                           Vec3d{1.0, 1.0, 1.0}, Vec3d{});
    ses::ImaginaryTimePropagator3D relaxer{grid, v, 0.01};
    relaxer.relax(psi, 800);  // tau = 8, gap = omega -> residual ~ e^-16

    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
    EXPECT_NEAR(ses::mean_energy(psi, v), 1.5 * omega, 1e-4);

    const Vec3d s = ses::sigma_position(psi);
    const double s_gs = 1.0 / std::sqrt(2.0 * omega);
    EXPECT_NEAR(s.x, s_gs, 1e-3);
    EXPECT_NEAR(s.y, s_gs, 1e-3);
    EXPECT_NEAR(s.z, s_gs, 1e-3);

    const Vec3d r = ses::mean_position(psi);
    EXPECT_NEAR(r.x, 0.0, 1e-6);
    EXPECT_NEAR(r.y, 0.0, 1e-6);
    EXPECT_NEAR(r.z, 0.0, 1e-6);
}

}  // namespace
