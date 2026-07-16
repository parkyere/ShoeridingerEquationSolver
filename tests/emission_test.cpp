// RED: semiclassical emission from the oscillating dipole.
// A superposition of energy levels has an oscillating charge density, hence a
// time-varying dipole d = -<r> that radiates. By Ehrenfest the dipole
// acceleration is the mean force,
//     d_ddot = -<r_ddot> = <grad V>,
// and the Larmor formula gives the radiated power (atomic units, 1/c^3 =
// alpha^3):
//     P = (2/3) alpha^3 |d_ddot|^2.
// This captures the COHERENT emission of superpositions (it is 0 for a pure
// eigenstate, whose density is static -- spontaneous decay of an eigenstate is
// QED, handled separately by the Einstein-A jumps).
//
// Oracles:
//  - <grad V> = w^2 <r> exactly for a harmonic V = 1/2 w^2 r^2 (grad V = w^2 r);
//  - <grad V> = 0 for a free particle and for a symmetric cloud in a central V;
//  - Larmor power: exact (2/3) alpha^3 factor and quadratic scaling.

#include <core/decay.hpp>
#include <core/field.hpp>
import ses.grid;
#include <core/observables.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <cmath>
import ses.wavepacket;
import ses.potential;

import ses.emission;

namespace {

using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

Grid3D cube(double half, int n) {
    const Grid1D a{-half, half, n};
    return Grid3D{a, a, a};
}

TEST(MeanPotentialGradient, HarmonicGivesOmegaSquaredMeanPosition) {
    const Grid3D g = cube(12.0, 64);
    const double w = 1.3;
    const std::vector<double> v = ses::harmonic_potential(g, w, Vec3d{});
    const ses::Field3D psi = ses::gaussian_wavepacket(
        g, Vec3d{2.0, 1.0, -0.5}, Vec3d{1.4, 1.4, 1.4}, Vec3d{});
    const Vec3d grad = ses::mean_potential_gradient(psi, v, g);
    const Vec3d r = ses::mean_position(psi);
    EXPECT_NEAR(grad.x, w * w * r.x, 1e-3);
    EXPECT_NEAR(grad.y, w * w * r.y, 1e-3);
    EXPECT_NEAR(grad.z, w * w * r.z, 1e-3);
}

TEST(MeanPotentialGradient, FreeParticleIsZero) {
    const Grid3D g = cube(10.0, 32);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const ses::Field3D psi = ses::gaussian_wavepacket(
        g, Vec3d{1.0, 0.0, 0.0}, Vec3d{1.5, 1.5, 1.5}, Vec3d{0.0, 0.4, 0.0});
    const Vec3d grad = ses::mean_potential_gradient(psi, v, g);
    EXPECT_NEAR(grad.x, 0.0, 1e-12);
    EXPECT_NEAR(grad.y, 0.0, 1e-12);
    EXPECT_NEAR(grad.z, 0.0, 1e-12);
}

TEST(MeanPotentialGradient, SymmetricCloudInCentralPotentialIsZero) {
    const Grid3D g = cube(12.0, 48);
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, Vec3d{});
    const ses::Field3D psi =
        ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{2.0, 2.0, 2.0}, Vec3d{});
    const Vec3d grad = ses::mean_potential_gradient(psi, v, g);
    EXPECT_NEAR(grad.x, 0.0, 1e-4);
    EXPECT_NEAR(grad.y, 0.0, 1e-4);
    EXPECT_NEAR(grad.z, 0.0, 1e-4);
}

TEST(LarmorPower, ExactFactorAndQuadraticScaling) {
    const double a3 = std::pow(ses::kFineStructureConstant, 3.0);
    EXPECT_DOUBLE_EQ(ses::larmor_power(Vec3d{2.0, 0.0, 0.0}),
                     (2.0 / 3.0) * a3 * 4.0);
    // isotropic: only |a|^2 matters.
    EXPECT_DOUBLE_EQ(ses::larmor_power(Vec3d{1.0, 2.0, 2.0}),
                     (2.0 / 3.0) * a3 * 9.0);
    // quadratic: doubling the acceleration quadruples the power.
    EXPECT_DOUBLE_EQ(ses::larmor_power(Vec3d{2.0, 0.0, 0.0}),
                     4.0 * ses::larmor_power(Vec3d{1.0, 0.0, 0.0}));
}

}  // namespace
