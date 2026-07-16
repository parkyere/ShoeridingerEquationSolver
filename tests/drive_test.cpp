// RED: time-dependent dipole drive -- the laser term
//     V_drive(r, t) = amplitude * (axis . r) * cos(omega t)
// entering the Strang step as scalar-coefficient half-kicks around the
// static factors (diagonal factors commute, so the tested static tables
// stay untouched):
//     kick(t) halfV FFT kinetic IFFT halfV kick(t + dt).
//
// Oracles:
//  - amplitude 0 reproduces the static propagator BITWISE (cis(0) = (1,0));
//  - omega 0 is a constant force: Ehrenfest gives <p_z> = -E0 t exactly and
//    <z> = -E0 t^2 / 2 (linear potential: no force-gradient error);
//  - COHERENT LADDER: a resonantly driven harmonic trap is NOT a two-level
//    Rabi system (all gaps are equal!) -- the exact solution is a coherent
//    state with |alpha(t)| = E0 t / (2 sqrt(2 w0)):
//      <z>(t)  = -(E0 / 2 w0) t sin(w0 t)        (Ehrenfest, exact),
//      P_n     = e^{-|a|^2} |a|^{2n} / n!         (Poisson populations);
//    (true two-level Rabi needs the ANHARMONIC soft-Coulomb spectrum);
//  - SELECTION RULE: the z-polarized drive leaves the y-flavored state
//    empty (<1_y|z|0> = 0 by symmetry) -- exact regardless of the ladder.

#include <core/complex.hpp>
#include <core/drive.hpp>
#include <core/field.hpp>
import ses.grid;
#include <core/observables.hpp>
#include <core/potential.hpp>
#include <core/propagator.hpp>
import ses.vec;
#include <core/wavepacket.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>
#include <vector>

namespace {

using ses::Complex;
using ses::DipoleDrive;
using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

Grid3D cube(double lo, double hi, int n) {
    return Grid3D{Grid1D{lo, hi, n}, Grid1D{lo, hi, n}, Grid1D{lo, hi, n}};
}

double population(const Field3D& state, const Field3D& psi) {
    const Complex<double> ip = ses::inner_product(state, psi);
    return ses::norm_sq(ip);
}

TEST(DipoleDrive, ZeroAmplitudeMatchesStaticBitwise) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const std::vector<double> v = ses::harmonic_potential(g, 1.0, Vec3d{});
    const ses::SplitOperator3D prop{g, v, 0.02};
    const Field3D psi0 = ses::gaussian_wavepacket(g, Vec3d{1.0, 0.0, 0.0},
                                                  Vec3d{1.0, 1.0, 1.0}, Vec3d{});

    Field3D driven = psi0;
    ses::driven_step(driven, prop, DipoleDrive{Vec3d{0.0, 0.0, 1.0}, 0.0, 0.5}, 0.0, 10);
    Field3D fixed = psi0;
    prop.step(fixed, 10);

    double max_diff = 0.0;
    for (std::size_t i = 0; i < fixed.data().size(); ++i) {
        max_diff = std::max(max_diff,
                            std::abs(driven.data()[i].real() - fixed.data()[i].real()));
        max_diff = std::max(max_diff,
                            std::abs(driven.data()[i].imag() - fixed.data()[i].imag()));
    }
    EXPECT_EQ(max_diff, 0.0);
}

TEST(DipoleDrive, ConstantFieldObeysEhrenfest) {
    // Free particle + V = E0 z (omega = 0): force -E0 along z, so
    // <p_z>(t) = -E0 t and <z>(t) = -E0 t^2 / 2 (packet far from the wrap).
    const Grid3D g = cube(-16.0, 16.0, 32);
    const std::vector<double> zero_v(static_cast<std::size_t>(g.size()), 0.0);
    const double dt = 0.04;
    const ses::SplitOperator3D prop{g, zero_v, dt};
    Field3D psi = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{1.5, 1.5, 1.5}, Vec3d{});

    const double e0 = 0.05;
    const int steps = 50;  // T = 2
    ses::driven_step(psi, prop, DipoleDrive{Vec3d{0.0, 0.0, 1.0}, e0, 0.0}, 0.0, steps);

    const double t = steps * dt;
    EXPECT_NEAR(ses::mean_momentum(psi).z, -e0 * t, 1e-8);
    EXPECT_NEAR(ses::mean_position(psi).z, -0.5 * e0 * t * t, 1e-6);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(DipoleDrive, ResonantCoherentLadderAndSelectionRule) {
    // Harmonic trap w0 = 1 driven z-polarized at resonance: the exact result
    // is a coherent state (equal level spacing means the drive climbs the
    // whole ladder -- there is NO two-level Rabi here).
    const double w0 = 1.0;
    const Grid3D g = cube(-8.0, 8.0, 32);
    const std::vector<double> v = ses::harmonic_potential(g, w0, Vec3d{});
    const double dt = 0.05;
    const ses::SplitOperator3D prop{g, v, dt};

    Field3D ground{g};
    Field3D excited_z{g};
    Field3D excited_y{g};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                const double env = std::exp(-0.5 * w0 * (x * x + y * y + z * z));
                ground(i, j, k) = Complex<double>{env, 0.0};
                excited_z(i, j, k) = Complex<double>{z * env, 0.0};
                excited_y(i, j, k) = Complex<double>{y * env, 0.0};
            }
        }
    }
    ses::normalize(ground);
    ses::normalize(excited_z);
    ses::normalize(excited_y);

    const double e0 = 0.2;
    const int steps = 283;  // t ~= 14.15, near a sin(w0 t) maximum
    const double t = steps * dt;

    Field3D psi = ground;
    ses::driven_step(psi, prop, DipoleDrive{Vec3d{0.0, 0.0, 1.0}, e0, w0}, 0.0, steps);

    // Ehrenfest is EXACT for quadratic + linear potentials:
    // <z>(t) = -(E0 / 2 w0) t sin(w0 t).
    const double expected_z = -(e0 / (2.0 * w0)) * t * std::sin(w0 * t);
    EXPECT_NEAR(ses::mean_position(psi).z, expected_z, 0.02);

    // Poisson populations of the coherent state, |alpha| = E0 t/(2 sqrt(2 w0)).
    const double alpha_sq = std::pow(e0 * t / (2.0 * std::sqrt(2.0 * w0)), 2.0);
    const double p0_expected = std::exp(-alpha_sq);
    const double p1_expected = alpha_sq * std::exp(-alpha_sq);
    EXPECT_NEAR(population(ground, psi), p0_expected, 0.06);
    EXPECT_NEAR(population(excited_z, psi), p1_expected, 0.06);

    // Selection rule: the z-polarized drive cannot populate the y state --
    // exact by symmetry, ladder or not.
    EXPECT_LT(population(excited_y, psi), 1e-4);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-10);
}

}  // namespace
