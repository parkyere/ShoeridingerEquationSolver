// RED: the magnetic split-operator propagator. The
// minimal-coupling Hamiltonian for an electron in B = B z^ (symmetric gauge,
// atomic units) is
//     H = 1/2 p^2 + V + (B/2) L_z + (B^2/8) rho^2,
// paramagnetic + diamagnetic. This solves it PROPERLY (psi actually evolves),
// not as a display rotation:
//   - the diamagnetic (B^2/8) rho^2 is diagonal in position -> folded into
//     the potential of an internal SplitOperator3D;
//   - the paramagnetic (B/2) L_z is the z-rotation generator -> applied to
//     psi via the exact three-shear rotate_z, Strang-split around the core.
//
// Oracles:
//  - effective_potential = V + (B^2/8) rho^2 exactly (diamagnetic present);
//  - B = 0 reduces to the plain SplitOperator3D;
//  - FACTORIZATION (the key one): for a z-symmetric V, L_z commutes with the
//    rest, so the magnetic evolution equals [H0' evolution] then a rigid
//    rotation by omega_L * t, omega_L = B/2 -- an exact identity;
//  - norm is conserved.

#include <core/field.hpp>
import ses.grid;
#include <core/magnetic.hpp>
#include <core/potential.hpp>
#include <core/propagator.hpp>
#include <core/rotation.hpp>
import ses.vec;
#include <core/wavepacket.hpp>

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

double max_abs_diff(const Field3D& a, const Field3D& b) {
    double d = 0.0;
    for (std::size_t i = 0; i < a.data().size(); ++i) {
        d = std::max(d, std::abs(a.data()[i] - b.data()[i]));
    }
    return d;
}

TEST(MagneticPropagator, EffectivePotentialCarriesTheDiamagneticTerm) {
    const Grid3D g = cube(12.0, 32);
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double b = 0.4;
    const ses::MagneticPropagator3D prop{g, v, 0.01, b};
    const std::vector<double>& veff = prop.effective_potential();
    for (int k = 0; k < g.z.n; k += 7) {
        for (int j = 0; j < g.y.n; j += 5) {
            for (int i = 0; i < g.x.n; i += 5) {
                const double rho2 =
                    g.x.coord(i) * g.x.coord(i) + g.y.coord(j) * g.y.coord(j);
                const double expected =
                    v[static_cast<std::size_t>(g.flat(i, j, k))] + b * b / 8.0 * rho2;
                EXPECT_NEAR(veff[static_cast<std::size_t>(g.flat(i, j, k))], expected,
                            1e-12);
            }
        }
    }
}

TEST(MagneticPropagator, ZeroFieldReducesToPlainPropagator) {
    const Grid3D g = cube(12.0, 32);
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.01;
    Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{2.0, 1.0, 0.0},
                                            ses::Vec3d{1.5, 1.5, 1.5},
                                            ses::Vec3d{0.0, 0.3, 0.0});
    Field3D mag = psi0;
    Field3D plain = psi0;
    ses::MagneticPropagator3D{g, v, dt, 0.0}.step(mag, 30);
    ses::SplitOperator3D{g, v, dt}.step(plain, 30);
    EXPECT_LT(max_abs_diff(mag, plain), 1e-10);
}

TEST(MagneticPropagator, EvolutionFactorsIntoCorePlusLarmorRotation) {
    const Grid3D g = cube(12.0, 32);
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.01;
    const double b = 0.5;
    const int n = 40;

    Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{2.5, 0.0, 0.5},
                                            ses::Vec3d{1.4, 1.4, 1.4},
                                            ses::Vec3d{0.0, 0.4, 0.0});
    const ses::MagneticPropagator3D prop{g, v, dt, b};

    Field3D mag = psi0;
    prop.step(mag, n);

    // Reference: evolve under the SAME diamagnetic-augmented core, then rotate
    // rigidly by omega_L * t = (B/2)(n dt). For a z-symmetric V the two are
    // an exact operator identity.
    Field3D ref = psi0;
    ses::SplitOperator3D{g, prop.effective_potential(), dt}.step(ref, n);
    ses::rotate_z(ref, 0.5 * b * (n * dt));

    // The factorization is EXACT in the continuum; on the discrete grid the
    // three-shear rotation does not perfectly commute with the core step
    // (band-limited resampling), so it holds to ~1e-5 at 32^3 -- the physical
    // identity confirmed to 5 figures, not a defect.
    EXPECT_LT(max_abs_diff(mag, ref), 1e-4);
}

TEST(MagneticPropagator, AboutXCarriesThePerpendicularDiamagneticTerm) {
    const Grid3D g = cube(12.0, 32);
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double b = 0.4;
    const ses::MagneticPropagator3D prop{g, v, 0.01, b, 0};  // axis x
    const std::vector<double>& veff = prop.effective_potential();
    for (int k = 0; k < g.z.n; k += 7) {
        for (int j = 0; j < g.y.n; j += 5) {
            for (int i = 0; i < g.x.n; i += 5) {
                const double perp2 =  // (B/2)L_x -> diamagnetic in y,z
                    g.y.coord(j) * g.y.coord(j) + g.z.coord(k) * g.z.coord(k);
                const double expected =
                    v[static_cast<std::size_t>(g.flat(i, j, k))] + b * b / 8.0 * perp2;
                EXPECT_NEAR(veff[static_cast<std::size_t>(g.flat(i, j, k))], expected,
                            1e-12);
            }
        }
    }
}

TEST(MagneticPropagator, AboutXFactorsIntoCorePlusRotation) {
    const Grid3D g = cube(12.0, 32);
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    const double dt = 0.01;
    const double b = 0.5;
    const int n = 40;
    Field3D psi0 = ses::gaussian_wavepacket(g, ses::Vec3d{0.5, 2.5, 0.0},
                                            ses::Vec3d{1.4, 1.4, 1.4},
                                            ses::Vec3d{0.4, 0.0, 0.0});
    const ses::MagneticPropagator3D prop{g, v, dt, b, 0};  // axis x
    Field3D mag = psi0;
    prop.step(mag, n);
    Field3D ref = psi0;
    ses::SplitOperator3D{g, prop.effective_potential(), dt}.step(ref, n);
    ses::rotate_axis(ref, 0, 0.5 * b * (n * dt));
    EXPECT_LT(max_abs_diff(mag, ref), 1e-4);
}

TEST(MagneticPropagator, ConservesNorm) {
    const Grid3D g = cube(12.0, 32);
    const std::vector<double> v = ses::soft_coulomb_potential(g, 1.0, 1.0, ses::Vec3d{});
    Field3D psi = ses::gaussian_wavepacket(g, ses::Vec3d{2.0, 0.0, 0.0},
                                           ses::Vec3d{1.5, 1.5, 1.5},
                                           ses::Vec3d{0.0, 0.3, 0.0});
    const double n0 = ses::norm_sq(psi);
    ses::MagneticPropagator3D{g, v, 0.01, 0.6}.step(psi, 60);
    EXPECT_NEAR(ses::norm_sq(psi), n0, 1e-8);
}

}  // namespace
