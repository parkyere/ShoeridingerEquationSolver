// Validation capstone: the 3D softened-hydrogen ground state -- the electron
// cloud the app renders. Validates existing machinery; introduces no new API.
//
// Oracles are RIGOROUS, no literature fits:
//  - Variational window: V_soft(r) = -Z/sqrt(r^2+a^2) > -Z/r pointwise, so
//    E0(soft) > E0(bare hydrogen) = -0.5 Hartree; and a Coulomb tail always
//    binds, so E0 < 0.
//  - Monotonicity: larger a -> shallower potential everywhere -> higher E0
//    (strict, by the variational principle on the same grid, where the
//    discretization bias largely cancels between the two runs).
//  - Symmetry: the cloud is spherical about the nucleus.

#include <core/field.hpp>
import ses.grid;
#include <core/imaginary_time.hpp>
#include <core/observables.hpp>
#include <core/potential.hpp>
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

const Grid3D kGrid = cube(-12.0, 12.0, 32);

struct Relaxed {
    Field3D psi;
    double energy;
};

Relaxed relax_hydrogen(double a) {
    const std::vector<double> v = ses::soft_coulomb_potential(kGrid, 1.0, a, Vec3d{});
    Field3D psi = ses::gaussian_wavepacket(kGrid, Vec3d{}, Vec3d{1.3, 1.3, 1.3}, Vec3d{});
    ses::ImaginaryTimePropagator3D relaxer{kGrid, v, 0.02};
    relaxer.relax(psi, 750);  // tau = 15
    const double e = ses::mean_energy(psi, v);
    return Relaxed{psi, e};
}

const Relaxed& hydrogen_a1() {
    static const Relaxed r = relax_hydrogen(1.0);
    return r;
}

TEST(Hydrogen3, GroundStateEnergyWithinVariationalWindow) {
    const double e0 = hydrogen_a1().energy;
    EXPECT_GT(e0, -0.5);  // soft potential is strictly above bare -1/r
    EXPECT_LT(e0, -0.1);  // clearly bound, far from the continuum
}

TEST(Hydrogen3, EnergyIncreasesWithSoftening) {
    const double e_soft = relax_hydrogen(1.4).energy;
    const double e_hard = hydrogen_a1().energy;
    EXPECT_LT(e_hard, e_soft);        // deeper potential binds more strongly
    EXPECT_GT(e_soft - e_hard, 0.01);  // by a clear margin, not noise
}

TEST(Hydrogen3, CloudIsSphericalAboutNucleus) {
    const Field3D& psi = hydrogen_a1().psi;
    // The periodic grid [-12,12) is slightly asymmetric about 0 (a point at
    // -12, none at +12), which biases each mean by ~6e-6 IDENTICALLY on all
    // axes. So the sharp sphericity oracle is axis EQUALITY (1e-9); the
    // absolute center check is looser (1e-4), still far below any real
    // asymmetry bug.
    const Vec3d r = ses::mean_position(psi);
    EXPECT_NEAR(r.x, 0.0, 1e-4);
    EXPECT_NEAR(r.x, r.y, 1e-9);
    EXPECT_NEAR(r.y, r.z, 1e-9);
    const Vec3d s = ses::sigma_position(psi);
    EXPECT_NEAR(s.x, s.y, 1e-9);
    EXPECT_NEAR(s.y, s.z, 1e-9);
    EXPECT_GT(s.x, 0.5);  // a real extended cloud, not a collapsed spike
}

}  // namespace
