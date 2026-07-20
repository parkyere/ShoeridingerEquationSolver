// RED: the mean-field spin lattice. Damped J > 0 orders a random boot
// into a ferromagnet (|M| -> 1); damped J < 0 staggers it into Neel
// (staggered -> 1, |M| stays small); the UNDAMPED lattice conserves the
// mean-field energy; and a fully aligned lattice in a transverse B
// precesses RIGIDLY at omega_L (parallel exchange exerts no torque).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

import ses.spinlattice;

namespace {

ses::SpinLattice random_lattice(int n, unsigned seed) {
    ses::SpinLattice l;
    l.nx = n;
    l.ny = n;
    l.s.resize(static_cast<std::size_t>(n * n));
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    std::uniform_real_distribution<double> ph(0.0, 6.28318530717958647692);
    for (auto& sp : l.s) {
        const double z = u(rng);
        const double t = ph(rng);
        const double r = std::sqrt(std::max(0.0, 1.0 - z * z));
        sp = ses::spinor_from_bloch(r * std::cos(t), r * std::sin(t), z);
    }
    return l;
}

TEST(SpinLattice, SpinorFromBlochRoundTrips) {
    const double r = 1.0 / std::sqrt(3.0);
    const ses::Spinor s = ses::spinor_from_bloch(r, -r, r);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    ses::bloch_vector(s, &x, &y, &z);
    EXPECT_NEAR(x, r, 1e-12);
    EXPECT_NEAR(y, -r, 1e-12);
    EXPECT_NEAR(z, r, 1e-12);
}

TEST(SpinLattice, DampedFerroOrdersDampedNeelStaggers) {
    ses::SpinLattice l = random_lattice(5, 11);
    for (int k = 0; k < 6000; ++k) {  // J = +0.5, alpha = 0.1, no B
        ses::spinlattice_step(l, 0.0, 0.0, 0.0, 0.5, 0.1, 0.05);
    }
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    ses::lattice_magnetization(l, &mx, &my, &mz);
    EXPECT_GT(std::sqrt(mx * mx + my * my + mz * mz), 0.95);

    ses::SpinLattice a = random_lattice(5, 12);
    for (int k = 0; k < 12000; ++k) {  // J = -0.5: Neel
        ses::spinlattice_step(a, 0.0, 0.0, 0.0, -0.5, 0.1, 0.05);
    }
    ses::lattice_magnetization(a, &mx, &my, &mz);
    EXPECT_LT(std::sqrt(mx * mx + my * my + mz * mz), 0.3);
    EXPECT_GT(ses::lattice_staggered(a), 0.95);
}

TEST(SpinLattice, UndampedEnergyIsConserved) {
    ses::SpinLattice l = random_lattice(5, 13);
    const double e0 = ses::lattice_energy(l, 0.1, -0.2, 0.3, 0.4);
    ASSERT_NE(e0, 0.0);
    for (int k = 0; k < 2000; ++k) {
        ses::spinlattice_step(l, 0.1, -0.2, 0.3, 0.4, 0.0, 0.01);
    }
    const double e1 = ses::lattice_energy(l, 0.1, -0.2, 0.3, 0.4);
    EXPECT_NEAR(e1, e0, 5e-3 * std::abs(e0) + 5e-3);
}

TEST(SpinLattice, AlignedLatticePrecessesRigidly) {
    ses::SpinLattice l;
    l.nx = 5;
    l.ny = 5;
    l.s.assign(25, ses::spinor_from_bloch(1.0, 0.0, 0.0));  // all +x
    const double b = 0.5;
    const double dt = 0.01;
    const int n = 400;  // phase 2 rad about +z
    for (int k = 0; k < n; ++k) {
        ses::spinlattice_step(l, 0.0, 0.0, b, 0.7, 0.0, dt);
    }
    // Every site still coherent with the mean (rigid), at the Larmor
    // phase: parallel exchange fields exert no torque.
    double mx = 0.0;
    double my = 0.0;
    double mz = 0.0;
    ses::lattice_magnetization(l, &mx, &my, &mz);
    EXPECT_NEAR(std::hypot(mx, my), 1.0, 1e-9);  // no fan-out at all
    EXPECT_NEAR(mz, 0.0, 1e-12);
    EXPECT_NEAR(std::atan2(my, mx), b * n * dt, 1e-9);  // +B CCW
}

}  // namespace
