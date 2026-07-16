// RED: 3D Gaussian wavepacket, per-axis observables, and 3D potential
// builders.
//
// The packet is a product of three 1D packets with PER-AXIS width and
// momentum (anisotropic on purpose: any axis mix-up in the factory, the
// observables, or the k-mapping shows up as a wrong per-axis number).

#include <core/field.hpp>
import ses.grid;
#include <core/observables.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.wavepacket;
import ses.potential;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

Grid3D cube(double lo, double hi, int n) {
    return Grid3D{Grid1D{lo, hi, n}, Grid1D{lo, hi, n}, Grid1D{lo, hi, n}};
}

const Grid3D kGrid = cube(-10.0, 10.0, 64);
const Vec3d kR0{1.0, -2.0, 0.5};
const Vec3d kSigma{0.8, 1.0, 1.2};
const Vec3d kK0{1.5, 0.0, -1.0};

TEST(GaussianWavepacket3, IsNormalized) {
    const Field3D psi = ses::gaussian_wavepacket(kGrid, kR0, kSigma, kK0);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(Observables3, MeanPositionPerAxis) {
    const Field3D psi = ses::gaussian_wavepacket(kGrid, kR0, kSigma, kK0);
    const Vec3d r = ses::mean_position(psi);
    EXPECT_NEAR(r.x, 1.0, 1e-10);
    EXPECT_NEAR(r.y, -2.0, 1e-10);
    EXPECT_NEAR(r.z, 0.5, 1e-10);
}

TEST(Observables3, SigmaPerAxis) {
    const Field3D psi = ses::gaussian_wavepacket(kGrid, kR0, kSigma, kK0);
    const Vec3d s = ses::sigma_position(psi);
    EXPECT_NEAR(s.x, 0.8, 1e-10);
    EXPECT_NEAR(s.y, 1.0, 1e-10);
    EXPECT_NEAR(s.z, 1.2, 1e-10);
}

TEST(Observables3, MeanMomentumPerAxis) {
    const Field3D psi = ses::gaussian_wavepacket(kGrid, kR0, kSigma, kK0);
    const Vec3d p = ses::mean_momentum(psi);
    EXPECT_NEAR(p.x, 1.5, 1e-8);
    EXPECT_NEAR(p.y, 0.0, 1e-8);
    EXPECT_NEAR(p.z, -1.0, 1e-8);
}

// Integer coordinates: [0,8)^3 with n=8 -> x_i = 0..7 exactly per axis.
const Grid3D kIntGrid = cube(0.0, 8.0, 8);

TEST(HarmonicPotential3, ExactValuesAndMinimum) {
    // omega = 2, center (1,2,3): V = 2 * |r - c|^2.
    const std::vector<double> v = ses::harmonic_potential(kIntGrid, 2.0, Vec3d{1.0, 2.0, 3.0});
    ASSERT_EQ(v.size(), 512u);
    EXPECT_EQ(v[static_cast<std::size_t>(kIntGrid.flat(1, 2, 3))], 0.0);
    EXPECT_EQ(v[static_cast<std::size_t>(kIntGrid.flat(3, 2, 3))], 8.0);   // dx=2
    EXPECT_EQ(v[static_cast<std::size_t>(kIntGrid.flat(1, 4, 3))], 8.0);   // dy=2
    EXPECT_EQ(v[static_cast<std::size_t>(kIntGrid.flat(1, 2, 5))], 8.0);   // dz=2
    EXPECT_EQ(v[static_cast<std::size_t>(kIntGrid.flat(2, 3, 4))], 6.0);   // (1,1,1)
}

TEST(SoftCoulombPotential3, ExactValuesAndFiniteAtNucleus) {
    // Z = 1, a = 1, nucleus (2,2,2): V = -1/sqrt(|r-c|^2 + 1).
    const std::vector<double> v =
        ses::soft_coulomb_potential(kIntGrid, 1.0, 1.0, Vec3d{2.0, 2.0, 2.0});
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(kIntGrid.flat(2, 2, 2))], -1.0);
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(kIntGrid.flat(4, 2, 2))],
                     -1.0 / std::sqrt(5.0));
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(kIntGrid.flat(2, 4, 2))],
                     -1.0 / std::sqrt(5.0));
    EXPECT_DOUBLE_EQ(v[static_cast<std::size_t>(kIntGrid.flat(3, 3, 3))],
                     -1.0 / 2.0);  // |r-c|^2 = 3, +1 -> sqrt(4)
}

}  // namespace
