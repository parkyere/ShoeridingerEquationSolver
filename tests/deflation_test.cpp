// RED: excited states via deflated imaginary-time relaxation.
// e^{-H dtau} cannot remove components along already-found lower
// eigenstates by decay alone once they dominate -- so each step PROJECTS
// them out (Gram-Schmidt deflation), and the flow converges to the ground
// state of the orthogonal complement: the next excited state.
//
// Building block: the discrete inner product <a|b> = sum conj(a_i) b_i dV.
//
// Oracles:
//  - inner_product: exact hand values, conjugate symmetry, <a|a> == norm_sq,
//    even/odd Gaussian orthogonality;
//  - 3D harmonic (w = 1): ground E0 = 3/2; the deflated relaxation from a
//    displaced (mixed-parity) guess must land at the first excited level
//    E1 = 5/2, orthogonal to the ground state -- while the SAME guess
//    without deflation falls back to the ground state (the control).

#include <complex>
#include <core/field.hpp>
import ses.grid;
#include <core/imaginary_time.hpp>
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

TEST(InnerProduct, ExactSingleCellValue) {
    const Grid1D axis{0.0, 2.0, 2};  // dV = 1 exactly
    const Grid3D g{axis, axis, axis};
    Field3D a{g};
    Field3D b{g};
    a(1, 0, 1) = std::complex<double>{1.0, 2.0};
    b(1, 0, 1) = std::complex<double>{3.0, -1.0};
    // conj(1+2i)(3-i) = (1-2i)(3-i) = 3 - i - 6i + 2i^2 = 1 - 7i
    const std::complex<double> ip = ses::inner_product(a, b);
    EXPECT_DOUBLE_EQ(ip.real(), 1.0);
    EXPECT_DOUBLE_EQ(ip.imag(), -7.0);
}

TEST(InnerProduct, SelfIsNormSqAndConjugateSymmetric) {
    const Grid1D axis{-8.0, 8.0, 16};
    const Grid3D g{axis, axis, axis};
    const Field3D a = ses::gaussian_wavepacket(g, Vec3d{1.0, 0.0, 0.0},
                                               Vec3d{1.5, 1.5, 1.5}, Vec3d{0.4, 0.0, 0.0});
    const Field3D b = ses::gaussian_wavepacket(g, Vec3d{-0.5, 0.5, 0.0},
                                               Vec3d{1.0, 1.2, 1.4}, Vec3d{0.0, 0.3, 0.0});

    const std::complex<double> aa = ses::inner_product(a, a);
    EXPECT_NEAR(aa.real(), ses::norm_sq(a), 1e-12);
    EXPECT_NEAR(aa.imag(), 0.0, 1e-12);

    const std::complex<double> ab = ses::inner_product(a, b);
    const std::complex<double> ba = ses::inner_product(b, a);
    EXPECT_NEAR(ab.real(), ba.real(), 1e-12);
    EXPECT_NEAR(ab.imag(), -ba.imag(), 1e-12);
}

TEST(InnerProduct, EvenAndOddGaussiansAreOrthogonal) {
    const Grid1D axis{-8.0, 8.0, 32};
    const Grid3D g{axis, axis, axis};
    Field3D even{g};
    Field3D odd{g};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                const double env = std::exp(-(x * x + y * y + z * z) / 4.0);
                even(i, j, k) = std::complex<double>{env, 0.0};
                odd(i, j, k) = std::complex<double>{x * env, 0.0};
            }
        }
    }
    const std::complex<double> ip = ses::inner_product(even, odd);
    EXPECT_NEAR(ip.real(), 0.0, 1e-12);
    EXPECT_NEAR(ip.imag(), 0.0, 1e-12);
}

TEST(DeflatedRelaxation, FindsTheFirstExcitedHarmonicState) {
    const double omega = 1.0;
    const Grid1D axis{-8.0, 8.0, 32};
    const Grid3D g{axis, axis, axis};
    const std::vector<double> v = ses::harmonic_potential(g, omega, Vec3d{});
    const ses::ImaginaryTimePropagator3D relaxer{g, v, 0.05};

    // Ground state (even sector), E0 = 3/2.
    Field3D ground = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{1.2, 1.2, 1.2}, Vec3d{});
    relaxer.relax(ground, 600);
    EXPECT_NEAR(ses::mean_energy(ground, v), 1.5 * omega, 2e-2);

    // Mixed-parity guess: displaced, so it overlaps BOTH the ground state
    // and the p-manifold.
    const Field3D guess =
        ses::gaussian_wavepacket(g, Vec3d{1.0, 0.5, 0.0}, Vec3d{1.2, 1.2, 1.2}, Vec3d{});

    // CONTROL: without deflation the same guess falls to the ground state.
    Field3D undeflated = guess;
    relaxer.relax(undeflated, 600);
    EXPECT_NEAR(ses::mean_energy(undeflated, v), 1.5 * omega, 2e-2);

    // Deflated: converges to the first excited level E1 = 5/2, orthogonal
    // to the ground state.
    Field3D excited = guess;
    relaxer.relax_deflated(excited, {&ground}, 600);
    EXPECT_NEAR(ses::mean_energy(excited, v), 2.5 * omega, 2e-2);
    const std::complex<double> overlap = ses::inner_product(ground, excited);
    EXPECT_NEAR(std::abs(overlap.real()), 0.0, 1e-6);
    EXPECT_NEAR(std::abs(overlap.imag()), 0.0, 1e-6);
    EXPECT_NEAR(ses::norm_sq(excited), 1.0, 1e-12);
}

}  // namespace
