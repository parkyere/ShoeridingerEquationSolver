// RED: real spherical harmonics + 3D orbital synthesis. The radial
// engine gives u_nl(r) on a 1D grid; psi = (u/r) Y_lm placed onto the 3D
// grid gives the eigenstate WITHOUT any imaginary-time ladder -- exact
// separation of variables, the textbook route.
//
// Oracles:
//  - point values of the real Y_lm on the unit sphere pin the formulas;
//  - synthesized states of the isotropic harmonic trap reproduce the
//    analytic 3D energies E = w (2k + l + 3/2) through the FULL 3D
//    mean_energy (spectral Laplacian) -- radial solve, synthesis, and 3D
//    machinery must all agree;
//  - the synthesized set is orthonormal on the grid (radial orthogonality
//    survives interpolation; angular orthogonality by symmetry);
//  - parity: p_x flips sign under x -> -x, d_xy is even under (x,y) -> -.

#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/harmonics.hpp>
#include <core/observables.hpp>
#include <core/potential.hpp>
#include <core/radial.hpp>
#include <core/vec.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::RadialGrid;

constexpr double kPi = 3.14159265358979323846;

TEST(RealSphericalHarmonic, PointValuesOnTheUnitSphere) {
    // Y_00 everywhere.
    EXPECT_NEAR(ses::real_spherical_harmonic(0, 0, 0.3, -0.5, 0.8),
                1.0 / (2.0 * std::sqrt(kPi)), 1e-12);
    // p_z at the +z pole; p_x on the +x axis.
    EXPECT_NEAR(ses::real_spherical_harmonic(1, 0, 0.0, 0.0, 1.0),
                std::sqrt(3.0 / (4.0 * kPi)), 1e-12);
    EXPECT_NEAR(ses::real_spherical_harmonic(1, 1, 1.0, 0.0, 0.0),
                std::sqrt(3.0 / (4.0 * kPi)), 1e-12);
    EXPECT_NEAR(ses::real_spherical_harmonic(1, -1, 0.0, -1.0, 0.0),
                -std::sqrt(3.0 / (4.0 * kPi)), 1e-12);
    // d_z2 at the pole: (3 z^2 - r^2)/r^2 = 2 there.
    EXPECT_NEAR(ses::real_spherical_harmonic(2, 0, 0.0, 0.0, 1.0),
                0.5 * std::sqrt(5.0 / kPi), 1e-12);
    // d_xy peaks on the diagonal x = y.
    const double s = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(ses::real_spherical_harmonic(2, -2, s, s, 0.0),
                0.25 * std::sqrt(15.0 / kPi), 1e-12);
    // Zeros where the nodal planes sit.
    EXPECT_NEAR(ses::real_spherical_harmonic(1, 0, 1.0, 0.0, 0.0), 0.0, 1e-12);
    EXPECT_NEAR(ses::real_spherical_harmonic(2, -2, 1.0, 0.0, 0.0), 0.0, 1e-12);
}

TEST(RealSphericalHarmonic, LEqualsThreePointValues) {
    // f_z3 at the +z pole: z(5z^2 - 3r^2)/r^3 = 2 there.
    EXPECT_NEAR(ses::real_spherical_harmonic(3, 0, 0.0, 0.0, 1.0),
                0.5 * std::sqrt(7.0 / kPi), 1e-12);
    // f_x(x2-3y2) on the +x axis: x(x^2 - 3y^2)/r^3 = 1.
    EXPECT_NEAR(ses::real_spherical_harmonic(3, 3, 1.0, 0.0, 0.0),
                0.25 * std::sqrt(35.0 / (2.0 * kPi)), 1e-12);
    // f_xyz on the (1,1,1) diagonal.
    const double s3 = 1.0 / std::sqrt(3.0);
    EXPECT_NEAR(ses::real_spherical_harmonic(3, -2, s3, s3, s3),
                0.5 * std::sqrt(105.0 / kPi) * s3 * s3 * s3, 1e-12);
    // f_y(5z2-r2) on the +y axis: y(5z^2 - r^2)/r^3 = -1.
    EXPECT_NEAR(ses::real_spherical_harmonic(3, -1, 0.0, 1.0, 0.0),
                -0.25 * std::sqrt(21.0 / (2.0 * kPi)), 1e-12);
    // Nodal planes.
    EXPECT_NEAR(ses::real_spherical_harmonic(3, 0, 1.0, 0.0, 0.0), 0.0, 1e-12);
    EXPECT_NEAR(ses::real_spherical_harmonic(3, -2, 1.0, 1.0, 0.0), 0.0, 1e-12);
    EXPECT_NEAR(ses::real_spherical_harmonic(3, 2, 1.0, 1.0, 1.0), 0.0, 1e-12);
}

TEST(RealSphericalHarmonic, LEqualsFourPointValues) {
    // g_z4 (m=0) at the +z pole: (35 z^4 - 30 z^2 r^2 + 3 r^4)/r^4 = 8 there.
    EXPECT_NEAR(ses::real_spherical_harmonic(4, 0, 0.0, 0.0, 1.0),
                (3.0 / 16.0) * std::sqrt(1.0 / kPi) * 8.0, 1e-12);
    // g (m=+4) on the +x axis: (x^4 - 6 x^2 y^2 + y^4)/r^4 = 1.
    EXPECT_NEAR(ses::real_spherical_harmonic(4, 4, 1.0, 0.0, 0.0),
                (3.0 / 16.0) * std::sqrt(35.0 / kPi), 1e-12);
    // g (m=+2) on the +x axis: (x^2 - y^2)(7 z^2 - r^2)/r^4 = (1)(-1) = -1.
    EXPECT_NEAR(ses::real_spherical_harmonic(4, 2, 1.0, 0.0, 0.0),
                -0.375 * std::sqrt(5.0 / kPi), 1e-12);
    // g (m=+3) on the (1,0,1)/sqrt2 diagonal: x z (x^2 - 3 y^2)/r^4 = 1/4.
    const double s = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(ses::real_spherical_harmonic(4, 3, s, 0.0, s),
                0.25 * 0.75 * std::sqrt(35.0 / (2.0 * kPi)), 1e-12);
    // Nodal: m=+4 vanishes at the pole; m=-4 vanishes on the x = y diagonal.
    EXPECT_NEAR(ses::real_spherical_harmonic(4, 4, 0.0, 0.0, 1.0), 0.0, 1e-12);
    EXPECT_NEAR(ses::real_spherical_harmonic(4, -4, s, s, 0.0), 0.0, 1e-12);
}

TEST(RealSphericalHarmonic, LEqualsFivePointValues) {
    // The l = 5 (h) shell: the six n = 6 orbitals' highest angular momentum.
    // Canonical Cartesian forms (numerator / r^5); constants and polynomials
    // verified harmonic + orthonormal over the sphere to machine precision.
    // h_z5 (m=0) at the +z pole: z(63 z^4 - 70 z^2 r^2 + 15 r^4)/r^5 = 8 there.
    EXPECT_NEAR(ses::real_spherical_harmonic(5, 0, 0.0, 0.0, 1.0),
                (1.0 / 16.0) * std::sqrt(11.0 / kPi) * 8.0, 1e-12);
    // m=+5 on the +x axis: x^5 / r^5 = 1.
    EXPECT_NEAR(ses::real_spherical_harmonic(5, 5, 1.0, 0.0, 0.0),
                (1.0 / 32.0) * std::sqrt(1386.0 / kPi), 1e-12);
    // m=-5 on the +y axis: (5 x^4 y - 10 x^2 y^3 + y^5)/r^5 = 1.
    EXPECT_NEAR(ses::real_spherical_harmonic(5, -5, 0.0, 1.0, 0.0),
                (1.0 / 32.0) * std::sqrt(1386.0 / kPi), 1e-12);
    // m=+1 on the +x axis: x(21 z^4 - 14 z^2 r^2 + r^4)/r^5 = 1.
    EXPECT_NEAR(ses::real_spherical_harmonic(5, 1, 1.0, 0.0, 0.0),
                (1.0 / 16.0) * std::sqrt(165.0 / kPi), 1e-12);
    // m=-3 on the +y axis: y(3 x^2 - y^2)(9 z^2 - r^2)/r^5 = (1)(-1)(-1) = 1.
    EXPECT_NEAR(ses::real_spherical_harmonic(5, -3, 0.0, 1.0, 0.0),
                (1.0 / 32.0) * std::sqrt(770.0 / kPi), 1e-12);
    // m=+3 on the +x axis: x(x^2 - 3 y^2)(9 z^2 - r^2)/r^5 = (1)(1)(-1) = -1.
    EXPECT_NEAR(ses::real_spherical_harmonic(5, 3, 1.0, 0.0, 0.0),
                -(1.0 / 32.0) * std::sqrt(770.0 / kPi), 1e-12);
    // m=+2 on the (1,0,1)/sqrt2 diagonal: (x^2-y^2) z (3 z^2 - r^2)/r^5 =
    // (1/2)(s)(1/2) = s/4 with s = 1/sqrt2.
    const double s = 1.0 / std::sqrt(2.0);
    EXPECT_NEAR(ses::real_spherical_harmonic(5, 2, s, 0.0, s),
                (1.0 / 8.0) * std::sqrt(1155.0 / kPi) * 0.25 * s, 1e-12);
    // m=-4 vanishes on the x = y diagonal (4 x y (x^2 - y^2) z = 0).
    const double s3 = 1.0 / std::sqrt(3.0);
    EXPECT_NEAR(ses::real_spherical_harmonic(5, -4, s3, s3, s3), 0.0, 1e-12);
    // Pole nodes: m=+4 and m=+5 need x, y and vanish on the +z axis.
    EXPECT_NEAR(ses::real_spherical_harmonic(5, 4, 0.0, 0.0, 1.0), 0.0, 1e-12);
    EXPECT_NEAR(ses::real_spherical_harmonic(5, 5, 0.0, 0.0, 1.0), 0.0, 1e-12);
}

struct SynthCase {
    int l;
    int m;
    int k;  // radial excitation (nodes)
    double energy;
};

TEST(SynthesizeOrbital, HarmonicTrapEnergiesThroughTheFull3DMachinery) {
    const Grid1D axis{-8.0, 8.0, 64};
    const Grid3D g{axis, axis, axis};
    const std::vector<double> v3 = ses::harmonic_potential(g, 1.0, ses::Vec3d{});

    const RadialGrid rg{8.0, 1599};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }

    // E = w (2k + l + 3/2): s, p_z, d_z2, d_xy, the 2s-like k=1, the
    // l = 3 f states (E = 4.5), the l = 4 g states (E = 5.5), and the l = 5
    // h states (E = 6.5) -- the highest angular momentum of the n = 6 shell.
    const SynthCase cases[] = {
        {0, 0, 0, 1.5}, {1, 0, 0, 2.5}, {2, 0, 0, 3.5}, {2, -2, 0, 3.5},
        {0, 0, 1, 3.5}, {3, 0, 0, 4.5}, {3, -2, 0, 4.5}, {3, 3, 0, 4.5},
        {4, 0, 0, 5.5}, {4, -4, 0, 5.5}, {4, 2, 0, 5.5},
        {5, 0, 0, 6.5}, {5, -5, 0, 6.5}, {5, 3, 0, 6.5},
    };
    for (const SynthCase& c : cases) {
        const ses::RadialState st =
            ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, c.l), c.k);
        const Field3D psi = ses::synthesize_orbital(g, rg, st.u, c.l, c.m);
        EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-10);
        EXPECT_NEAR(ses::mean_energy(psi, v3), c.energy, 2e-3)
            << "l=" << c.l << " m=" << c.m << " k=" << c.k;
        EXPECT_NEAR(st.energy, c.energy, 1e-3);
    }
}

TEST(SynthesizeOrbital, SetIsOrthonormalOnTheGrid) {
    const Grid1D axis{-8.0, 8.0, 64};
    const Grid3D g{axis, axis, axis};
    const RadialGrid rg{8.0, 1599};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }

    std::vector<Field3D> states;
    const SynthCase cases[] = {
        {0, 0, 0, 0.0}, {0, 0, 1, 0.0},                                    // 1s 2s
        {1, -1, 0, 0.0}, {1, 0, 0, 0.0}, {1, 1, 0, 0.0},                   // p
        {2, -2, 0, 0.0}, {2, -1, 0, 0.0}, {2, 0, 0, 0.0}, {2, 1, 0, 0.0},
        {2, 2, 0, 0.0},                                                    // d
        {3, -3, 0, 0.0}, {3, -2, 0, 0.0}, {3, -1, 0, 0.0}, {3, 0, 0, 0.0},
        {3, 1, 0, 0.0}, {3, 2, 0, 0.0}, {3, 3, 0, 0.0},                    // f
        {4, -4, 0, 0.0}, {4, -3, 0, 0.0}, {4, -2, 0, 0.0}, {4, -1, 0, 0.0},
        {4, 0, 0, 0.0}, {4, 1, 0, 0.0}, {4, 2, 0, 0.0}, {4, 3, 0, 0.0},
        {4, 4, 0, 0.0},                                                    // g
        {5, -5, 0, 0.0}, {5, -4, 0, 0.0}, {5, -3, 0, 0.0}, {5, -2, 0, 0.0},
        {5, -1, 0, 0.0}, {5, 0, 0, 0.0}, {5, 1, 0, 0.0}, {5, 2, 0, 0.0},
        {5, 3, 0, 0.0}, {5, 4, 0, 0.0}, {5, 5, 0, 0.0},                    // h
    };
    for (const SynthCase& c : cases) {
        const ses::RadialState st =
            ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, c.l), c.k);
        states.push_back(ses::synthesize_orbital(g, rg, st.u, c.l, c.m));
    }
    for (std::size_t a = 0; a < states.size(); ++a) {
        for (std::size_t b = 0; b < states.size(); ++b) {
            const double overlap = std::abs(ses::inner_product(states[a], states[b]));
            if (a == b) {
                EXPECT_NEAR(overlap, 1.0, 1e-9);
            } else {
                EXPECT_LT(overlap, 5e-3) << "pair " << a << "," << b;
            }
        }
    }
}

TEST(SynthesizeOrbital, ParityFollowsTheHarmonic) {
    const Grid1D axis{-8.0, 8.0, 32};
    const Grid3D g{axis, axis, axis};
    const RadialGrid rg{8.0, 799};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = 0.5 * rg.r(i) * rg.r(i);
    }
    const ses::RadialState p =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 1), 0);
    const Field3D px = ses::synthesize_orbital(g, rg, p.u, 1, 1);
    // Odd in x: psi(-x, y, z) = -psi(x, y, z). Periodic grid: coordinate
    // index i maps to n - i for the mirrored point (i > 0).
    const int n = g.x.n;
    for (int i = 1; i < n; i += 7) {
        EXPECT_NEAR(px(i, 5, 9).real(), -px(n - i, 5, 9).real(), 1e-12);
    }
    const ses::RadialState d =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, 2), 0);
    const Field3D dxy = ses::synthesize_orbital(g, rg, d.u, 2, -2);
    for (int i = 1; i < n; i += 7) {
        // Even under the simultaneous flip of x AND y.
        EXPECT_NEAR(dxy(i, 3, 9).real(), dxy(n - i, n - 3, 9).real(), 1e-12);
    }
}

TEST(RealSphericalHarmonic, CosSinPairsCombineToLzEigenfunctions) {
    // The L_z-measurement convention: for every l <= 5, 0 < m <= l, the
    // combination Y_{l,+m} + i Y_{l,-m} must be proportional to e^{+i m phi}
    // -- azimuthally UNIFORM magnitude with phase advancing as +m*phi. This
    // pins the (cos, sin) pairing AND its sign for the whole table at once.
    const double z = 0.23;  // generic polar angle (away from theta nodes)
    const double rho = std::sqrt(1.0 - z * z);
    for (int l = 1; l <= 5; ++l) {
        for (int m = 1; m <= l; ++m) {
            const auto f = [&](double phi) {
                const double x = rho * std::cos(phi);
                const double y = rho * std::sin(phi);
                return ses::Complex<double>{
                    ses::real_spherical_harmonic(l, m, x, y, z),
                    ses::real_spherical_harmonic(l, -m, x, y, z)};
            };
            const ses::Complex<double> f0 = f(0.0);
            ASSERT_GT(std::abs(f0), 1e-4) << "theta node hit at l=" << l;
            for (int k = 1; k <= 7; ++k) {
                const double phi = 0.83 * k;
                const ses::Complex<double> ratio = f(phi) / f0;
                // ratio must equal e^{+i m phi} exactly (uniform ring,
                // right-handed phase advance).
                EXPECT_NEAR(ratio.real(), std::cos(m * phi), 1e-12)
                    << "l=" << l << " m=" << m;
                EXPECT_NEAR(ratio.imag(), std::sin(m * phi), 1e-12)
                    << "l=" << l << " m=" << m;
            }
        }
    }
}

}  // namespace
