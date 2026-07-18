// RED: the license physics of the one-electron molecule scenes (fixed
// nuclei, Born-Oppenheimer).
//
//  - H2+ (two bare protons): the ground sigma_g piles charge BETWEEN the
//    nuclei, the deflated first excited sigma_u carries a nodal plane
//    there; and the total energy E_elec(R) + 1/R is LOWER at the bond
//    length than both at large R and than the isolated-atom limit on the
//    same grid -- the chemical bond as one inequality chain.
//  - Benzene toy (six equal soft cores on a ring): the uniform hexagon's
//    first excited pair is (near-)degenerate; a Kekule 1-2-1-2 distortion
//    of the SAME ring splits the pair -- the one-electron energy signature
//    of what X-ray diffraction settled (equal bonds, no alternation).

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

import ses.field;
import ses.grid;
import ses.imaginary_time;
import ses.observables;
import ses.potential;
import ses.vec;
import ses.wavepacket;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

double relax_ground_energy(const Grid3D& g, const std::vector<double>& v,
                           Field3D& psi, int steps) {
    const ses::ImaginaryTimePropagator3D relaxer{g, v, 0.05};
    relaxer.relax(psi, steps);
    return ses::mean_energy(psi, v);
}

TEST(H2Plus, BondingAndAntibondingAndTheChemicalBond) {
    const Grid1D ax{-8.0, 8.0, 32};  // h = 0.5; centers land on grid points
    const Grid3D g{ax, ax, ax};

    // R = 2 (near equilibrium): sigma_g then the deflated sigma_u.
    const std::vector<Vec3d> near = {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const std::vector<double> v2 =
        ses::regularized_coulomb_potential(g, 1.0, near);
    Field3D ground = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{1.5, 1.5, 1.5},
                                              Vec3d{});
    const double e_g = relax_ground_energy(g, v2, ground, 600);

    // Antisymmetric-in-x seed + deflation vs the ground: sigma_u.
    Field3D odd{g};
    const Field3D lobe_r = ses::gaussian_wavepacket(
        g, Vec3d{1.0, 0.0, 0.0}, Vec3d{1.2, 1.2, 1.2}, Vec3d{});
    const Field3D lobe_l = ses::gaussian_wavepacket(
        g, Vec3d{-1.0, 0.0, 0.0}, Vec3d{1.2, 1.2, 1.2}, Vec3d{});
    for (int i = 0; i < g.size(); ++i) {
        odd.data()[static_cast<std::size_t>(i)] =
            lobe_r.data()[static_cast<std::size_t>(i)] -
            lobe_l.data()[static_cast<std::size_t>(i)];
    }
    const ses::ImaginaryTimePropagator3D relaxer{g, v2, 0.05};
    relaxer.relax_deflated(odd, {&ground}, 600);
    const double e_u = ses::mean_energy(odd, v2);

    EXPECT_LT(e_g, e_u) << "bonding below antibonding";
    // sigma_u has a nodal plane between the nuclei; sigma_g piles charge
    // there (the bond density).
    const int mid = 16;  // coord 0
    const double den_g = std::norm(ground(mid, mid, mid));
    const double den_u = std::norm(odd(mid, mid, mid));
    EXPECT_LT(den_u, 0.05 * den_g);

    // The bond: E_total(R = 2) undercuts both the stretched molecule and
    // the isolated atom (same grid, same regularization -- honest ladder).
    const std::vector<Vec3d> far = {{-3.0, 0.0, 0.0}, {3.0, 0.0, 0.0}};
    const std::vector<double> v6 =
        ses::regularized_coulomb_potential(g, 1.0, far);
    Field3D stretched = ses::gaussian_wavepacket(
        g, Vec3d{}, Vec3d{2.0, 1.5, 1.5}, Vec3d{});
    const double e_g6 = relax_ground_energy(g, v6, stretched, 600);

    const std::vector<Vec3d> lone = {{0.0, 0.0, 0.0}};
    const std::vector<double> v1 =
        ses::regularized_coulomb_potential(g, 1.0, lone);
    Field3D atom = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{1.2, 1.2, 1.2},
                                            Vec3d{});
    const double e_atom = relax_ground_energy(g, v1, atom, 600);

    const double et2 = e_g + 1.0 / 2.0;
    const double et6 = e_g6 + 1.0 / 6.0;
    EXPECT_LT(et2, et6) << "the molecule binds vs the stretched one";
    EXPECT_LT(et2, e_atom) << "the molecule binds vs H + p at rest";
}

TEST(BenzeneToy, KekuleDistortionSplitsTheDegeneratePair) {
    const Grid1D ax{-8.0, 8.0, 32};
    const Grid3D g{ax, ax, ax};
    const double ring_r = 2.63;  // benzene C-C = 1.39 A in bohr
    const double soft_a = 0.8;
    const double kPi = 3.14159265358979323846;

    auto ring = [&](double delta_deg) {
        // Kekule distortion: alternate the vertex ANGLES by +-delta so the
        // six sides alternate short-long-short-long around the SAME circle.
        std::vector<Vec3d> c;
        for (int i = 0; i < 6; ++i) {
            const double th = kPi / 3.0 * i +
                              (i % 2 == 0 ? 1.0 : -1.0) * delta_deg * kPi / 180.0;
            c.push_back({ring_r * std::cos(th), ring_r * std::sin(th), 0.0});
        }
        return c;
    };

    auto excited_pair_split = [&](const std::vector<Vec3d>& centers,
                                  double* out_gap01) {
        const std::vector<double> v =
            ses::soft_coulomb_potential(g, 1.0, soft_a, centers);
        const ses::ImaginaryTimePropagator3D relaxer{g, v, 0.05};
        Field3D e0 = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{2.0, 2.0, 1.2},
                                              Vec3d{});
        relaxer.relax(e0, 400);
        // Two independent mixed-symmetry seeds, deflated in sequence.
        Field3D e1 = ses::gaussian_wavepacket(
            g, Vec3d{ring_r, 0.4, 0.0}, Vec3d{1.5, 1.5, 1.2}, Vec3d{});
        relaxer.relax_deflated(e1, {&e0}, 400);
        Field3D e2 = ses::gaussian_wavepacket(
            g, Vec3d{-0.5, ring_r, 0.0}, Vec3d{1.5, 1.5, 1.2}, Vec3d{});
        relaxer.relax_deflated(e2, {&e0, &e1}, 400);
        const double ee0 = ses::mean_energy(e0, v);
        const double ee1 = ses::mean_energy(e1, v);
        const double ee2 = ses::mean_energy(e2, v);
        if (out_gap01 != nullptr) {
            *out_gap01 = std::min(ee1, ee2) - ee0;
        }
        return std::abs(ee2 - ee1);
    };

    double gap_uniform = 0.0;
    const double split_uniform = excited_pair_split(ring(0.0), &gap_uniform);
    const double split_kekule = excited_pair_split(ring(5.0), nullptr);

    EXPECT_GT(gap_uniform, 0.01) << "the pair sits above the ground state";
    // Uniform ring: the pair is degenerate up to lattice C4-vs-C6 error.
    EXPECT_LT(split_uniform, 0.01);
    // Kekule alternation opens a visible splitting: the one-electron
    // fingerprint that equal bonds (X-ray) vs 1-2-1-2 are DIFFERENT physics.
    EXPECT_GT(split_kekule, 3.0 * split_uniform + 0.005);
}

}  // namespace
