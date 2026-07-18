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

TEST(BenzeneToy, PairStaysDegenerateButBondChargeAlternatesUnderKekule) {
    // The honest one-electron fingerprints of uniform vs Kekule 1-2-1-2
    // (measured, not assumed -- a first draft asserting the pair SPLITS
    // failed at 1.4e-6: the Kekule ring keeps D3h, whose 2-dim irrep
    // protects the degeneracy; in SSH-ring terms the levels are
    // +-(t1+t2), +-sqrt(t1^2+t2^2-t1t2) x2 -- pairs remain pairs):
    //  (a) the first excited PAIR is degenerate for BOTH geometries;
    //  (b) what DOES change is the ground state's bond charge: equal
    //      midpoint densities on the uniform ring, piled onto the SHORT
    //      bonds under Kekule alternation.
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

    // Probability within 0.7 bohr of a point (robust midpoint sampling).
    auto blob = [&](const Field3D& f, Vec3d p) {
        double acc = 0.0;
        for (int k = 0; k < g.z.n; ++k) {
            for (int j = 0; j < g.y.n; ++j) {
                for (int i = 0; i < g.x.n; ++i) {
                    const double dx = g.x.coord(i) - p.x;
                    const double dy = g.y.coord(j) - p.y;
                    const double dz = g.z.coord(k) - p.z;
                    if (dx * dx + dy * dy + dz * dz < 0.49) {
                        acc += std::norm(f(i, j, k));
                    }
                }
            }
        }
        return acc;
    };

    struct Result {
        double gap01 = 0.0;
        double pair_split = 0.0;
        double bond_ratio = 0.0;  // side(0-1) midpoint over side(1-2)
    };
    auto solve = [&](const std::vector<Vec3d>& c) {
        const std::vector<double> v =
            ses::soft_coulomb_potential(g, 1.0, soft_a, c);
        const ses::ImaginaryTimePropagator3D relaxer{g, v, 0.05};
        Field3D e0 = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{2.0, 2.0, 1.2},
                                              Vec3d{});
        relaxer.relax(e0, 400);
        Field3D e1 = ses::gaussian_wavepacket(
            g, Vec3d{ring_r, 0.4, 0.0}, Vec3d{1.5, 1.5, 1.2}, Vec3d{});
        relaxer.relax_deflated(e1, {&e0}, 400);
        Field3D e2 = ses::gaussian_wavepacket(
            g, Vec3d{-0.5, ring_r, 0.0}, Vec3d{1.5, 1.5, 1.2}, Vec3d{});
        relaxer.relax_deflated(e2, {&e0, &e1}, 400);
        Result r;
        const double ee0 = ses::mean_energy(e0, v);
        const double ee1 = ses::mean_energy(e1, v);
        const double ee2 = ses::mean_energy(e2, v);
        r.gap01 = std::min(ee1, ee2) - ee0;
        r.pair_split = std::abs(ee2 - ee1);
        const Vec3d m01{0.5 * (c[0].x + c[1].x), 0.5 * (c[0].y + c[1].y), 0.0};
        const Vec3d m12{0.5 * (c[1].x + c[2].x), 0.5 * (c[1].y + c[2].y), 0.0};
        r.bond_ratio = blob(e0, m01) / blob(e0, m12);
        return r;
    };

    const Result u = solve(ring(0.0));
    // delta = +5 deg on even vertices: side 0-1 subtends 50 deg (SHORT),
    // side 1-2 subtends 70 deg (LONG).
    const Result k = solve(ring(5.0));

    EXPECT_GT(u.gap01, 0.01) << "the pair sits above the ground state";
    EXPECT_LT(u.pair_split, 0.01) << "uniform: C6 degeneracy";
    EXPECT_LT(k.pair_split, 0.01) << "Kekule: D3h STILL protects the pair";
    EXPECT_GT(u.bond_ratio, 0.9);
    EXPECT_LT(u.bond_ratio, 1.1);  // uniform: equal bond charge
    EXPECT_GT(k.bond_ratio, u.bond_ratio + 0.08)
        << "Kekule: the short bond hoards the bonding charge";
    EXPECT_GT(k.bond_ratio, 1.1);
}

}  // namespace
