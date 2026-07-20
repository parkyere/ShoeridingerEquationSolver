// RED: the H2+ scene's state seeds -- the symmetry-resolved known molecular
// orbitals + an arbitrary random-shaped wavefunction.
//
// The MO seeds are fed to the deflated imaginary-time chain: a symmetry
// sector's seed converges to the LOWEST orbital of that irrep (symmetry
// orthogonality keeps sectors apart; deflation handles the same-irrep
// tower). The random seed is a legitimate normalized state of arbitrary
// (symmetry-broken) shape -- what the user drops in to watch evolve.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <vector>

import ses.scenario.molecular_seed;
import ses.field;
import ses.grid;
import ses.imaginary_time;
import ses.observables;
import ses.potential;
import ses.vec;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;
using ses_shell::molecular_orbital_seed;
using ses_shell::MolOrbital;
using ses_shell::random_molecular_seed;

// h = 0.5, origin at index 16; mirror of index i (1..15) is 32 - i, i = 16
// is the plane itself (coord 0). Index 0 (coord -8) has no in-range mirror.
Grid3D grid() {
    const Grid1D ax{-8.0, 8.0, 32};
    return {ax, ax, ax};
}

// Plane density (node candidate) and bulk density on an axis-`ax` mid-plane.
void plane_vs_bulk(const Field3D& f, int axis, int mid, double& node,
                   double& bulk) {
    const Grid3D& g = f.grid();
    node = 0.0;
    bulk = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const int idx = axis == 0 ? i : (axis == 1 ? j : k);
                const double w = std::norm(f(i, j, k));
                if (idx == mid) {
                    node += w;
                } else {
                    bulk += w;
                }
            }
        }
    }
}

TEST(MolecularSeed, SigmaGIsEvenAndNodeless) {
    const Grid3D g = grid();
    const Field3D s = molecular_orbital_seed(g, MolOrbital::SigmaG);
    EXPECT_NEAR(ses::norm_sq(s), 1.0, 1e-9);
    // no nodal plane at the center: the x = 0 mid-plane carries real density.
    double node = 0.0;
    double bulk = 0.0;
    plane_vs_bulk(s, 0, 16, node, bulk);
    EXPECT_GT(node, 0.01 * bulk) << "sigma_g has NO node between the nuclei";
}

TEST(MolecularSeed, SigmaUIsOddUnderXReflection) {
    const Grid3D g = grid();
    const Field3D s = molecular_orbital_seed(g, MolOrbital::SigmaU);
    EXPECT_NEAR(ses::norm_sq(s), 1.0, 1e-9);
    // antisymmetric in x about index 16, with a node in the x = 0 plane.
    double anti = 0.0;
    double mag = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 1; i < 16; ++i) {
                const std::complex<double> a = s(i, j, k);
                const std::complex<double> b = s(32 - i, j, k);
                anti += std::norm(a + b);  // 0 iff b == -a
                mag += std::norm(a);
            }
        }
    }
    EXPECT_LT(anti, 1e-6 * mag) << "sigma_u is x-odd (perpendicular node)";
    double node = 0.0;
    double bulk = 0.0;
    plane_vs_bulk(s, 0, 16, node, bulk);
    EXPECT_LT(node, 1e-6 * bulk);
}

TEST(MolecularSeed, PiUyIsOddUnderYReflection) {
    const Grid3D g = grid();
    const Field3D s = molecular_orbital_seed(g, MolOrbital::PiUy);
    EXPECT_NEAR(ses::norm_sq(s), 1.0, 1e-9);
    double node = 0.0;
    double bulk = 0.0;
    plane_vs_bulk(s, 1, 16, node, bulk);
    EXPECT_LT(node, 1e-6 * bulk) << "pi_u has a nodal plane containing the axis";
    // and it is EVEN in x (not a sigma_u): the x = 0 plane is NOT a node.
    double xnode = 0.0;
    double xbulk = 0.0;
    plane_vs_bulk(s, 0, 16, xnode, xbulk);
    EXPECT_GT(xnode, 0.01 * xbulk);
}

TEST(MolecularSeed, RandomIsNormalizedDeterministicAndArbitrary) {
    const Grid3D g = grid();
    const Field3D a = random_molecular_seed(g, 42);
    const Field3D b = random_molecular_seed(g, 42);
    const Field3D c = random_molecular_seed(g, 7);
    EXPECT_NEAR(ses::norm_sq(a), 1.0, 1e-9);
    // deterministic: same seed -> identical field.
    double diff_ab = 0.0;
    double diff_ac = 0.0;
    for (int i = 0; i < g.size(); ++i) {
        diff_ab += std::norm(a.data()[static_cast<std::size_t>(i)] -
                             b.data()[static_cast<std::size_t>(i)]);
        diff_ac += std::norm(a.data()[static_cast<std::size_t>(i)] -
                             c.data()[static_cast<std::size_t>(i)]);
    }
    EXPECT_LT(diff_ab, 1e-18) << "same seed reproduces the field";
    EXPECT_GT(diff_ac, 0.1) << "different seeds give different shapes";
    // Arbitrary shape: NOT an inversion eigenstate. The parity overlap
    // ratio |<psi|P psi>| / <psi|psi> is 1 for an even/odd state and well
    // below 1 for a symmetry-broken one.
    std::complex<double> ov{};
    double self = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const int pi = (32 - i) % 32;
                const int pj = (32 - j) % 32;
                const int pk = (32 - k) % 32;
                ov += std::conj(a(i, j, k)) * a(pi, pj, pk);
                self += std::norm(a(i, j, k));
            }
        }
    }
    EXPECT_LT(std::abs(ov) / self, 0.9)
        << "a random shape is not an inversion eigenstate";
}

TEST(MolecularSeed, SeedsRelaxToTheOrderedH2plusOrbitals) {
    // 32^3 license grid, R = 2 (nuclei at +-1, on grid points).
    const Grid3D g = grid();
    const std::vector<Vec3d> nuc = {{-1.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    const std::vector<double> v =
        ses::regularized_coulomb_potential(g, 1.0, nuc);
    const ses::ImaginaryTimePropagator3D relaxer{g, v, 0.05};

    Field3D sg = molecular_orbital_seed(g, MolOrbital::SigmaG);
    relaxer.relax(sg, 800);
    const double e_sg = ses::mean_energy(sg, v);

    Field3D su = molecular_orbital_seed(g, MolOrbital::SigmaU);
    relaxer.relax_deflated(su, {&sg}, 800);
    const double e_su = ses::mean_energy(su, v);

    Field3D pu = molecular_orbital_seed(g, MolOrbital::PiUy);
    relaxer.relax_deflated(pu, {&sg}, 800);
    const double e_pu = ses::mean_energy(pu, v);

    // MO energy ordering at R = 2: 1sigma_g < 1sigma_u* < 1pi_u.
    EXPECT_LT(e_sg, e_su) << "bonding below antibonding";
    EXPECT_LT(e_su, e_pu) << "pi_u sits above the antibonding sigma_u";
    // pi_u KEEPS its axis-containing node through the relax (symmetry).
    double node = 0.0;
    double bulk = 0.0;
    plane_vs_bulk(pu, 1, 16, node, bulk);
    EXPECT_LT(node, 0.05 * bulk) << "pi_u relaxes with its nodal plane intact";
}

}  // namespace
