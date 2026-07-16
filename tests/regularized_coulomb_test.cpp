// The 3D spectrum of the potential the APP actually diagonalizes:
// regularized_coulomb_potential (bare -Z/r with the singular nucleus cell
// replaced by its analytic average -Z*kCoulombCellAverage/h). potential_test
// only checks its VALUES; nothing asserted its resulting SPECTRUM -- so a
// wrong kCoulombCellAverage or any spectrum-shifting bug in the 3D Coulomb
// path passed every test. This is that missing net for the project's
// headline claim (textbook hydrogen: -0.5 Ha 1s, n=2 degeneracy).
//
// Oracles (grid-honest -- the coarse cusp lifts s-states ~1 eV, so absolute
// bands carry that margin while the SYMMETRY oracles stay tight):
//  - relaxed ground state E0 near the bare-hydrogen -0.5 Ha and clearly
//    DEEPER than soft-Coulomb (~-0.27), which a wrong cell-average breaks;
//  - the three 2p orbitals are EXACTLY degenerate (cubic-grid symmetry);
//  - n=1 sits well below n=2; 2s and 2p share the n=2 shell (approx SO(4)).

#include <core/field.hpp>
import ses.grid;
#include <core/imaginary_time.hpp>
#include <core/observables.hpp>
#include <core/radial.hpp>
import ses.vec;

#include <gtest/gtest.h>

#include <cmath>
#include <vector>
import ses.harmonics;
import ses.wavepacket;
import ses.potential;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::RadialGrid;
using ses::Vec3d;

// 32^3 / +-12 Bohr (h = 0.75): power-of-two per axis (the hand-rolled radix-2
// FFT requires it) and the box holds 1s tightly plus the n=2 orbitals' tails.
const Grid3D kGrid{Grid1D{-12.0, 12.0, 32}, Grid1D{-12.0, 12.0, 32},
                   Grid1D{-12.0, 12.0, 32}};

const std::vector<double>& reg_potential() {
    static const std::vector<double> v =
        ses::regularized_coulomb_potential(kGrid, 1.0, Vec3d{});
    return v;
}

// <H> of a synthesized (u_nl/r) Y_lm orbital under the regularized potential:
// the app's exact construction. Bare -1/r radial solve feeds the synthesis
// (the two differ only in the single nucleus cell).
double synth_energy(int level_l, int node_k, int m) {
    const RadialGrid rg{20.0, 3999};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = -1.0 / rg.r(i);
    }
    const ses::RadialState st =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, level_l),
                               node_k);
    const Field3D psi = ses::synthesize_orbital(kGrid, rg, st.u, level_l, m);
    return ses::mean_energy(psi, reg_potential());
}

TEST(RegularizedCoulomb3D, GroundStateNearBareHydrogenAndDeeperThanSoft) {
    Field3D psi =
        ses::gaussian_wavepacket(kGrid, Vec3d{}, Vec3d{1.0, 1.0, 1.0}, Vec3d{});
    ses::ImaginaryTimePropagator3D relaxer{kGrid, reg_potential(), 0.02};
    relaxer.relax(psi, 900);  // tau = 18: well converged to the 1s
    const double e0 = ses::mean_energy(psi, reg_potential());
    EXPECT_LT(e0, 0.0);  // bound
    // Anchored at the analytic bare-hydrogen -0.5 Ha. On this grid the
    // correct cell-average relaxes to -0.473 (a +0.027 Ha cusp gap); the
    // 0.08 band absorbs that yet is tight enough to CATCH a wrong constant
    // (probed: a halved kCoulombCellAverage gives -0.400, outside; a doubled
    // one -1.12, far outside).
    EXPECT_NEAR(e0, -0.5, 0.08);
    // Decisively deeper than the soft-Coulomb well (~-0.27 Ha): the two
    // regularizations are physically different and must not be confused.
    EXPECT_LT(e0, -0.35);
}

TEST(RegularizedCoulomb3D, TwoPTripletIsDegenerate) {
    // 2p_x, 2p_y, 2p_z map into each other under the cubic grid's axis
    // symmetries, so their energies must match to fp precision -- a sharp
    // net against any axis-asymmetric bug in potential/synthesis/energy.
    const double e_pz = synth_energy(1, 0, 0);
    const double e_px = synth_energy(1, 0, 1);
    const double e_py = synth_energy(1, 0, -1);
    EXPECT_NEAR(e_px, e_pz, 1e-5);
    EXPECT_NEAR(e_py, e_pz, 1e-5);
}

TEST(RegularizedCoulomb3D, ShellOrderingAndN2Near) {
    const double e_1s = synth_energy(0, 0, 0);
    const double e_2s = synth_energy(0, 1, 0);
    const double e_2p = synth_energy(1, 0, 0);
    // n=1 well below n=2.
    EXPECT_LT(e_1s, e_2s - 0.2);
    EXPECT_LT(e_1s, e_2p - 0.2);
    // n=2 shell: both near the -1/8 = -0.125 Ha Rydberg level. 2s carries the
    // coarse-grid cusp lift that 2p (zero at the nucleus) does not, so the
    // SO(4) 2s=2p degeneracy holds only approximately on the grid.
    EXPECT_NEAR(e_2p, -0.125, 0.03);
    EXPECT_NEAR(e_2s, -0.125, 0.08);
}

}  // namespace
