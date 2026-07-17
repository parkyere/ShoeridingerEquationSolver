// RED: specification for the post-collapse eigenstate-error flush.
//
// Every collapse target (decay jump, full energy measurement) is synthesized
// from a RADIAL eigenstate of bare -Z/r sampled onto the 3D grid ((u/r) Y_lm),
// so it is NOT an eigenstate of the grid Hamiltonian the propagator actually
// exponentiates (spectral kinetic + regularized Coulomb). The mismatch is
// dominated by nucleus-cell cusp junk at Ha-scale energies: post-collapse the
// density shimmers and <H> sits above the relaxed grid level.
//
// The correction under test: a FIXED small imaginary-time burst right after
// the collapse (director budget dtau = 0.05, 6 steps, tau = 0.3). Contract:
//   (1) it flushes the junk -- Var(H) = <H^2> - <H>^2 (zero iff eigenstate)
//       drops by >= 10x, because the junk's Ha-scale components damp as
//       e^{-dE tau} while bound-bound mismatch (dE <= 1 Ha) barely moves;
//   (2) it preserves the state's identity in the SYNTHESIZED basis, which is
//       what every population / rate / measurement CDF projects onto;
//   (3) it opens no non-radiative channel: 2p -> 1s stays parity-blocked,
//       and a 2s target's 1s admixture must not grow meaningfully at this
//       budget -- the design forgoes deflation, this test is that license.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <vector>

import ses.imaginary_time;
import ses.observables;
import ses.radial;
import ses.grid;
import ses.vec;
import ses.field;
import ses.fft;
import ses.spectral;
import ses.harmonics;
import ses.potential;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::RadialGrid;
using ses::Vec3d;

// Same lattice as regularized_coulomb_test: 32^3 / +-12 Bohr (h = 0.75)
// holds 1s and the n=2 tails; the coarse h makes the cusp junk LARGE, so the
// flush is exercised against a worst-case artifact.
const Grid3D kGrid{Grid1D{-12.0, 12.0, 32}, Grid1D{-12.0, 12.0, 32},
                   Grid1D{-12.0, 12.0, 32}};

// The director's burst budgets (kRelaxDtau = 0.05): 6 steps (tau = 0.3) for
// excited targets (bounded on purpose -- longer drains them downward), 24
// steps (tau = 1.2) for the 1s target, which is the ITP fixed point and so
// cannot be overshot: 6 steps left a user-visible offset from the grid
// ground state.
constexpr double kBurstDtau = 0.05;
constexpr int kBurstSteps = 6;
constexpr int kBurstStepsGround = 24;

const std::vector<double>& reg_potential() {
    static const std::vector<double> v =
        ses::regularized_coulomb_potential(kGrid, 1.0, Vec3d{});
    return v;
}

// The app's exact collapse-target construction: bare -1/r radial solve,
// sampled onto the grid, normalized.
Field3D synth(int l, int node_k, int m) {
    const RadialGrid rg{20.0, 3999};
    std::vector<double> vr(static_cast<std::size_t>(rg.n));
    for (int i = 0; i < rg.n; ++i) {
        vr[static_cast<std::size_t>(i)] = -1.0 / rg.r(i);
    }
    const ses::RadialState st =
        ses::radial_eigenstate(rg, ses::radial_hamiltonian(rg, vr, l), node_k);
    return ses::synthesize_orbital(kGrid, rg, st.u, l, m);
}

// H psi = IFFT((k^2/2) FFT(psi)) + V psi -- the operator the split-step
// propagator exponentiates, applied once.
Field3D apply_h(const Field3D& f, const std::vector<double>& v) {
    Field3D t = f;
    ses::fft(t);
    const Grid3D& g = f.grid();
    const std::vector<double> kx = ses::wavenumbers(g.x);
    const std::vector<double> ky = ses::wavenumbers(g.y);
    const std::vector<double> kz = ses::wavenumbers(g.z);
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double kxx = kx[static_cast<std::size_t>(i)];
                const double kyy = ky[static_cast<std::size_t>(j)];
                const double kzz = kz[static_cast<std::size_t>(k)];
                t(i, j, k) *= 0.5 * (kxx * kxx + kyy * kyy + kzz * kzz);
            }
        }
    }
    ses::ifft(t);
    for (std::size_t i = 0; i < t.data().size(); ++i) {
        t.data()[i] += v[i] * f.data()[i];
    }
    return t;
}

// Var(H) = <H^2> - <H>^2 over a normalized-or-not field (ratios cancel the
// scale). <H^2> = <H psi|H psi> since H is Hermitian.
double energy_variance(const Field3D& f, const std::vector<double>& v) {
    const Field3D hf = apply_h(f, v);
    const double n2 = ses::norm_sq(f);
    const double e = ses::inner_product(f, hf).real() / n2;
    const double e2 = ses::norm_sq(hf) / n2;
    return e2 - e * e;
}

double population(const Field3D& phi, const Field3D& psi) {
    return std::norm(ses::inner_product(phi, psi));
}

void burst(Field3D& psi) {
    const ses::ImaginaryTimePropagator3D relaxer{kGrid, reg_potential(),
                                                 kBurstDtau};
    relaxer.relax(psi, kBurstSteps);  // renormalizes every step
}

TEST(EigenstateFlush, OneSBurstFlushesJunkTenfoldAndKeepsIdentity) {
    Field3D psi = synth(0, 0, 0);
    const Field3D psi0 = psi;
    const double var0 = energy_variance(psi, reg_potential());
    const double e0 = ses::mean_energy(psi, reg_potential());
    // Precondition documents the artifact: the sampled 1s is measurably NOT a
    // grid eigenstate (an exact eigenstate reads Var(H) = 0).
    ASSERT_GT(var0, 1e-3);

    burst(psi);

    const double var1 = energy_variance(psi, reg_potential());
    const double e1 = ses::mean_energy(psi, reg_potential());
    EXPECT_LT(var1, 0.1 * var0);          // (1) the junk is flushed
    EXPECT_GT(population(psi0, psi), 0.97);  // (2) identity in sampled basis
    EXPECT_LT(e1, e0);                    // ITP is monotone toward the grid 1s
}

TEST(EigenstateFlush, GroundDeepBurstConvergesToTheGridGroundState) {
    Field3D psi = synth(0, 0, 0);
    const Field3D psi0 = psi;
    const ses::ImaginaryTimePropagator3D relaxer{kGrid, reg_potential(),
                                                 kBurstDtau};
    // Converged reference: the same flow run to its fixed point.
    Field3D ref = psi;
    relaxer.relax(ref, 400);
    const double e_ref = ses::mean_energy(ref, reg_potential());

    relaxer.relax(psi, kBurstStepsGround);

    // "Properly converged": the deep burst must land within 5 mHa of the
    // grid ground state (the 6-step budget visibly does not) while the
    // state still reads as 1s in the sampled basis.
    EXPECT_NEAR(ses::mean_energy(psi, reg_potential()), e_ref, 5e-3);
    EXPECT_GT(population(psi0, psi), 0.99);
}

TEST(EigenstateFlush, TwoPzBurstOpensNoParityForbiddenOneSChannel) {
    Field3D psi = synth(1, 0, 0);
    const Field3D psi0 = psi;
    const Field3D s1 = synth(0, 0, 0);

    burst(psi);

    // (3) no non-radiative 2p -> 1s decay: the synthesized 2p_z is exactly
    // z-odd, the Hamiltonian is z-even, so the 1s admixture stays at double
    // rounding -- a fake jump channel would show up here first.
    EXPECT_LT(population(s1, psi), 1e-12);
    EXPECT_GT(population(psi0, psi), 0.99);  // 2p has no cusp: barely moves
}

TEST(EigenstateFlush, TwoSBurstDoesNotAmplifyGroundAdmixture) {
    Field3D psi = synth(0, 1, 0);
    const Field3D s1 = synth(0, 0, 0);
    const double p1_before = population(s1, psi);
    const double var0 = energy_variance(psi, reg_potential());

    burst(psi);

    // 2s shares the 1s parity sector, so this is the worst case for the
    // no-deflation design: at tau = 0.3 the 1s amplitude gain is at most
    // e^{0.35 tau} ~ 1.11, i.e. the admixture must stay the same order.
    EXPECT_LT(population(s1, psi), 2.0 * p1_before + 1e-8);
    EXPECT_LT(energy_variance(psi, reg_potential()), 0.1 * var0);
}

}  // namespace
