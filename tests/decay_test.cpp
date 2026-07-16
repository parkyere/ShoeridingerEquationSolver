// RED: spontaneous decay via quantum jumps (the Monte-Carlo-wavefunction
// picture): the Schrodinger
// equation itself carries no lifetimes, but the decay RATE follows from the
// computed spectrum through the Einstein A coefficient (atomic units)
//     A = (4/3) alpha^3 omega^3 |<f|r|i>|^2,
// and decay events are a Poisson process: p(jump in dt) = 1 - e^{-A P_e dt}
// weighted by the current excited-state population. Selection rules enter
// automatically: a forbidden channel has |<f|r|i>|^2 = 0, hence A = 0.
//
// Randomness stays OUT of core (the measurement-feature pattern): callers
// inject u in [0,1), so stratified draws give EXACT jump statistics.
//
// Oracles:
//  - harmonic analytic matrix element <1_z|z|0> = 1/sqrt(2 w0), other
//    components zero; same-parity pair -> strength 0 (forbidden);
//  - Einstein A: exact factor check and the omega^3 scaling law;
//  - jumps: exact stratified count for p = 1/4, collapse lands on the
//    ground state, no-jump leaves psi bitwise untouched, and the jump
//    probability is weighted by the ACTUAL excited population.

#include <complex>
#include <core/decay.hpp>
#include <core/field.hpp>
import ses.grid;
import ses.vec;

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

struct HarmonicStates {
    Field3D ground;
    Field3D excited_z;
};

HarmonicStates make_states(const Grid3D& g, double w0) {
    HarmonicStates s{Field3D{g}, Field3D{g}};
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                const double env = std::exp(-0.5 * w0 * (x * x + y * y + z * z));
                s.ground(i, j, k) = std::complex<double>{env, 0.0};
                s.excited_z(i, j, k) = std::complex<double>{z * env, 0.0};
            }
        }
    }
    ses::normalize(s.ground);
    ses::normalize(s.excited_z);
    return s;
}

Grid3D cube(double lo, double hi, int n) {
    return Grid3D{Grid1D{lo, hi, n}, Grid1D{lo, hi, n}, Grid1D{lo, hi, n}};
}

TEST(DipoleMatrixElement, HarmonicAnalyticValue) {
    const double w0 = 1.0;
    const Grid3D g = cube(-8.0, 8.0, 32);
    const HarmonicStates s = make_states(g, w0);

    const ses::DipoleMatrixElement d =
        ses::dipole_matrix_element(s.excited_z, s.ground);
    EXPECT_NEAR(std::abs(d.z.real()), 1.0 / std::sqrt(2.0 * w0), 1e-6);
    EXPECT_NEAR(d.z.imag(), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(d.x), 0.0, 1e-10);
    EXPECT_NEAR(std::abs(d.y), 0.0, 1e-10);
    EXPECT_NEAR(ses::dipole_strength_sq(d), 1.0 / (2.0 * w0), 1e-6);
}

TEST(DipoleMatrixElement, SameParityPairIsForbidden) {
    const Grid3D g = cube(-8.0, 8.0, 32);
    const HarmonicStates s = make_states(g, 1.0);
    const ses::DipoleMatrixElement d = ses::dipole_matrix_element(s.ground, s.ground);
    EXPECT_NEAR(ses::dipole_strength_sq(d), 0.0, 1e-10);
}

TEST(EinsteinA, FactorsAndCubicScaling) {
    const double a = ses::einstein_a(1.0, 0.5);
    const double alpha3 = std::pow(ses::kFineStructureConstant, 3.0);
    EXPECT_DOUBLE_EQ(a, (4.0 / 3.0) * alpha3 * 0.5);
    EXPECT_DOUBLE_EQ(ses::einstein_a(2.0, 0.5), 8.0 * a);  // omega^3 law
    EXPECT_EQ(ses::einstein_a(1.0, 0.0), 0.0);             // forbidden channel
}

TEST(QuantumJump, StratifiedStatisticsAreExact) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const HarmonicStates s = make_states(g, 1.0);

    // gamma * dt = -ln(3/4)  ->  p = 1/4 exactly.
    const double gamma_dt = -std::log(0.75);
    int jumps = 0;
    const int kDraws = 1000;
    for (int k = 0; k < kDraws; ++k) {
        Field3D psi = s.excited_z;  // P_e = 1
        const double u = (k + 0.5) / kDraws;
        const ses::JumpResult r =
            ses::quantum_jump(psi, s.excited_z, s.ground, gamma_dt, 1.0, u);
        EXPECT_NEAR(r.p_jump, 0.25, 1e-9);
        if (r.jumped) {
            ++jumps;
            // The collapse lands on the ground state.
            EXPECT_NEAR(std::norm(ses::inner_product(s.ground, psi)), 1.0, 1e-9);
            EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
        }
    }
    EXPECT_EQ(jumps, 250);
}

TEST(QuantumJump, NoJumpLeavesPsiUntouched) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const HarmonicStates s = make_states(g, 1.0);
    Field3D psi = s.excited_z;
    const Field3D before = psi;
    const ses::JumpResult r =
        ses::quantum_jump(psi, s.excited_z, s.ground, 0.1, 1.0, 0.999);
    EXPECT_FALSE(r.jumped);
    for (std::size_t i = 0; i < psi.data().size(); ++i) {
        EXPECT_EQ(psi.data()[i].real(), before.data()[i].real());
        EXPECT_EQ(psi.data()[i].imag(), before.data()[i].imag());
    }
}

TEST(QuantumJump, ProbabilityIsPopulationWeighted) {
    const Grid3D g = cube(-8.0, 8.0, 16);
    const HarmonicStates s = make_states(g, 1.0);

    // psi = sqrt(0.3) |e> + sqrt(0.7) |g>  ->  P_e = 0.3.
    Field3D psi{g};
    const double ce = std::sqrt(0.3);
    const double cg = std::sqrt(0.7);
    for (std::size_t i = 0; i < psi.data().size(); ++i) {
        psi.data()[i] = ce * s.excited_z.data()[i] + cg * s.ground.data()[i];
    }

    const double gamma_dt = 0.5;
    const ses::JumpResult r =
        ses::quantum_jump(psi, s.excited_z, s.ground, gamma_dt, 1.0, 0.999);
    EXPECT_NEAR(r.p_jump, 1.0 - std::exp(-gamma_dt * 0.3), 1e-6);
}

}  // namespace
