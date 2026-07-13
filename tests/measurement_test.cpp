// RED: soft position measurement -- the user-facing "observe the electron"
// feature. NOT a Dirac-delta collapse: the post-measurement state is the old
// psi multiplied by a Gaussian mask of width sigma_m (a Gaussian POVM),
// renormalized, and then handed back to the TDSE.
//
// Design: randomness stays OUT of core. sample_collapse_index takes a
// uniform draw u in [0,1) and inverts the discrete CDF of |psi|^2 (flat-index
// order), so every test is deterministic.
//
// Sharp oracles:
//  - CDF inversion on hand-built two-cell densities (exact indices);
//  - stratified draws u_k=(k+1/2)/K reproduce cell probabilities EXACTLY;
//  - Gaussian x Gaussian is analytic: with our amplitude convention
//    e^{-(x-c)^2/(4 s^2)}, collapsing a packet of width s_pre centered at r0
//    with a mask of width s_m at r_m gives
//        center = (r_m/s_m^2 + r0/s_pre^2) / (1/s_m^2 + 1/s_pre^2)
//        1/s_post^2 = 1/s_m^2 + 1/s_pre^2;
//  - the uncertainty principle on screen: a SHARPER measurement re-spreads
//    FASTER under subsequent free evolution.

#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/measurement.hpp>
#include <core/observables.hpp>
#include <core/potential.hpp>
#include <core/propagator.hpp>
#include <core/simulation.hpp>
#include <core/vec.hpp>
#include <core/wavepacket.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using ses::Complex;
using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;
using ses::Vec3d;

TEST(SampleCollapseIndex, InvertsTheDiscreteCdf) {
    // Two occupied cells in flat order: flat(3,0,2)=35 carries 3/4 of the
    // probability, flat(1,2,3)=57 carries 1/4.
    const Grid1D axis{0.0, 4.0, 4};
    const Grid3D g{axis, axis, axis};
    Field3D psi{g};
    psi(3, 0, 2) = Complex<double>{std::sqrt(3.0), 0.0};
    psi(1, 2, 3) = Complex<double>{1.0, 0.0};

    EXPECT_EQ(ses::sample_collapse_index(psi, 0.0), 35);
    EXPECT_EQ(ses::sample_collapse_index(psi, 0.5), 35);
    EXPECT_EQ(ses::sample_collapse_index(psi, 0.7499), 35);
    EXPECT_EQ(ses::sample_collapse_index(psi, 0.7501), 57);
    EXPECT_EQ(ses::sample_collapse_index(psi, 0.9999), 57);
}

TEST(SampleCollapseIndex, StratifiedDrawsReproduceProbabilitiesExactly) {
    const Grid1D axis{0.0, 4.0, 4};
    const Grid3D g{axis, axis, axis};
    Field3D psi{g};
    psi(3, 0, 2) = Complex<double>{std::sqrt(3.0), 0.0};  // p = 3/4
    psi(1, 2, 3) = Complex<double>{1.0, 0.0};             // p = 1/4

    const int kDraws = 1000;
    int count_b = 0;
    for (int k = 0; k < kDraws; ++k) {
        const double u = (k + 0.5) / kDraws;
        if (ses::sample_collapse_index(psi, u) == 35) {
            ++count_b;
        }
    }
    EXPECT_EQ(count_b, 750);  // exact for stratified sampling
}

TEST(SampleEnergyEigenstate, InvertsThePopulationCdfWithIncompleteManifold) {
    // Projective energy measurement over eigenstate populations that do NOT
    // sum to 1: the tracked bound manifold is incomplete, so the deficit
    // 1 - sum(P) is the "outside the manifold" (continuum) outcome, index -1.
    const std::vector<double> pops = {0.5, 0.3};  // sum 0.8, deficit 0.2
    EXPECT_EQ(ses::sample_energy_eigenstate(pops, 0.0), 0);
    EXPECT_EQ(ses::sample_energy_eigenstate(pops, 0.4999), 0);
    EXPECT_EQ(ses::sample_energy_eigenstate(pops, 0.5), 1);
    EXPECT_EQ(ses::sample_energy_eigenstate(pops, 0.7999), 1);
    EXPECT_EQ(ses::sample_energy_eigenstate(pops, 0.8), -1);  // into the deficit
    EXPECT_EQ(ses::sample_energy_eigenstate(pops, 0.9999), -1);
}

TEST(SampleEnergyEigenstate, CompleteManifoldNeverEscapesAndIsExactStratified) {
    // Sum = 1: every draw lands on an eigenstate, and stratified draws
    // u_k = (k + 1/2)/K reproduce the populations exactly.
    const std::vector<double> pops = {0.25, 0.25, 0.5};
    const int kDraws = 1000;
    int counts[3] = {0, 0, 0};
    for (int k = 0; k < kDraws; ++k) {
        const int idx = ses::sample_energy_eigenstate(pops, (k + 0.5) / kDraws);
        ASSERT_GE(idx, 0);
        ASSERT_LT(idx, 3);
        ++counts[idx];
    }
    EXPECT_EQ(counts[0], 250);
    EXPECT_EQ(counts[1], 250);
    EXPECT_EQ(counts[2], 500);
}

TEST(CollapseWavepacket, MatchesAnalyticGaussianPosterior) {
    // Broad packet (s_pre = 2, r0 = 0) collapsed at r_m with s_m = 0.5:
    // per-axis center r_m * (1/s_m^2)/(1/s_m^2 + 1/s_pre^2) = r_m * 16/17,
    // width s_post = 1/sqrt(4.25).
    const Grid1D axis{-8.0, 8.0, 64};
    const Grid3D g{axis, axis, axis};
    Field3D psi = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{2.0, 2.0, 2.0}, Vec3d{});

    const Vec3d rm{1.0, -0.5, 0.5};
    ses::collapse_wavepacket(psi, rm, 0.5);

    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);

    const double pull = (1.0 / 0.25) / (1.0 / 0.25 + 1.0 / 4.0);  // 16/17
    const Vec3d r = ses::mean_position(psi);
    EXPECT_NEAR(r.x, rm.x * pull, 0.01);
    EXPECT_NEAR(r.y, rm.y * pull, 0.01);
    EXPECT_NEAR(r.z, rm.z * pull, 0.01);

    const double s_post = 1.0 / std::sqrt(1.0 / 0.25 + 1.0 / 4.0);
    const Vec3d s = ses::sigma_position(psi);
    EXPECT_NEAR(s.x, s_post, 0.02);
    EXPECT_NEAR(s.y, s_post, 0.02);
    EXPECT_NEAR(s.z, s_post, 0.02);
}

TEST(CollapseWavepacket, SharperMeasurementRespreadsFaster) {
    // Uncertainty principle: post-collapse width s_m implies momentum spread
    // ~1/(2 s_m), so the sharper collapse disperses faster under V = 0.
    const Grid1D axis{-8.0, 8.0, 32};
    const Grid3D g{axis, axis, axis};
    const std::vector<double> free_v(static_cast<std::size_t>(g.size()), 0.0);
    const ses::SplitOperator3D prop{g, free_v, 0.04};

    auto respread_ratio = [&](double sigma_m) {
        Field3D psi = ses::gaussian_wavepacket(g, Vec3d{}, Vec3d{2.0, 2.0, 2.0}, Vec3d{});
        ses::collapse_wavepacket(psi, Vec3d{}, sigma_m);
        const double s0 = ses::sigma_position(psi).x;
        Field3D evolved = psi;
        prop.step(evolved, 25);  // t = 1
        return ses::sigma_position(evolved).x / s0;
    };

    const double sharp = respread_ratio(0.4);
    const double broad = respread_ratio(0.8);
    EXPECT_GT(sharp, broad * 1.5);  // decisively faster, not marginally
}

TEST(PovmOutcomeDensity, IsRawDensityBlurredByTheKrausWidth) {
    // POVM consistency: E_c = M_c^dag M_c with M_c = e^{-(r-c)^2/(4 s^2)}
    // means P(c) ~ |psi|^2 convolved with e^{-(r-c)^2/(2 s^2)} -- a Gaussian
    // of std sigma_m. A single occupied cell gives the separable kernel
    // itself: exact per-cell oracles.
    const Grid1D axis{0.0, 16.0, 16};
    const Grid3D g{axis, axis, axis};
    Field3D psi{g};
    psi(8, 8, 8) = Complex<double>{1.0, 0.0};

    const std::vector<double> d = ses::povm_outcome_density(psi, 1.0);

    // Per-axis taps t = -4..4 (radius 4 sigma), w_t = e^{-t^2/2} / S.
    double s = 1.0;
    for (int t = 1; t <= 4; ++t) {
        s += 2.0 * std::exp(-0.5 * t * t);
    }
    const auto at = [&](int i, int j, int k) {
        return d[static_cast<std::size_t>(g.flat(i, j, k))];
    };
    EXPECT_NEAR(at(8, 8, 8), 1.0 / (s * s * s), 1e-12);
    EXPECT_NEAR(at(9, 8, 8) / at(8, 8, 8), std::exp(-0.5), 1e-12);

    // The blur conserves total probability (periodic, normalized kernel).
    double total = 0.0;
    for (double p : d) {
        total += p;
    }
    EXPECT_NEAR(total, 1.0, 1e-12);
}

TEST(SamplePovmIndex, CanLandWhereRawDensityHasANode) {
    // Two occupied cells straddle an empty one. Raw CDF sampling can never
    // return the node cell; a detector of resolution sigma_m ~ h must be
    // able to (its outcome density there is nonzero).
    const Grid1D axis{0.0, 16.0, 16};
    const Grid3D g{axis, axis, axis};
    Field3D psi{g};
    psi(7, 8, 8) = Complex<double>{1.0, 0.0};
    psi(9, 8, 8) = Complex<double>{1.0, 0.0};

    const int node = g.flat(8, 8, 8);
    const std::vector<double> d = ses::povm_outcome_density(psi, 1.0);
    EXPECT_GT(d[static_cast<std::size_t>(node)], 0.0);

    bool raw_hits_node = false;
    bool povm_hits_node = false;
    const int kDraws = 2000;
    for (int k = 0; k < kDraws; ++k) {
        const double u = (k + 0.5) / kDraws;
        raw_hits_node |= ses::sample_collapse_index(psi, u) == node;
        povm_hits_node |= ses::sample_povm_index(psi, 1.0, u) == node;
    }
    EXPECT_FALSE(raw_hits_node);
    EXPECT_TRUE(povm_hits_node);
}

TEST(SimulationMeasure, CollapsesAtTheSampledCellAndKeepsTime) {
    const Grid1D axis{-8.0, 8.0, 16};
    const Grid3D g{axis, axis, axis};
    ses::WavepacketSimulation sim{ses::WavepacketSimulation::Config{
        g,
        ses::harmonic_potential(g, 1.0, Vec3d{}),
        Vec3d{1.0, 0.0, 0.0},
        Vec3d{1.5, 1.5, 1.5},
        Vec3d{},
        0.02,
    }};
    sim.advance(3);
    const double t_before = sim.time();

    // The center the sim reports must be the coordinate of exactly the cell
    // the POVM-consistent inversion picks for the same draw and detector
    // width (outcomes sample the sigma_m-blurred density, not raw |psi|^2).
    const int expected_idx = ses::sample_povm_index(sim.psi(), 0.6, 0.37);
    const Vec3d c = sim.measure(0.37, 0.6);

    const int nx = g.x.n;
    const int ny = g.y.n;
    const int i = expected_idx % nx;
    const int j = (expected_idx / nx) % ny;
    const int k = expected_idx / (nx * ny);
    EXPECT_DOUBLE_EQ(c.x, g.x.coord(i));
    EXPECT_DOUBLE_EQ(c.y, g.y.coord(j));
    EXPECT_DOUBLE_EQ(c.z, g.z.coord(k));

    EXPECT_EQ(sim.time(), t_before);  // measurement is instantaneous
    EXPECT_NEAR(ses::norm_sq(sim.psi()), 1.0, 1e-12);

    // The cloud actually moved toward the collapse center.
    const Vec3d r = ses::mean_position(sim.psi());
    EXPECT_LT(std::abs(r.x - c.x), 1.0);
    EXPECT_LT(std::abs(r.y - c.y), 1.0);
    EXPECT_LT(std::abs(r.z - c.z), 1.0);
}

}  // namespace
