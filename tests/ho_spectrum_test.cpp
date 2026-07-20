// RED: the linear-combination spectra. 1D: an eigenstate is ONE line, a
// coherent state's weighted mean energy reconstructs <H>. 2D (Fock-
// Darwin, lattice gauge): the relaxed ground is the single bottom line
// at Omega, and at B != 0 the gauge-corrected decomposition still
// reconstructs the lattice <H> -- the chirality assignment w_-+ =
// Omega -+ B/2 is pinned by that reconstruction.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>

import ses.ladder;
import ses.lattice2d;
import ses.field;
import ses.grid;
import ses.observables;
import ses.wavepacket;

namespace {

TEST(HoSpectrum, OneDEigenstateIsOneLineAndCoherentReconstructs) {
    const ses::Grid1D g{-40.0, 40.0, 2048};
    const double omega = 0.5;
    const ses::Field1D n3 = ses::ho_eigenstate(g, omega, 3);
    const auto lines = ses::ho1d_spectrum(n3, omega, 10.0);
    ASSERT_GE(lines.size(), 5u);
    double total = 0.0;
    for (const auto& [e, w] : lines) {
        total += w;
    }
    EXPECT_NEAR(total, 1.0, 1e-6);
    EXPECT_NEAR(lines[3].first, 3.5 * omega, 1e-12);
    EXPECT_GT(lines[3].second, 0.999);  // ONE line

    // Coherent state: Poissonian weights whose mean rebuilds <H>.
    const double sig = 1.0 / std::sqrt(2.0 * omega);
    const ses::Field1D coh = ses::gaussian_wavepacket(g, 4.0, sig, 0.0);
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        const double x = g.coord(i);
        v[static_cast<std::size_t>(i)] = 0.5 * omega * omega * x * x;
    }
    const auto cl = ses::ho1d_spectrum(coh, omega, 20.0);
    double mean = 0.0;
    double tot = 0.0;
    for (const auto& [e, w] : cl) {
        mean += e * w;
        tot += w;
    }
    EXPECT_GT(tot, 0.999);  // the band held the whole state
    EXPECT_NEAR(mean / tot, ses::mean_energy(coh, v), 0.01);
}

TEST(HoSpectrum, FockDarwinGroundIsOneLineAndBReconstructs) {
    const ses::Grid3D g{ses::Grid1D{-20.0, 20.0, 128},
                        ses::Grid1D{-20.0, 20.0, 128},
                        ses::Grid1D{0.0, 2.0, 1}};
    const double w0 = 0.5;
    const double b = 0.6;
    const double om = std::sqrt(w0 * w0 + 0.25 * b * b);
    // Relax the LATTICE ground at B (the qdot boot path).
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            v[static_cast<std::size_t>(g.flat(i, j, 0))] =
                0.5 * w0 * w0 * (x * x + y * y);
        }
    }
    ses::PeierlsLattice2D prop{g, v, 0.005};
    prop.set_uniform_field(b);
    ses::Field3D psi{g};
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            psi(i, j, 0) = std::exp(-(x * x + y * y) / 18.0);
        }
    }
    ses::normalize(psi);
    prop.relax(psi, 4000);
    const double e_meas = prop.energy(psi);
    ASSERT_NEAR(e_meas, om, 0.02 * om);  // the qdot contract

    const auto lines = ses::fock_darwin_spectrum(psi, w0, b, 4.0);
    ASSERT_GE(lines.size(), 3u);
    double tot = 0.0;
    double mean = 0.0;
    double wmax = 0.0;
    std::size_t imax = 0;
    for (std::size_t k = 0; k < lines.size(); ++k) {
        tot += lines[k].second;
        mean += lines[k].first * lines[k].second;
        if (lines[k].second > wmax) {
            wmax = lines[k].second;
            imax = k;
        }
    }
    EXPECT_GT(tot, 0.98);                       // basis holds the state
    EXPECT_GT(wmax, 0.95);                      // ...as ONE line
    EXPECT_NEAR(lines[imax].first, om, 0.03);   // ...at E = Omega
    EXPECT_NEAR(mean / tot, e_meas, 0.05);      // <H> reconstruction
}

}  // namespace
