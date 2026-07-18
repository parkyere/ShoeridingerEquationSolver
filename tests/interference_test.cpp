// RED: the 1D interference scenes' core physics.
//
// DOUBLE SLIT (transverse frame). The 1D axis is the coordinate PARALLEL
// to the slit plane; propagation toward the screen is time (t = z/v, the
// exact paraxial/Fresnel reduction of the real 2D experiment -- not an
// approximation). The infinite wall with two slits acts at t = 0 as the
// aperture function: psi is windowed to the two openings (wall absorbs the
// rest; the state is renormalized = post-selection on transmission). From
// that instant the momentum distribution |psi~(k)|^2 IS the screen: free
// flight preserves it exactly, and the far-field arrival density at angle
// k is cos^2((k d + phi)/2) fringes under the single-slit sinc^2(k w / 2)
// envelope. A solenoid tucked behind the wall between the slits multiplies
// slit 2 by e^{i phi} (the exact reduced Aharonov-Bohm effect): fringes
// SHIFT by phi/d, the envelope does not move (Chambers).
//
// AB RING. The periodic FFT grid IS a ring. With flux Phi through it the
// Hamiltonian is (p - A)^2 / 2, A = Phi / L, B = 0 everywhere ON the ring:
// two wavepacket halves sent around opposite sides pick up a relative
// phase of exactly Phi at recombination -- constructive at Phi = 0,
// destructive at Phi = pi, period one flux quantum (2 pi in a.u.).

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

import ses.field;
import ses.grid;
import ses.interference;
import ses.propagator;
import ses.wavepacket;

namespace {

const ses::Grid1D kGrid{-100.0, 100.0, 8192};
constexpr double kSep = 8.0;    // slit separation d (center to center)
constexpr double kWidth = 1.5;  // slit width w

// Broad incident wave (sigma >> d): a flat wavefront across both slits.
ses::Field1D incident() {
    return ses::gaussian_wavepacket(kGrid, 0.0, 30.0, 0.0);
}

// Index of the |psi~|^2 peak nearest to wavenumber k_target.
int peak_near(const std::vector<double>& spec, const std::vector<double>& axis,
              double k_target, double window) {
    int best = -1;
    for (std::size_t j = 0; j < spec.size(); ++j) {
        if (std::abs(axis[j] - k_target) > window) {
            continue;
        }
        if (best < 0 || spec[j] > spec[static_cast<std::size_t>(best)]) {
            best = static_cast<int>(j);
        }
    }
    return best;
}

TEST(DoubleSlitAperture, WindowsTheWavefrontAndPostSelects) {
    const std::vector<std::complex<double>> ap =
        ses::double_slit_aperture(kGrid, kSep, kWidth, 0.0);
    ASSERT_EQ(static_cast<int>(ap.size()), kGrid.n);
    int open = 0;
    for (int i = 0; i < kGrid.n; ++i) {
        const double x = kGrid.coord(i);
        const bool slit1 = std::abs(x + 0.5 * kSep) <= 0.5 * kWidth;
        const bool slit2 = std::abs(x - 0.5 * kSep) <= 0.5 * kWidth;
        const double mag = std::abs(ap[static_cast<std::size_t>(i)]);
        if (slit1 || slit2) {
            EXPECT_NEAR(mag, 1.0, 1e-12) << "open at x = " << x;
            ++open;
        } else {
            EXPECT_EQ(mag, 0.0) << "wall at x = " << x;
        }
    }
    EXPECT_GT(open, 0);

    // The solenoid phase rides on slit 2 only.
    const std::vector<std::complex<double>> ab =
        ses::double_slit_aperture(kGrid, kSep, kWidth, 1.2);
    for (int i = 0; i < kGrid.n; ++i) {
        const double x = kGrid.coord(i);
        if (std::abs(x - 0.5 * kSep) <= 0.5 * kWidth) {
            EXPECT_NEAR(std::arg(ab[static_cast<std::size_t>(i)]), 1.2, 1e-12);
        } else if (std::abs(x + 0.5 * kSep) <= 0.5 * kWidth) {
            EXPECT_NEAR(std::arg(ab[static_cast<std::size_t>(i)]), 0.0, 1e-12);
        }
    }

    // apply_aperture: psi -> normalized windowed psi; returns the
    // transmitted fraction (the norm the wall did NOT absorb).
    ses::Field1D psi = incident();
    double direct = 0.0;
    for (int i = 0; i < kGrid.n; ++i) {
        direct += std::norm(ap[static_cast<std::size_t>(i)] * psi[i]);
    }
    direct *= kGrid.spacing();
    const double frac = ses::apply_aperture(psi, ap);
    EXPECT_NEAR(frac, direct, 1e-12);
    EXPECT_GT(frac, 0.01);
    EXPECT_LT(frac, 0.2);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-10);
}

TEST(DoubleSlit, YoungFringesUnderTheSingleSlitEnvelope) {
    ses::Field1D psi = incident();
    ses::apply_aperture(psi,
                        ses::double_slit_aperture(kGrid, kSep, kWidth, 0.0));
    const std::vector<double> spec = ses::momentum_spectrum(psi);
    const std::vector<double> axis = ses::momentum_axis(kGrid);
    ASSERT_EQ(spec.size(), axis.size());
    // Axis ascending, spectrum aligned to it.
    for (std::size_t j = 1; j < axis.size(); ++j) {
        EXPECT_GT(axis[j], axis[j - 1]);
    }
    const double dk = 2.0 * std::numbers::pi / kSep;  // fringe spacing
    // Central maximum at k = 0; neighbors at +-2 pi / d.
    const int c = peak_near(spec, axis, 0.0, 0.3 * dk);
    const int p = peak_near(spec, axis, dk, 0.3 * dk);
    const int m = peak_near(spec, axis, -dk, 0.3 * dk);
    ASSERT_GE(c, 0);
    ASSERT_GE(p, 0);
    ASSERT_GE(m, 0);
    const double bin = axis[1] - axis[0];
    EXPECT_NEAR(axis[static_cast<std::size_t>(c)], 0.0, 2.0 * bin);
    EXPECT_NEAR(axis[static_cast<std::size_t>(p)], dk, 0.05 * dk + 2.0 * bin);
    EXPECT_NEAR(axis[static_cast<std::size_t>(m)], -dk, 0.05 * dk + 2.0 * bin);
    // Dark fringe between them: destructive to a few percent.
    const int dark = peak_near(spec, axis, 0.5 * dk, 0.1 * dk);
    EXPECT_LT(spec[static_cast<std::size_t>(dark)],
              0.05 * spec[static_cast<std::size_t>(c)]);
    // Single-slit envelope zero at k = 2 pi / w kills the fringe there.
    const double kz = 2.0 * std::numbers::pi / kWidth;
    const int env = peak_near(spec, axis, kz, 0.5 * dk);
    EXPECT_LT(spec[static_cast<std::size_t>(env)],
              0.02 * spec[static_cast<std::size_t>(c)]);
}

TEST(DoubleSlit, TheScreenIsFrozenByFreeFlight) {
    // |psi~(k)|^2 -- the far-field screen -- must not change under free
    // evolution: the pattern is decided the instant the wall transmits.
    ses::Field1D psi = incident();
    ses::apply_aperture(psi,
                        ses::double_slit_aperture(kGrid, kSep, kWidth, 0.0));
    const std::vector<double> before = ses::momentum_spectrum(psi);
    const std::vector<double> zero(static_cast<std::size_t>(kGrid.n), 0.0);
    const ses::SplitOperator1D prop{kGrid, zero, 0.05};
    prop.step(psi, 200);
    const std::vector<double> after = ses::momentum_spectrum(psi);
    double num = 0.0;
    double den = 0.0;
    for (std::size_t j = 0; j < before.size(); ++j) {
        num += std::abs(after[j] - before[j]);
        den += before[j];
    }
    EXPECT_LT(num / den, 1e-9);
}

TEST(DoubleSlit, SolenoidPhaseSlidesFringesButNotTheEnvelope) {
    ses::Field1D ref = incident();
    ses::apply_aperture(ref,
                        ses::double_slit_aperture(kGrid, kSep, kWidth, 0.0));
    const std::vector<double> s0 = ses::momentum_spectrum(ref);

    ses::Field1D shifted = incident();
    ses::apply_aperture(
        shifted, ses::double_slit_aperture(kGrid, kSep, kWidth,
                                           std::numbers::pi));
    const std::vector<double> spi = ses::momentum_spectrum(shifted);

    const std::vector<double> axis = ses::momentum_axis(kGrid);
    const double dk = 2.0 * std::numbers::pi / kSep;
    // Phi = pi: the center becomes DARK, the maxima move to +-pi/d.
    const int c0 = peak_near(s0, axis, 0.0, 0.3 * dk);
    const int cpi = peak_near(spi, axis, 0.0, 0.1 * dk);
    EXPECT_LT(spi[static_cast<std::size_t>(cpi)],
              0.05 * s0[static_cast<std::size_t>(c0)]);
    const int half = peak_near(spi, axis, 0.5 * dk, 0.3 * dk);
    const double bin = axis[1] - axis[0];
    EXPECT_NEAR(axis[static_cast<std::size_t>(half)], 0.5 * dk,
                0.05 * dk + 2.0 * bin);
    // The envelope (total transmitted weight) does not care about phi.
    double t0 = 0.0;
    double tpi = 0.0;
    for (std::size_t j = 0; j < s0.size(); ++j) {
        t0 += s0[j];
        tpi += spi[j];
    }
    EXPECT_NEAR(tpi / t0, 1.0, 1e-9);
    // One flux quantum restores the pattern exactly.
    ses::Field1D again = incident();
    ses::apply_aperture(
        again, ses::double_slit_aperture(kGrid, kSep, kWidth,
                                         2.0 * std::numbers::pi));
    const std::vector<double> s2pi = ses::momentum_spectrum(again);
    for (std::size_t j = 0; j < s0.size(); j += 64) {
        EXPECT_NEAR(s2pi[j], s0[j], 1e-9 + 1e-6 * s0[j]);
    }
}

// ---- AB ring --------------------------------------------------------------

const ses::Grid1D kRing{-50.0, 50.0, 2048};

TEST(AbRing, PlaneWaveEigenphaseWithVectorPotential) {
    // On the ring, e^{i k_n x} (k_n an FFT bin) is an exact eigenstate of
    // (p - A)^2 / 2 with energy (k_n - A)^2 / 2; the split-operator with a
    // vector potential must reproduce that phase to round-off (V = 0, so
    // the propagator is EXACT -- no Trotter error).
    const double l = kRing.xmax - kRing.xmin;
    const double kn = 2.0 * std::numbers::pi / l * 7.0;  // bin 7
    const double a = 0.31;                               // flux 0.31 * L
    ses::Field1D psi{kRing};
    for (int i = 0; i < kRing.n; ++i) {
        const double x = kRing.coord(i);
        psi[i] = std::complex<double>{std::cos(kn * x), std::sin(kn * x)};
    }
    ses::normalize(psi);
    const ses::Field1D start = psi;
    const std::vector<double> zero(static_cast<std::size_t>(kRing.n), 0.0);
    const double dt = 0.05;
    const int steps = 100;
    const ses::SplitOperator1D prop{kRing, zero, dt, a};
    prop.step(psi, steps);
    std::complex<double> ov{};
    for (int i = 0; i < kRing.n; ++i) {
        ov += std::conj(start[i]) * psi[i];
    }
    ov *= kRing.spacing();
    const double want = -0.5 * (kn - a) * (kn - a) * dt * steps;
    EXPECT_NEAR(std::abs(ov), 1.0, 1e-10);
    const double got = std::arg(ov);
    const double diff = std::remainder(got - want, 2.0 * std::numbers::pi);
    EXPECT_NEAR(diff, 0.0, 1e-8);
}

TEST(AbRing, OneFluxQuantumIsALargeGaugeTransformation) {
    // Flux is physical only mod one quantum: A -> A + 2 pi / L is undone
    // by the LARGE gauge transformation e^{i (2 pi / L) x}, which on the
    // grid is an exact one-bin cyclic shift in k-space. So evolving under
    // A + q must equal gauge-in, evolve under A, gauge-out -- exactly, for
    // any band-interior state. Topology, not dynamics.
    const double l = kRing.xmax - kRing.xmin;
    const double a = 0.17;
    const double q = 2.0 * std::numbers::pi / l;
    const std::vector<double> zero(static_cast<std::size_t>(kRing.n), 0.0);
    const double dt = 0.03;
    const int steps = 60;
    ses::Field1D psi0 = ses::gaussian_wavepacket(kRing, -10.0, 5.0, 3.0);

    ses::Field1D direct = psi0;
    const ses::SplitOperator1D pq{kRing, zero, dt, a + q};
    pq.step(direct, steps);

    ses::Field1D gauged{kRing};
    for (int i = 0; i < kRing.n; ++i) {
        const double x = kRing.coord(i);
        gauged[i] = std::complex<double>{std::cos(q * x), std::sin(q * x)} *
                    psi0[i];
    }
    const ses::SplitOperator1D pa{kRing, zero, dt, a};
    pa.step(gauged, steps);
    for (int i = 0; i < kRing.n; ++i) {
        const double x = kRing.coord(i);
        gauged[i] *= std::complex<double>{std::cos(q * x), -std::sin(q * x)};
    }

    std::complex<double> ov{};
    for (int i = 0; i < kRing.n; ++i) {
        ov += std::conj(direct[i]) * gauged[i];
    }
    ov *= kRing.spacing();
    EXPECT_NEAR(std::norm(ov), 1.0, 1e-10);
}

TEST(AbRing, RecombinationInterferenceCarriesTheFlux) {
    // Two halves with MECHANICAL momentum +-k0 (canonical k = +-k0 + A:
    // the injection matches kinetic energy, as in the real interferometer)
    // launched from x0 run around opposite sides and meet at the antipode
    // x0 + L/2. Their relative phase there is EXACTLY the enclosed flux
    // Phi = A L: constructive at Phi = 0, destructive at Phi = pi.
    const double l = kRing.xmax - kRing.xmin;
    const double k0 = 16.0 * 2.0 * std::numbers::pi / l;  // bin 16
    const double x0 = -25.0;
    const double xm = x0 + 0.5 * l;  // meeting point
    const double t_meet = 0.5 * l / k0;
    const double dt = 0.01;
    const int steps = static_cast<int>(t_meet / dt + 0.5);
    const std::vector<double> zero(static_cast<std::size_t>(kRing.n), 0.0);

    auto density_at_meeting = [&](double phi) {
        const double a = phi / l;
        ses::Field1D psi{kRing};
        for (int i = 0; i < kRing.n; ++i) {
            const double x = kRing.coord(i);
            const double env = std::exp(-(x - x0) * (x - x0) / (2.0 * 16.0));
            const std::complex<double> boost{std::cos(a * x),
                                             std::sin(a * x)};
            psi[i] = env * boost * 2.0 *
                     std::cos(k0 * (x - x0));  // e^{+ik0} + e^{-ik0}
        }
        ses::normalize(psi);
        const ses::SplitOperator1D prop{kRing, zero, dt, a};
        prop.step(psi, steps);
        // Average density over the CREST width only (|u| < 0.15 of the
        // pi/k0 fringe period): a wider window would always contain some
        // crest of the shifted pattern and wash the contrast out.
        double sum = 0.0;
        int cnt = 0;
        for (int i = 0; i < kRing.n; ++i) {
            if (std::abs(kRing.coord(i) - xm) <
                0.15 * std::numbers::pi / k0) {
                sum += std::norm(psi[i]);
                ++cnt;
            }
        }
        return cnt > 0 ? sum / cnt : 0.0;
    };

    const double bright = density_at_meeting(0.0);
    const double dark = density_at_meeting(std::numbers::pi);
    const double again = density_at_meeting(2.0 * std::numbers::pi);
    EXPECT_GT(bright, 0.0);
    // Phi = pi: the fringe at the meeting point flips to a node -- the
    // pattern shifted by half a fringe, so the peak WITHIN the sampling
    // window drops by the fringe contrast.
    EXPECT_LT(dark, 0.35 * bright);
    // One flux quantum restores the bright meeting.
    EXPECT_NEAR(again, bright, 0.05 * bright);
}

}  // namespace
