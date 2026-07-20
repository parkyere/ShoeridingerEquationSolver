// RED: the pinned-spin contracts. Larmor: +x under B_z precesses at
// EXACTLY omega = B (z frozen, norm round-off). Rabi: resonant circular
// drive flips z fully at Omega_R = b1. Measurement: Born collapse onto
// +-n with complementary probabilities and an EIGENSTATE afterward.
// Echo: a detuned ensemble fans out, the pi pulse time-reverses the fan,
// the mean transverse spin refocuses at 2 tau.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <random>
#include <vector>

import ses.spin;

namespace {

constexpr double kPi = 3.14159265358979323846;

ses::Spinor plus_x() {
    ses::Spinor s;
    const double r = 1.0 / std::sqrt(2.0);
    s.up = {r, 0.0};
    s.dn = {r, 0.0};
    return s;
}

TEST(Spin, LarmorPrecessesAtExactlyB) {
    ses::Spinor s = plus_x();
    const double b = 0.8;
    const double dt = 0.01;
    const int n = 500;  // t = 5: phase = 4 rad
    for (int k = 0; k < n; ++k) {
        ses::spin_step(s, 0.0, 0.0, b, dt);
    }
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    ses::bloch_vector(s, &x, &y, &z);
    // <sigma_z> frozen, the transverse pair rotates by +b t CCW about
    // +B (the U = e^{-i theta sigma/2} convention, pinned HERE).
    EXPECT_NEAR(z, 0.0, 1e-12);
    EXPECT_NEAR(std::hypot(x, y), 1.0, 1e-12);
    const double phase = std::atan2(y, x);
    double expect = b * n * dt;
    while (expect > kPi) {
        expect -= 2.0 * kPi;
    }
    EXPECT_NEAR(phase, expect, 1e-9);
    EXPECT_NEAR(std::norm(s.up) + std::norm(s.dn), 1.0, 1e-12);
}

TEST(Spin, ResonantCircularDriveRabiFlips) {
    ses::Spinor s;  // |up> = +z
    const double b0 = 1.0;
    const double b1 = 0.05;
    const double dt = 0.001;
    const int n = static_cast<int>(kPi / b1 / dt + 0.5);  // half Rabi cycle
    for (int k = 0; k < n; ++k) {
        const double t = k * dt;
        // Lab-frame circular drive co-rotating with the +B CCW sense.
        const double bx = b1 * std::cos(b0 * t);
        const double by = b1 * std::sin(b0 * t);
        ses::spin_step(s, bx, by, b0, dt);
    }
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    ses::bloch_vector(s, &x, &y, &z);
    EXPECT_NEAR(z, -1.0, 5e-3);  // full flip at t = pi / Omega_R
}

TEST(Spin, MeasurementCollapsesOntoTheAxis) {
    const double r = 1.0 / std::sqrt(3.0);
    ses::Spinor s = plus_x();
    // Along +x the state IS |+x>: certainty.
    EXPECT_EQ(ses::spin_measure(s, 1.0, 0.0, 0.0, 0.5), +1);
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    ses::bloch_vector(s, &x, &y, &z);
    EXPECT_NEAR(x, 1.0, 1e-12);
    // Along a tilted axis both outcomes fire per Born; the collapsed
    // state is the EIGENSTATE (<sigma . n> = +-1).
    ses::Spinor a = plus_x();
    const int oa = ses::spin_measure(a, r, r, r, 0.0);   // u=0 -> plus
    EXPECT_EQ(oa, +1);
    ses::bloch_vector(a, &x, &y, &z);
    EXPECT_NEAR(x * r + y * r + z * r, 1.0, 1e-12);
    ses::Spinor b = plus_x();
    const int ob = ses::spin_measure(b, r, r, r, 0.999);  // u~1 -> minus
    EXPECT_EQ(ob, -1);
    ses::bloch_vector(b, &x, &y, &z);
    EXPECT_NEAR(x * r + y * r + z * r, -1.0, 1e-12);
}

TEST(Spin, PiPulseRefocusesTheDetunedEnsemble) {
    const int kSpins = 64;
    const double tau = 8.0;
    const double dt = 0.005;
    std::mt19937 rng(7);
    std::uniform_real_distribution<double> det(-0.4, 0.4);
    std::vector<ses::Spinor> ens(kSpins);
    std::vector<double> dw(kSpins);
    for (int i = 0; i < kSpins; ++i) {
        ens[static_cast<std::size_t>(i)] = plus_x();  // after a pi/2
        dw[static_cast<std::size_t>(i)] = det(rng);
    }
    auto mxy = [&] {
        double sx = 0.0;
        double sy = 0.0;
        for (const ses::Spinor& s : ens) {
            double x = 0.0;
            double y = 0.0;
            double z = 0.0;
            ses::bloch_vector(s, &x, &y, &z);
            sx += x;
            sy += y;
        }
        return std::hypot(sx, sy) / kSpins;
    };
    const int n_tau = static_cast<int>(tau / dt + 0.5);
    for (int k = 0; k < n_tau; ++k) {
        for (int i = 0; i < kSpins; ++i) {
            ses::spin_step(ens[static_cast<std::size_t>(i)], 0.0, 0.0,
                           dw[static_cast<std::size_t>(i)], dt);
        }
    }
    const double fanned = mxy();
    EXPECT_LT(fanned, 0.4);  // the fan opened (dephased)
    for (ses::Spinor& s : ens) {
        ses::spin_rotate(s, 1.0, 0.0, 0.0, kPi);  // the echo pi pulse
    }
    for (int k = 0; k < n_tau; ++k) {
        for (int i = 0; i < kSpins; ++i) {
            ses::spin_step(ens[static_cast<std::size_t>(i)], 0.0, 0.0,
                           dw[static_cast<std::size_t>(i)], dt);
        }
    }
    EXPECT_GT(mxy(), 0.999);  // the echo: perfect refocus at 2 tau
}

}  // namespace
