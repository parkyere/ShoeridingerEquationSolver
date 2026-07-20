// RED: Anderson localization contracts. The disorder landscape is
// deterministic per seed, bounded by the amplitude range, and every
// barrier sits BELOW the packet energy (the classical particle passes);
// yet the quantum packet's transport HALTS -- coherent backscattering --
// while the clean (W = 0) twin flies ballistically.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

import ses.scenario.anderson1d_director;
import ses.field;
import ses.grid;
import ses.observables;
import ses.propagator;
import ses.wavepacket;

namespace {

TEST(Anderson1D, LandscapeIsDeterministicBoundedAndSubEnergy) {
    const ses::Grid1D g{-40.0, 40.0, 2048};
    const std::vector<double> a = ses_shell::anderson_potential(g, 1.0, 7);
    const std::vector<double> b = ses_shell::anderson_potential(g, 1.0, 7);
    const std::vector<double> c = ses_shell::anderson_potential(g, 1.0, 8);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(a, b);  // same seed -> bitwise identical
    EXPECT_NE(a, c);  // fresh seed -> fresh landscape
    double vmax = 0.0;
    double vsum = 0.0;
    for (const double v : a) {
        vmax = std::max(vmax, std::abs(v));
        vsum += std::abs(v);
    }
    EXPECT_GT(vsum, 0.0);  // there IS a landscape
    // Overlap-bounded: |V| stays under ~1.2 W, and in particular UNDER
    // the packet energy k0^2/2 = 1.125 -- the classical-passes premise.
    EXPECT_LT(vmax, 1.2);
    EXPECT_LT(vmax, 0.5 * ses_shell::kAn1dK0 * ses_shell::kAn1dK0);
}

TEST(Anderson1D, DisorderHaltsTheBallisticPacket) {
    const ses::Grid1D g{-40.0, 40.0, 2048};
    const double x0 = -25.0;
    auto run = [&](const std::vector<double>& v, double t_probe,
                   double* sig) {
        const ses::SplitOperator1D prop{g, v, 0.01};
        ses::Field1D psi =
            ses::gaussian_wavepacket(g, x0, 2.0, ses_shell::kAn1dK0);
        const int n = static_cast<int>(t_probe / 0.01 + 0.5);
        for (int s = 0; s < n; ++s) {
            prop.step(psi, 1);
        }
        if (sig != nullptr) {
            *sig = ses::sigma_x(psi);
        }
        return ses::mean_position(psi) - x0;
    };
    const std::vector<double> clean(static_cast<std::size_t>(g.n), 0.0);
    const std::vector<double> dis =
        ses_shell::anderson_potential(g, ses_shell::kAn1dW, 7);
    const double clean_dx = run(clean, 25.0, nullptr);
    double dis_sig_t1 = 0.0;
    double dis_sig_t2 = 0.0;
    const double dis_dx_t1 = run(dis, 15.0, &dis_sig_t1);
    const double dis_dx = run(dis, 25.0, &dis_sig_t2);
    std::printf("anderson: clean dx %.1f, disordered dx %.1f -> %.1f, "
                "sigma %.1f -> %.1f\n",
                clean_dx, dis_dx_t1, dis_dx, dis_sig_t1, dis_sig_t2);
    EXPECT_GT(clean_dx, 30.0);            // ballistic: v_g t = 37.5
    EXPECT_LT(dis_dx, 0.5 * clean_dx);    // transport halted...
    // ...and the spread has SATURATED (localized envelope), not merely
    // slowed: the late-window growth is a small fraction of ballistic.
    EXPECT_LT(dis_sig_t2 - dis_sig_t1, 0.2 * (clean_dx * 10.0 / 25.0));
}

}  // namespace
