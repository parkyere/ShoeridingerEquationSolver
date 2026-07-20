// RED: the IBM-corral physics the scene had simplified away (user order):
//  - Cu(111) surface-state effective mass m* = 0.38: spectral relax inside
//    the adatom fence lands in the leaky-wall band around j01^2/(2 m* R^2),
//    and E0 scales as 1/m (E0(m*) * m* matches E0(1) to a few %);
//  - infinite substrate: the outer absorbing mask on the collapsed-z 2D
//    grid makes leaked flux VANISH -- no periodic wrap-around revival --
//    while the conditional wavefunction is renormalized back to 1;
//  - black-dot adatoms: a fence-localized per-step damp factor absorbs a
//    crossing packet's pre-renormalization norm PARTIALLY (Crommie's ~50%
//    absorbing scatterers), neither unitary nor a black hole.
// The corral director copies these exact constructions (CONTRACT comments
// there point back here); grid-honest tolerances throughout.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
import ses.grid;
import ses.vec;
import ses.field;
import ses.potential;
import ses.propagator;
import ses.imaginary_time;
import ses.observables;
import ses.wavepacket;
import ses.parallel;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Solid annular wall with the corral geometry: V = amp on R <= r < R + w.
// The mass-scaling contract uses a STRONG non-leaky ring: under imaginary
// time the disc mode is only metastable, and the m* = 0.38 state tunnels
// out ~e^{sqrt(m)} FASTER than m = 1 (measured: its plateau all but
// vanishes at the scene's 1.5-high fence), so the wall here is high enough
// (amp 4, width 3) that both masses plateau hard. The live leaky-fence
// 512^2 relax is the selftest-corral arc's job.
std::vector<double> ring_wall(const ses::Grid3D& g, double radius, double amp,
                              double width) {
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double r = std::sqrt(x * x + y * y);
            if (r >= radius && r < radius + width) {
                v[static_cast<std::size_t>(g.flat(i, j, 0))] = amp;
            }
        }
    }
    return v;
}

// The disc mode is only METASTABLE under imaginary time: the barrier-tail
// seeds the (lower) outside box ground, whose weight grows as e^{2 dE tau},
// so a convergence gate races the drain and loses for small m*. The honest
// capture is a FIXED tau on the plateau: excited settling is done by
// tau ~ 16 (m = 1 gaps) while the crossover sits beyond tau ~ 45 (m*, this
// wall) -- tau = 20 reads the metastable ground for both masses.
double relax_ground_energy(const ses::Grid3D& g, const std::vector<double>& v,
                           double mass) {
    ses::Field3D psi = ses::gaussian_wavepacket(
        g, ses::Vec3d{0.5, 0.0, 0.0}, ses::Vec3d{2.0, 2.0, 0.5},
        ses::Vec3d{0.0, 0.0, 0.0});
    const ses::ImaginaryTimePropagator3D itp{g, v, 0.02, mass};
    itp.relax(psi, 1000);  // tau = 20
    return ses::mean_energy(psi, v, mass);
}

TEST(Corral2D, EffectiveMassScalesTheGroundToJZeroBand) {
    const double r = 6.0;
    const ses::Grid3D g{ses::Grid1D{-16.0, 16.0, 128},
                        ses::Grid1D{-16.0, 16.0, 128}, ses::Grid1D{0.0, 2.0, 1}};
    const std::vector<double> v = ring_wall(g, r, 4.0, 3.0);
    const double mstar = 0.38;
    const double e_star = relax_ground_energy(g, v, mstar);
    const double e_unit = relax_ground_energy(g, v, 1.0);
    // Leaky-fence band around the hard-disk J0 ground (same band the arc
    // uses): j01^2 / (2 m R^2).
    const double j01 = 2.405;
    const double hard_star = j01 * j01 / (2.0 * mstar * r * r);
    EXPECT_GT(e_star, 0.6 * hard_star);
    EXPECT_LT(e_star, 2.0 * hard_star);
    // E0 ~ 1/m: the fence interior is near-free, so E0(m*) m* tracks E0(1).
    EXPECT_NEAR(e_star * mstar / e_unit, 1.0, 0.15);
}

TEST(Corral2D, OpenBoundaryKillsThePeriodicRevival) {
    const ses::Grid3D g{ses::Grid1D{-16.0, 16.0, 128},
                        ses::Grid1D{-16.0, 16.0, 128}, ses::Grid1D{0.0, 2.0, 1}};
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const std::vector<double> mask = ses::absorbing_mask(g, 3.0);
    const double k0 = 3.0;
    const double dt = 0.05;
    const int nsteps = 240;  // t = 12 au: wall at ~5.3, wrap-return ~10.7
    auto p_center = [&](const ses::Field3D& f) {
        double inside = 0.0;
        double total = 0.0;
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                const double w = std::norm(f(i, j, 0));
                total += w;
                if (x * x + y * y < 5.0 * 5.0) {
                    inside += w;
                }
            }
        }
        return total > 0.0 ? inside / total : 0.0;
    };
    const auto packet = [&] {
        return ses::gaussian_wavepacket(g, ses::Vec3d{0.0, 0.0, 0.0},
                                        ses::Vec3d{1.5, 1.5, 0.5},
                                        ses::Vec3d{k0, 0.0, 0.0});
    };
    const ses::SplitOperator3D prop{g, v, dt};

    // Periodic (no mask): the packet wraps around and revives at the center.
    ses::Field3D per = packet();
    prop.step(per, nsteps);
    EXPECT_GT(p_center(per), 0.25);

    // Open boundary: per-step mask -- the leaked flux is GONE (pre-renorm
    // norm collapses; nothing returns), then the conditional state renorms.
    ses::Field3D open = packet();
    for (int s = 0; s < nsteps; ++s) {
        prop.step(open, 1);
        for (std::size_t i = 0; i < open.data().size(); ++i) {
            open.data()[i] *= mask[i];
        }
    }
    EXPECT_LT(ses::norm_sq(open), 0.05);  // absorbed, never re-entered
    ses::normalize(open);
    if (ses::norm_sq(open) > 0.0) {
        EXPECT_NEAR(ses::norm_sq(open), 1.0, 1e-9);  // conditional renorm
    }
}

TEST(Corral2D, BlackDotFenceAbsorbsPartially) {
    // 1D crossing of a single fence bump: the damp factor exp(-W(x) dt)
    // localized on the bump eats a PART of the packet (neither unitary nor
    // total absorption).
    const ses::Grid3D g{ses::Grid1D{-32.0, 32.0, 256}, ses::Grid1D{0.0, 2.0, 1},
                        ses::Grid1D{0.0, 2.0, 1}};
    const double amp = 1.5;
    const double sigma = 0.6;
    const double w0 = 0.8;  // black-dot strength (peak of W)
    const double dt = 0.05;
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    std::vector<double> damp(static_cast<std::size_t>(g.size()), 1.0);
    for (int i = 0; i < g.x.n; ++i) {
        const double x = g.x.coord(i);
        const double bump = amp * std::exp(-x * x / (2.0 * sigma * sigma));
        v[static_cast<std::size_t>(g.flat(i, 0, 0))] = bump;
        damp[static_cast<std::size_t>(g.flat(i, 0, 0))] =
            std::exp(-w0 * (bump / amp) * dt);
    }
    ses::Field3D psi = ses::gaussian_wavepacket(
        g, ses::Vec3d{-10.0, 0.0, 0.0}, ses::Vec3d{2.0, 0.5, 0.5},
        ses::Vec3d{2.5, 0.0, 0.0});  // E_k ~ 3.1 > barrier 1.5: mostly crosses
    const ses::SplitOperator3D prop{g, v, dt};
    for (int s = 0; s < 200; ++s) {  // t = 10: fully past the bump
        prop.step(psi, 1);
        for (std::size_t i = 0; i < psi.data().size(); ++i) {
            psi.data()[i] *= damp[i];
        }
    }
    const double survived = ses::norm_sq(psi);
    EXPECT_LT(survived, 0.90);  // it absorbed
    EXPECT_GT(survived, 0.05);  // but it is not a black hole
}

}  // namespace
