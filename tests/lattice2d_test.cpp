// RED: the 2D Peierls lattice propagator -- the honest engine for the REAL
// double-slit + Aharonov-Bohm experiment (electron flying +x, a pierced
// wall, a solenoid hidden inside the wall between the slits, a screen).
//
// WHY a lattice propagator: the FFT split-operator cannot Trotterize
// (p - A)^2/2 for a LOCALIZED flux (A depends on both coordinates in every
// gauge, so no factor is diagonal in any FFT basis), and approximate A.p
// splittings drift the norm. On the lattice the flux enters EXACTLY as
// Peierls phases on the hopping links: kinetic = 4 bond groups (x-even,
// x-odd, y-even, y-odd), each a direct sum of independent 2-site bonds
// exponentiated EXACTLY as 2x2 rotations -- unitary to round-off by
// construction, Trotter error only in the ordering.
//
// The solenoid's string gauge: the cut runs from the solenoid center
// (buried mid-wall) STRAIGHT UP (+y) to the boundary; only x-links
// crossing the cut carry e^{+-i Phi}. Everywhere the field is zero the
// plaquette phase-product is 1; the single plaquette at the solenoid
// carries Phi -- topology on the link table itself, testable without
// evolving anything. The physical beam picks the phase up ONLY through
// the upper slit opening: exactly the AB double-slit.
//
// Honesty note: the lattice dispersion is E(k) = (1 - cos kh)/h^2 per
// axis, group velocity sin(k h)/h -- the contracts below test against the
// DISCRETE formulas, not the continuum ones.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

import ses.field;
import ses.grid;
import ses.lattice2d;

namespace {

using ses::Field3D;
using ses::Grid1D;
using ses::Grid3D;

// A single-plane (nz = 1) 3D grid: the 2D stage.
Grid3D plane_grid(double lx, int nx, double ly, int ny) {
    return Grid3D{Grid1D{-lx, lx, nx}, Grid1D{-ly, ly, ny},
                  Grid1D{-1.0, 1.0, 1}};
}

// Gaussian packet on the plane, momentum k0 along +x.
Field3D plane_packet(const Grid3D& g, double x0, double y0, double sigma,
                     double k0) {
    Field3D psi{g};
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double y = g.y.coord(j);
            const double env = std::exp(
                -((x - x0) * (x - x0) + (y - y0) * (y - y0)) /
                (4.0 * sigma * sigma));
            psi(i, j, 0) = env * std::complex<double>{std::cos(k0 * x),
                                                      std::sin(k0 * x)};
        }
    }
    ses::normalize(psi);
    return psi;
}

double mean_x(const Field3D& psi) {
    const Grid3D& g = psi.grid();
    double num = 0.0;
    double den = 0.0;
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double w = std::norm(psi(i, j, 0));
            num += g.x.coord(i) * w;
            den += w;
        }
    }
    return num / den;
}

TEST(PeierlsLattice2D, IsUnitaryToRoundOff) {
    const Grid3D g = plane_grid(10.0, 64, 10.0, 64);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    ses::PeierlsLattice2D prop{g, v, 0.02};
    Field3D psi = plane_packet(g, -3.0, 1.0, 2.0, 1.5);
    prop.step(psi, 500);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-12);
}

TEST(PeierlsLattice2D, FreePacketMovesAtTheLatticeGroupVelocity) {
    // v_g = sin(k0 h)/h, the DISCRETE dispersion -- not k0.
    const Grid3D g = plane_grid(30.0, 256, 10.0, 32);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double dt = 0.01;
    ses::PeierlsLattice2D prop{g, v, dt};
    const double k0 = 1.2;
    Field3D psi = plane_packet(g, -15.0, 0.0, 3.0, k0);
    const double x_start = mean_x(psi);
    const int steps = 1500;
    prop.step(psi, steps);
    const double h = g.x.spacing();
    const double vg = std::sin(k0 * h) / h;
    EXPECT_NEAR(mean_x(psi) - x_start, vg * dt * steps,
                0.01 * vg * dt * steps);
}

TEST(PeierlsLattice2D, SolenoidLinksCarryPureTopology) {
    // Phase-product around every elementary plaquette: 1 everywhere the
    // field vanishes, e^{i Phi} at the solenoid's plaquette ONLY. This is
    // the whole Aharonov-Bohm setup verified on the link table, before
    // any dynamics.
    const Grid3D g = plane_grid(8.0, 32, 8.0, 32);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    ses::PeierlsLattice2D prop{g, v, 0.02};
    const double phi = 1.9;
    const double xs = g.x.coord(15) + 0.5 * g.x.spacing();
    const double ys = g.y.coord(12) + 0.5 * g.y.spacing();
    prop.set_solenoid(phi, xs, ys);
    int hot = 0;
    for (int j = 0; j + 1 < g.y.n; ++j) {
        for (int i = 0; i + 1 < g.x.n; ++i) {
            // Counter-clockwise around plaquette (i, j):
            const std::complex<double> loop =
                prop.link_x(i, j) * prop.link_y(i + 1, j) *
                std::conj(prop.link_x(i, j + 1)) *
                std::conj(prop.link_y(i, j));
            const double flux = std::arg(loop);
            const bool at_solenoid = i == 15 && j == 12;
            if (at_solenoid) {
                EXPECT_NEAR(std::remainder(flux - phi,
                                           2.0 * std::numbers::pi),
                            0.0, 1e-12);
                ++hot;
            } else {
                EXPECT_NEAR(flux, 0.0, 1e-12)
                    << "stray field at plaquette " << i << "," << j;
            }
        }
    }
    EXPECT_EQ(hot, 1);
}

TEST(PeierlsLattice2D, GaugeCutDirectionIsInvisible) {
    // The SAME flux with the string cut running down instead of up is a
    // gauge transformation: every density must match to round-off.
    const Grid3D g = plane_grid(10.0, 64, 10.0, 64);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double phi = 2.4;
    const double xs = 0.31;
    const double ys = -0.17;
    ses::PeierlsLattice2D up{g, v, 0.02};
    up.set_solenoid(phi, xs, ys, true);
    ses::PeierlsLattice2D down{g, v, 0.02};
    down.set_solenoid(phi, xs, ys, false);
    Field3D a = plane_packet(g, -3.0, 0.5, 2.0, 1.0);
    Field3D b = a;
    up.step(a, 400);
    down.step(b, 400);
    for (int j = 0; j < g.y.n; j += 3) {
        for (int i = 0; i < g.x.n; i += 3) {
            EXPECT_NEAR(std::norm(a(i, j, 0)), std::norm(b(i, j, 0)), 1e-10)
                << "at " << i << "," << j;
        }
    }
}

TEST(PeierlsLattice2D, DoubleSlitCarriesTheSolenoidFlux) {
    // The REAL experiment in miniature: packet flying +x, a high wall with
    // two slits, the solenoid buried mid-wall between them. Phi = 0 gives
    // a bright central fringe on the axis downstream; Phi = pi kills it;
    // Phi = 2 pi restores it exactly (one flux quantum).
    const Grid3D g = plane_grid(24.0, 192, 16.0, 128);
    const double h = g.x.spacing();
    const double wall_lo = 0.0;
    const double wall_hi = 1.5;
    const double sep = 4.0;    // slit centers at y = +-2
    const double width = 1.2;
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        const bool open = std::abs(y - 0.5 * sep) <= 0.5 * width ||
                          std::abs(y + 0.5 * sep) <= 0.5 * width;
        if (open) {
            continue;
        }
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            if (x >= wall_lo && x <= wall_hi) {
                v[static_cast<std::size_t>(g.flat(i, j, 0))] = 40.0;
            }
        }
    }

    auto axis_density = [&](double phi) {
        ses::PeierlsLattice2D prop{g, v, 0.01};
        prop.set_solenoid(phi, 0.5 * (wall_lo + wall_hi) + 0.25 * h, 0.0);
        Field3D psi = plane_packet(g, -12.0, 0.0, 4.0, 2.0);
        prop.step(psi, 1200);  // t = 12: transit + downstream flight
        // Average density in a small axis window well past the wall.
        double sum = 0.0;
        int cnt = 0;
        for (int j = 0; j < g.y.n; ++j) {
            const double y = g.y.coord(j);
            if (std::abs(y) > 0.6) {
                continue;
            }
            for (int i = 0; i < g.x.n; ++i) {
                const double x = g.x.coord(i);
                if (x > 6.0 && x < 12.0) {
                    sum += std::norm(psi(i, j, 0));
                    ++cnt;
                }
            }
        }
        return sum / cnt;
    };

    const double bright = axis_density(0.0);
    const double dark = axis_density(std::numbers::pi);
    const double again = axis_density(2.0 * std::numbers::pi);
    EXPECT_GT(bright, 0.0);
    EXPECT_LT(dark, 0.35 * bright);
    EXPECT_NEAR(again, bright, 1e-6 * bright);
}

}  // namespace
