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
#include <utility>
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
    // gauge transformation G = diag(e^{i chi}): the link algebra
    // (u_dn = u_up e^{i(chi_i - chi_{i+1})}, y-links force chi to be
    // j-independent) gives chi = -phi on every site RIGHT of the solenoid
    // column. U_down = G U_up G^dag, and gauge equivalence transforms the
    // STATE too -- evolve G psi0 under the down cut and psi0 under the up
    // cut, and every density matches to round-off. (Evolving the SAME
    // psi0 under both differs at the packet-tail level: physically
    // distinct preparations.)
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
    int is = -1;
    for (int i = 0; i + 1 < g.x.n; ++i) {
        if (g.x.coord(i) <= xs && xs < g.x.coord(i + 1)) {
            is = i;
        }
    }
    ASSERT_GE(is, 0);
    const std::complex<double> gph{std::cos(phi), -std::sin(phi)};
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = is + 1; i < g.x.n; ++i) {
            b(i, j, 0) *= gph;
        }
    }
    up.step(a, 400);
    down.step(b, 400);
    for (int j = 0; j < g.y.n; j += 3) {
        for (int i = 0; i < g.x.n; i += 3) {
            EXPECT_NEAR(std::norm(a(i, j, 0)), std::norm(b(i, j, 0)), 1e-10)
                << "at " << i << "," << j;
        }
    }
}

// RED: UNIFORM field (Landau levels / cyclotron motion). In the Landau
// gauge the x-links of row j carry e^{-i B hx hy j}: EVERY plaquette then
// holds the same flux B hx hy -- a uniform B along z, checkable on the
// link table. Dynamics: a packet with mechanical momentum k0 rides a
// cyclotron orbit of radius k0/B at omega_c = B; and because the Landau
// spectrum is EQUALLY spaced (E_n = B(n + 1/2)), the state REVIVES at
// T = 2 pi / B -- the orbit closes and the wavefunction re-coheres.
TEST(PeierlsLattice2D, UniformFieldFillsEveryPlaquetteEqually) {
    const Grid3D g = plane_grid(8.0, 32, 8.0, 32);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    ses::PeierlsLattice2D prop{g, v, 0.02};
    const double b = 0.7;
    prop.set_uniform_field(b);
    const double want = b * g.x.spacing() * g.y.spacing();
    for (int j = 0; j + 1 < g.y.n; ++j) {
        for (int i = 0; i + 1 < g.x.n; ++i) {
            const std::complex<double> loop =
                prop.link_x(i, j) * prop.link_y(i + 1, j) *
                std::conj(prop.link_x(i, j + 1)) *
                std::conj(prop.link_y(i, j));
            EXPECT_NEAR(std::remainder(std::arg(loop) - want,
                                       2.0 * std::numbers::pi),
                        0.0, 1e-12)
                << "plaquette " << i << "," << j;
        }
    }
}

TEST(PeierlsLattice2D, CyclotronOrbitAndLandauRevival) {
    // B = 0.5, k0 = 1: radius k0/B = 2, period T = 2 pi / B ~ 12.57.
    // Link phase theta_x = -B hx y_j is (by the plane-wave band
    // E ~ (k + theta/h)^2/2) the Peierls form of A_x = +B y, so v rotates
    // COUNTERCLOCKWISE at omega_c = B and a tangential launch at (x0, 0)
    // with v = (0, k0) orbits the center (x0 - k0/B, 0). Launch at
    // (2, 0): the orbit circles the ORIGIN -- antipode (-2, 0) at T/2,
    // back at T. And not just in position: the equally spaced Landau
    // ladder re-coheres the whole state (revival overlap), up to small
    // lattice-band corrections.
    const Grid3D g = plane_grid(12.0, 96, 12.0, 96);
    const std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    const double b = 0.5;
    const double k0 = 1.0;
    const double dt = 0.01;
    ses::PeierlsLattice2D prop{g, v, dt};
    prop.set_uniform_field(b);
    // Packet with mechanical momentum +k0 in y: A_y = 0, and A_x = 0 on
    // the launch row y = 0, so the plain plane-wave factor IS mechanical.
    Field3D psi{g};
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double y = g.y.coord(j);
            const double env = std::exp(
                -((x - 2.0) * (x - 2.0) + y * y) / (4.0 * 1.4 * 1.4));
            psi(i, j, 0) = env * std::complex<double>{std::cos(k0 * y),
                                                      std::sin(k0 * y)};
        }
    }
    ses::normalize(psi);
    const Field3D start = psi;

    auto mean_r = [&](const Field3D& f) {
        double mx = 0.0;
        double my = 0.0;
        double den = 0.0;
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(f(i, j, 0));
                mx += g.x.coord(i) * w;
                my += g.y.coord(j) * w;
                den += w;
            }
        }
        return std::pair<double, double>{mx / den, my / den};
    };

    const double period = 2.0 * std::numbers::pi / b;
    const int half = static_cast<int>(0.5 * period / dt + 0.5);
    prop.step(psi, half);
    const auto [hx, hy] = mean_r(psi);
    EXPECT_NEAR(hx, -2.0, 0.4);
    EXPECT_NEAR(hy, 0.0, 0.4);
    prop.step(psi, half);
    const auto [fx, fy] = mean_r(psi);
    EXPECT_NEAR(fx, 2.0, 0.4);
    EXPECT_NEAR(fy, 0.0, 0.4);
    std::complex<double> ov{};
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            ov += std::conj(start(i, j, 0)) * psi(i, j, 0);
        }
    }
    ov *= g.x.spacing() * g.y.spacing() * g.z.spacing();
    EXPECT_GT(std::norm(ov), 0.8);  // the Landau ladder re-coheres
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

// RED: imaginary-time relaxation ON the lattice (same bond splitting with
// cosh/sinh mixing, renormalized) plus the energy readout <H> -- the
// ground-state preparers for the quantum-dot and quantum-corral scenes.
// The killer feature over the FFT imaginary-time propagator: the LINK
// PHASES ride along, so it can relax the ground state of a dot IN a
// magnetic field -- the Fock-Darwin ground, E = Omega = sqrt(w0^2 + B^2/4)
// (a.u.), which no B = 0 machinery can reach.
TEST(PeierlsLattice2D, RelaxFindsTheFockDarwinGround) {
    const Grid3D g = plane_grid(20.0, 128, 20.0, 128);
    const double w0 = 0.5;
    const double b = 0.6;
    std::vector<double> v(static_cast<std::size_t>(g.size()));
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            const double y = g.y.coord(j);
            v[static_cast<std::size_t>(g.flat(i, j, 0))] =
                0.5 * w0 * w0 * (x * x + y * y);
        }
    }
    ses::PeierlsLattice2D prop{g, v, 0.02};
    prop.set_uniform_field(b);
    Field3D psi = plane_packet(g, 1.0, -2.0, 3.0, 0.0);  // crooked seed
    prop.relax(psi, 3000);
    const double omega = std::sqrt(w0 * w0 + 0.25 * b * b);
    EXPECT_NEAR(prop.energy(psi), omega, 0.03 * omega);
    EXPECT_NEAR(ses::norm_sq(psi), 1.0, 1e-9);
}

TEST(PeierlsLattice2D, RelaxConfinesTheCorralGround) {
    // The 1993 IBM quantum corral: 48 atoms on a ring of radius 10 (each a
    // Gaussian bump), B = 0. The relaxed ground state lives INSIDE the
    // ring (a leaky circular box) with energy near the hard-wall J0 mode
    // j01^2 / (2 R^2) -- generously bracketed, the fence is soft and the
    // atoms are discrete.
    const Grid3D g = plane_grid(16.0, 160, 16.0, 160);
    const double ring_r = 10.0;
    const double pi = std::numbers::pi;
    std::vector<double> v(static_cast<std::size_t>(g.size()), 0.0);
    for (int a = 0; a < 48; ++a) {
        const double th = 2.0 * pi * a / 48.0;
        const double ax = ring_r * std::cos(th);
        const double ay = ring_r * std::sin(th);
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double dx = g.x.coord(i) - ax;
                const double dy = g.y.coord(j) - ay;
                v[static_cast<std::size_t>(g.flat(i, j, 0))] +=
                    1.5 * std::exp(-(dx * dx + dy * dy) / (2.0 * 0.6 * 0.6));
            }
        }
    }
    ses::PeierlsLattice2D prop{g, v, 0.02};
    Field3D psi = plane_packet(g, 0.5, -1.0, 4.0, 0.0);
    prop.relax(psi, 4000);
    double inside = 0.0;
    double total = 0.0;
    for (int j = 0; j < g.y.n; ++j) {
        for (int i = 0; i < g.x.n; ++i) {
            const double w = std::norm(psi(i, j, 0));
            total += w;
            if (std::hypot(g.x.coord(i), g.y.coord(j)) < ring_r - 1.0) {
                inside += w;
            }
        }
    }
    EXPECT_GT(inside / total, 0.85);
    const double e_hard = 2.405 * 2.405 / (2.0 * ring_r * ring_r);
    const double e = prop.energy(psi);
    EXPECT_GT(e, 0.6 * e_hard);
    EXPECT_LT(e, 2.0 * e_hard);
}

}  // namespace
