// RED: the radial engine (T7) -- lifetimes for ALL bound orbitals up to a
// requested principal quantum number. A 3D grid cannot hold Rydberg states
// (n = 10 reaches ~400 Bohr), but the potential is spherically symmetric,
// so the eigenproblem reduces EXACTLY to 1D per angular momentum l:
//     u'' = 2 (V(r) + l(l+1)/(2 r^2) - E) u,   u(0) = u(R) = 0,
// with psi = (u(r)/r) Y_lm. Finite differences give a symmetric tridiagonal
// Hamiltonian; eigenvalues come from Sturm-sequence bisection (the k-th
// state has k radial nodes and n = l + 1 + k), eigenvectors from inverse
// iteration. Level-averaged E1 decay rates follow from the radial dipole
// integrals with the standard angular factor:
//     A(n l -> n' l') = (4/3) alpha^3 w^3 * max(l, l') / (2 l + 1) * Rint^2.
//
// Oracles:
//  - particle in a box: the EXACT discrete FD spectrum (1 - cos(k pi h))/h^2
//    and the node count;
//  - isotropic harmonic trap: E = w (2k + l + 3/2);
//  - hydrogen (-1/r): the Rydberg series -1/(2 n^2), the analytic dipole
//    integral <u_10|r|u_21> = 128 sqrt(6)/243, and the MEASURED lifetimes
//    tau(2p) = 1.60 ns, tau(3s) = 158 ns, tau(3p) = 5.4 ns, tau(3d) = 15.6
//    ns; 1s and 2s are E1-stable;
//  - the soft-core atom (a = 1): energies must match the 3D solver's ITP
//    values and the level lifetime tau(2p) must match the T5 pipeline.

#include <core/decay.hpp>
#include <core/radial.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <vector>

namespace {

using ses::RadialGrid;

std::vector<double> zero_potential(const RadialGrid& g) {
    return std::vector<double>(static_cast<std::size_t>(g.n), 0.0);
}

std::vector<double> harmonic_potential_r(const RadialGrid& g) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        v[static_cast<std::size_t>(i)] = 0.5 * g.r(i) * g.r(i);
    }
    return v;
}

std::vector<double> coulomb_potential_r(const RadialGrid& g) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        v[static_cast<std::size_t>(i)] = -1.0 / g.r(i);
    }
    return v;
}

std::vector<double> soft_coulomb_potential_r(const RadialGrid& g, double a) {
    std::vector<double> v(static_cast<std::size_t>(g.n));
    for (int i = 0; i < g.n; ++i) {
        v[static_cast<std::size_t>(i)] = -1.0 / std::sqrt(g.r(i) * g.r(i) + a * a);
    }
    return v;
}

int count_interior_sign_changes(const std::vector<double>& u) {
    int changes = 0;
    double prev = 0.0;
    for (const double x : u) {
        if (x != 0.0) {
            if (prev != 0.0 && x * prev < 0.0) {
                ++changes;
            }
            prev = x;
        }
    }
    return changes;
}

constexpr double kAuPerSecond = 1.0 / 2.4188843265857e-17;  // au of time in 1 s

TEST(SturmCount, MatchesTheExactDiscreteBoxSpectrum) {
    const RadialGrid g{1.0, 499};
    const ses::RadialHamiltonian h = ses::radial_hamiltonian(g, zero_potential(g), 0);
    const double dh = g.h();
    // Exact FD eigenvalues of -(1/2) d2/dr2 with u(0)=u(R)=0:
    // E_k = (1 - cos(k pi h / R... k pi / (n+1))) / h^2, k = 1..n.
    auto exact = [&](int k) {
        return (1.0 - std::cos(k * 3.14159265358979323846 / (g.n + 1))) / (dh * dh);
    };
    EXPECT_EQ(ses::sturm_count_below(h, 0.5 * exact(1)), 0);
    EXPECT_EQ(ses::sturm_count_below(h, 0.5 * (exact(1) + exact(2))), 1);
    EXPECT_EQ(ses::sturm_count_below(h, 0.5 * (exact(3) + exact(4))), 3);
    EXPECT_EQ(ses::sturm_count_below(h, exact(g.n) + 1.0), g.n);
}

TEST(RadialEigenstate, BoxEigenvaluesExactAndNodesCounted) {
    const RadialGrid g{1.0, 499};
    const ses::RadialHamiltonian h = ses::radial_hamiltonian(g, zero_potential(g), 0);
    const double dh = g.h();
    for (int k = 0; k < 3; ++k) {
        const ses::RadialState s = ses::radial_eigenstate(g, h, k);
        const double exact =
            (1.0 - std::cos((k + 1) * 3.14159265358979323846 / (g.n + 1))) / (dh * dh);
        EXPECT_NEAR(s.energy, exact, 1e-9 * exact);
        EXPECT_EQ(count_interior_sign_changes(s.u), k);
        // Normalized: sum u^2 h = 1.
        double norm = 0.0;
        for (const double x : s.u) {
            norm += x * x;
        }
        EXPECT_NEAR(norm * dh, 1.0, 1e-10);
    }
}

TEST(RadialEigenstate, IsotropicHarmonicLadder) {
    const RadialGrid g{12.0, 2399};
    const std::vector<double> v = harmonic_potential_r(g);
    // E = w (2k + l + 3/2), w = 1.
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 0).energy,
                1.5, 1e-3);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 1), 0).energy,
                2.5, 1e-3);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 2), 0).energy,
                3.5, 1e-3);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 1).energy,
                3.5, 1e-3);
}

TEST(RadialEigenstate, HydrogenRydbergSeries) {
    const RadialGrid g{200.0, 9999};
    const std::vector<double> v = coulomb_potential_r(g);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 0).energy,
                -0.5, 1e-3);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 1).energy,
                -0.125, 5e-4);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 1), 0).energy,
                -0.125, 5e-4);
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 2), 0).energy,
                -1.0 / 18.0, 5e-4);
    // n = 5, l = 4 -- a genuinely Rydberg-ish oracle on the same grid.
    EXPECT_NEAR(ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 4), 0).energy,
                -0.02, 2e-4);
}

TEST(RadialDipole, HydrogenAnalyticIntegral) {
    const RadialGrid g{200.0, 9999};
    const std::vector<double> v = coulomb_potential_r(g);
    const ses::RadialState u10 =
        ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 0), 0);
    const ses::RadialState u21 =
        ses::radial_eigenstate(g, ses::radial_hamiltonian(g, v, 1), 0);
    const double rint = ses::radial_dipole_integral(g, u10.u, u21.u);
    // <u_10| r |u_21> = 128 sqrt(6) / 243 = 1.29027...
    EXPECT_NEAR(std::abs(rint), 128.0 * std::sqrt(6.0) / 243.0, 2e-3);
}

TEST(EinsteinALevel, AngularFactorsAreExact) {
    const double alpha3 = std::pow(ses::kFineStructureConstant, 3.0);
    // Upper l = 1 -> lower l = 0: factor max(1,0)/(2*1+1) = 1/3.
    EXPECT_DOUBLE_EQ(ses::einstein_a_level(1.0, 1, 0, 1.0), (4.0 / 9.0) * alpha3);
    // Upper l = 0 -> lower l = 1: factor max(0,1)/(2*0+1) = 1.
    EXPECT_DOUBLE_EQ(ses::einstein_a_level(1.0, 0, 1, 1.0), (4.0 / 3.0) * alpha3);
    // Upper l = 2 -> lower l = 1: factor 2/5.
    EXPECT_DOUBLE_EQ(ses::einstein_a_level(1.0, 2, 1, 1.0),
                     (4.0 / 3.0) * alpha3 * (2.0 / 5.0));
}

TEST(BoundLevelTable, HydrogenLifetimesMatchTheMeasuredValues) {
    const RadialGrid g{200.0, 9999};
    const std::vector<ses::LevelInfo> table =
        ses::bound_level_table(g, coulomb_potential_r(g), 3);
    ASSERT_EQ(table.size(), 6u);  // 1s 2s 2p 3s 3p 3d

    auto level = [&](int n, int l) -> const ses::LevelInfo& {
        for (const ses::LevelInfo& e : table) {
            if (e.n == n && e.l == l) {
                return e;
            }
        }
        static const ses::LevelInfo missing{};
        ADD_FAILURE() << "missing level n=" << n << " l=" << l;
        return missing;
    };

    EXPECT_EQ(level(1, 0).lifetime, 0.0);  // stable (0 = no open E1 channel)
    EXPECT_EQ(level(2, 0).lifetime, 0.0);  // 2s: E1-stable (two-photon in QED)
    // Measured hydrogen lifetimes, in au of time (1 s = 4.134e16 au):
    EXPECT_NEAR(level(2, 1).lifetime, 1.60e-9 * kAuPerSecond,
                0.03 * 1.60e-9 * kAuPerSecond);
    EXPECT_NEAR(level(3, 0).lifetime, 158.0e-9 * kAuPerSecond,
                0.05 * 158.0e-9 * kAuPerSecond);
    EXPECT_NEAR(level(3, 1).lifetime, 5.4e-9 * kAuPerSecond,
                0.05 * 5.4e-9 * kAuPerSecond);
    EXPECT_NEAR(level(3, 2).lifetime, 15.6e-9 * kAuPerSecond,
                0.05 * 15.6e-9 * kAuPerSecond);
}

TEST(BoundLevelTable, CountsAllFiftyFiveLevelsUpToNTen) {
    const RadialGrid g{600.0, 14999};
    const std::vector<ses::LevelInfo> table =
        ses::bound_level_table(g, coulomb_potential_r(g), 10);
    ASSERT_EQ(table.size(), 55u);
    // Every excited level except 2s decays; lifetimes grow with n.
    for (const ses::LevelInfo& e : table) {
        if (e.n == 1 || (e.n == 2 && e.l == 0)) {
            EXPECT_EQ(e.lifetime, 0.0);
        } else if (e.n > 1) {
            EXPECT_GT(e.lifetime, 0.0);
        }
    }
    auto lifetime = [&](int n, int l) {
        for (const ses::LevelInfo& e : table) {
            if (e.n == n && e.l == l) {
                return e.lifetime;
            }
        }
        return -1.0;
    };
    // Circular states (l = n-1) have the longest lifetimes in each n and
    // grow steeply with n: tau(10,9) >> tau(3,2).
    EXPECT_GT(lifetime(10, 9), 100.0 * lifetime(3, 2));
    // n = 10 energy sits on the Rydberg series.
    for (const ses::LevelInfo& e : table) {
        if (e.n == 10) {
            EXPECT_NEAR(e.energy, -0.005, 2e-4);
        }
    }
}

TEST(BoundLevelTable, SoftCoreAtomMatchesThe3DSolver) {
    const RadialGrid g{200.0, 9999};
    const std::vector<ses::LevelInfo> table =
        ses::bound_level_table(g, soft_coulomb_potential_r(g, 1.0), 2);
    auto level = [&](int n, int l) {
        for (const ses::LevelInfo& e : table) {
            if (e.n == n && e.l == l) {
                return e;
            }
        }
        return ses::LevelInfo{};
    };
    // The 3D ITP values from the 128^3 GPU solver (T5 report).
    EXPECT_NEAR(level(1, 0).energy, -0.2749, 2e-3);
    EXPECT_NEAR(level(2, 1).energy, -0.1129, 2e-3);
    EXPECT_NEAR(level(2, 0).energy, -0.0927, 2e-3);
    // The T5 3D pipeline found tau(2p_z) = 1.93e8 au; the radial route must
    // agree (independent discretizations, same physics).
    EXPECT_NEAR(level(2, 1).lifetime, 1.93e8, 0.05 * 1.93e8);
}

}  // namespace
