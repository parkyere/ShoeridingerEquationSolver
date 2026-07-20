module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>
export module ses.spinexact;
export import ses.spin;
export import ses.spinlattice;
import ses.parallel;


// EXACT 4x4 Heisenberg lattice: the FULL 2^16 = 65536-amplitude
// wavefunction (basis bit i = 1 means site i spin-DOWN), evolved by a
// Strang split of H = (1/2) sum_i B.sigma_i - J sum_bonds sigma_i.sigma_j:
// half single-site field rotations, then the 24 open-boundary bond gates
// exp(+i J dt sigma.sigma) EXACTLY via sigma.sigma = 2 SWAP - 1, then the
// other half. Entanglement is REAL here: the per-site Bloch arrows
// SHRINK (|<sigma_i>| < 1) as the reduced states mix -- the visible
// difference against the mean-field product ansatz.
// CONTRACT: tests/spinexact_test.cpp.


export namespace ses {

inline constexpr int kExactSide = 4;
inline constexpr int kExactSites = 16;
inline constexpr std::size_t kExactDim = 1u << kExactSites;

struct SpinState16 {
    std::vector<std::complex<double>> c;
};

// The 24 open-boundary bonds of the 4x4 lattice.
inline int exact_bonds(int (*out)[2]) {
    int n = 0;
    for (int y = 0; y < kExactSide; ++y) {
        for (int x = 0; x < kExactSide; ++x) {
            const int i = y * kExactSide + x;
            if (x + 1 < kExactSide) {
                out[n][0] = i;
                out[n][1] = i + 1;
                ++n;
            }
            if (y + 1 < kExactSide) {
                out[n][0] = i;
                out[n][1] = i + kExactSide;
                ++n;
            }
        }
    }
    return n;
}

// Tensor product of the lattice's per-site spinors (bit 0 of the index
// is site 0; bit set = down).
inline SpinState16 exact_from_product(const SpinLattice& l) {
    SpinState16 s;
    s.c.resize(kExactDim);
    for (std::size_t m = 0; m < kExactDim; ++m) {
        std::complex<double> a{1.0, 0.0};
        for (int i = 0; i < kExactSites; ++i) {
            a *= ((m >> i) & 1u) != 0
                     ? l.s[static_cast<std::size_t>(i)].dn
                     : l.s[static_cast<std::size_t>(i)].up;
        }
        s.c[m] = a;
    }
    return s;
}

// Reduced per-site Bloch vector <sigma_i> (same up/down convention as
// the single spinor's bloch_vector).
inline void exact_site_bloch(const SpinState16& s, int site, double* x,
                             double* y, double* z) {
    const std::size_t b = std::size_t{1} << site;
    std::complex<double> cr{};
    double zz = 0.0;
    for (std::size_t m = 0; m < kExactDim; ++m) {
        if ((m & b) != 0) {
            continue;
        }
        cr += std::conj(s.c[m]) * s.c[m | b];
        zz += std::norm(s.c[m]) - std::norm(s.c[m | b]);
    }
    *x = 2.0 * cr.real();
    *y = 2.0 * cr.imag();
    *z = zz;
}

// The 2x2 single-site rotation matrix U = exp(-i angle/2 n.sigma), the
// ONE source shared by the CPU evolution, the GPU kernel UBO, and the
// vkcheck oracle.
struct SiteGate {
    std::complex<double> a00, a01, a10, a11;
};
inline SiteGate site_gate_matrix(double nx, double ny, double nz,
                                 double angle) {
    const double c = std::cos(0.5 * angle);
    const double sn = std::sin(0.5 * angle);
    const std::complex<double> i{0.0, 1.0};
    return SiteGate{c - i * sn * nz, -i * sn * (nx - i * ny),
                    -i * sn * (nx + i * ny), c + i * sn * nz};
}

// The bond-gate coefficients of exp(+i theta sigma_i.sigma_j): parallel
// phase, and the antiparallel 2x2 (diag, off). Shared source.
struct BondGate {
    std::complex<double> phase, diag, off;
};
inline BondGate bond_gate_params(double theta) {
    const std::complex<double> em{std::cos(theta), -std::sin(theta)};
    return BondGate{std::complex<double>{std::cos(theta), std::sin(theta)},
                    em * std::cos(2.0 * theta),
                    em * std::complex<double>{0.0, std::sin(2.0 * theta)}};
}

// Single-site basis rotation U_i = exp(-i angle/2 n.sigma).
inline void exact_site_rotate(SpinState16& s, int site, double nx,
                              double ny, double nz, double angle) {
    const std::size_t b = std::size_t{1} << site;
    const SiteGate g = site_gate_matrix(nx, ny, nz, angle);
    const std::complex<double> a00 = g.a00;
    const std::complex<double> a01 = g.a01;
    const std::complex<double> a10 = g.a10;
    const std::complex<double> a11 = g.a11;
    // Disjoint pairs (m, m|b): parallel over m with the bit-set skip.
    parallel_for(static_cast<int>(kExactDim), [&](int mi) {
        const std::size_t m = static_cast<std::size_t>(mi);
        if ((m & b) != 0) {
            return;
        }
        const std::complex<double> up = s.c[m];
        const std::complex<double> dn = s.c[m | b];
        s.c[m] = a00 * up + a01 * dn;
        s.c[m | b] = a10 * up + a11 * dn;
    });
}

// One bond gate exp(+i theta sigma_i.sigma_j), exact via
// sigma.sigma = 2 SWAP - 1: parallel pairs pick up e^{+i theta}; the
// antiparallel block mixes with e^{-i theta}(cos 2theta + i sin 2theta P).
inline void exact_bond_gate(SpinState16& s, int si, int sj,
                            double theta) {
    const std::size_t bi = std::size_t{1} << si;
    const std::size_t bj = std::size_t{1} << sj;
    const BondGate g = bond_gate_params(theta);
    const std::complex<double> ph_par = g.phase;
    const std::complex<double> diag = g.diag;
    const std::complex<double> off = g.off;
    // Each m is touched by exactly one branch and the touched index
    // pairs are disjoint: parallel over m.
    parallel_for(static_cast<int>(kExactDim), [&](int mi) {
        const std::size_t m = static_cast<std::size_t>(mi);
        const bool i_dn = (m & bi) != 0;
        const bool j_dn = (m & bj) != 0;
        if (i_dn == j_dn) {
            if (!i_dn) {  // visit each parallel pair once via up-up
                s.c[m] *= ph_par;
                s.c[m | bi | bj] *= ph_par;
            }
            return;
        }
        if (i_dn) {
            return;  // handle the antiparallel pair from (up_i, dn_j)
        }
        const std::size_t m_ud = m;              // i up, j down
        const std::size_t m_du = (m ^ bi) ^ bj;  // i down, j up
        const std::complex<double> a = s.c[m_ud];
        const std::complex<double> bamp = s.c[m_du];
        s.c[m_ud] = diag * a + off * bamp;
        s.c[m_du] = off * a + diag * bamp;
    });
}

// One Strang step: half field, bonds forward at dt/2, bonds REVERSED at
// dt/2 (the bonds share sites, so the palindrome keeps second order),
// half field.
inline void exact_step(SpinState16& s, double bx, double by, double bz,
                       double j, double dt) {
    const double bmag = std::sqrt(bx * bx + by * by + bz * bz);
    int bonds[2 * kExactSites][2];
    const int nb = exact_bonds(bonds);
    if (bmag > 0.0) {
        for (int i = 0; i < kExactSites; ++i) {
            exact_site_rotate(s, i, bx / bmag, by / bmag, bz / bmag,
                              bmag * 0.5 * dt);
        }
    }
    for (int k = 0; k < nb; ++k) {
        exact_bond_gate(s, bonds[k][0], bonds[k][1], 0.5 * j * dt);
    }
    for (int k = nb - 1; k >= 0; --k) {
        exact_bond_gate(s, bonds[k][0], bonds[k][1], 0.5 * j * dt);
    }
    if (bmag > 0.0) {
        for (int i = 0; i < kExactSites; ++i) {
            exact_site_rotate(s, i, bx / bmag, by / bmag, bz / bmag,
                              bmag * 0.5 * dt);
        }
    }
}

// <H> = sum_i (1/2) B.<sigma_i> - J sum_bonds <sigma_i.sigma_j>, the
// bond expectation via <sigma.sigma> = 2 <SWAP> - 1.
inline double exact_energy(const SpinState16& s, double bx, double by,
                           double bz, double j) {
    double e = 0.0;
    for (int i = 0; i < kExactSites; ++i) {
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        exact_site_bloch(s, i, &x, &y, &z);
        e += 0.5 * (bx * x + by * y + bz * z);
    }
    int bonds[2 * kExactSites][2];
    const int nb = exact_bonds(bonds);
    for (int k = 0; k < nb; ++k) {
        const std::size_t bi = std::size_t{1} << bonds[k][0];
        const std::size_t bj = std::size_t{1} << bonds[k][1];
        double swp = 0.0;
        for (std::size_t m = 0; m < kExactDim; ++m) {
            const bool i_dn = (m & bi) != 0;
            const bool j_dn = (m & bj) != 0;
            if (i_dn == j_dn) {
                swp += std::norm(s.c[m]);
            } else {
                swp += (std::conj(s.c[m]) * s.c[(m ^ bi) ^ bj]).real();
            }
        }
        e -= j * (2.0 * swp - 1.0);
    }
    return e;
}

// Born-measure site i along +-z: collapse + renormalize, return +-1.
inline int exact_measure_z(SpinState16& s, int site, double u) {
    const std::size_t b = std::size_t{1} << site;
    double p_up = 0.0;
    for (std::size_t m = 0; m < kExactDim; ++m) {
        if ((m & b) == 0) {
            p_up += std::norm(s.c[m]);
        }
    }
    const int outcome = u < p_up ? +1 : -1;
    double kept = 0.0;
    for (std::size_t m = 0; m < kExactDim; ++m) {
        const bool dn = (m & b) != 0;
        if (dn == (outcome > 0)) {
            s.c[m] = 0.0;
        } else {
            kept += std::norm(s.c[m]);
        }
    }
    if (kept > 0.0) {
        const double inv = 1.0 / std::sqrt(kept);
        for (auto& z : s.c) {
            z *= inv;
        }
    }
    return outcome;
}

}  // namespace ses
