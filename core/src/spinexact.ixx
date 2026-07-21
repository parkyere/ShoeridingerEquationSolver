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


// EXACT 4x4 Heisenberg: full 2^16 amplitude wavefunction; basis bit i set
// = site i spin-DOWN. H = (1/2) sum_i B.sigma_i - J sum_bonds sigma_i.sigma_j,
// Strang-split, bond gates via sigma.sigma = 2 SWAP - 1.
// CONTRACT: tests/spinexact_test.cpp.


export namespace ses {

inline constexpr int kExactSide = 4;
inline constexpr int kExactSites = 16;
inline constexpr std::size_t kExactDim = 1u << kExactSites;

struct SpinState16 {
    std::vector<std::complex<double>> c;
};

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

// Reduced per-site Bloch vector <sigma_i>.
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

// 2x2 site rotation U = exp(-i angle/2 n.sigma). Shared source: CPU, GPU
// kernel UBO, vkcheck oracle must match.
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

// Coefficients of exp(+i theta sigma_i.sigma_j): parallel phase, antiparallel
// 2x2 (diag, off). Shared source.
struct BondGate {
    std::complex<double> phase, diag, off;
};
inline BondGate bond_gate_params(double theta) {
    const std::complex<double> em{std::cos(theta), -std::sin(theta)};
    return BondGate{std::complex<double>{std::cos(theta), std::sin(theta)},
                    em * std::cos(2.0 * theta),
                    em * std::complex<double>{0.0, std::sin(2.0 * theta)}};
}

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

// One bond gate exp(+i theta sigma_i.sigma_j) via sigma.sigma = 2 SWAP - 1.
inline void exact_bond_gate(SpinState16& s, int si, int sj,
                            double theta) {
    const std::size_t bi = std::size_t{1} << si;
    const std::size_t bj = std::size_t{1} << sj;
    const BondGate g = bond_gate_params(theta);
    const std::complex<double> ph_par = g.phase;
    const std::complex<double> diag = g.diag;
    const std::complex<double> off = g.off;
    // Each m touched by one branch, pairs disjoint -> parallel over m.
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
        const std::size_t m_ud = m;
        const std::size_t m_du = (m ^ bi) ^ bj;
        const std::complex<double> a = s.c[m_ud];
        const std::complex<double> bamp = s.c[m_du];
        s.c[m_ud] = diag * a + off * bamp;
        s.c[m_du] = off * a + diag * bamp;
    });
}

// Strang step; bond sweep reversed on the return (shared sites -> palindrome
// keeps 2nd order).
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

// <H> = (1/2) sum_i B.<sigma_i> - J sum_bonds <sigma_i.sigma_j>;
// <sigma.sigma> = 2 <SWAP> - 1.
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

// Born-measure site i along z; u = uniform draw. Returns +-1.
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

// ---- Gate fusion (Stage 1 of the canonical state-vector path) ----
// A dense gate on up to k qubits: (1<<k)x(1<<k) row-major matrix + ascending
// qubit list. apply_fused = per orbit gather 2^k amplitudes, matmul, scatter --
// the CPU truth the GPU shared-memory-blocked kernel will mirror. Basis bit b of
// the local index selects qubits[b]; local 0 = site up (grid bit clear).
struct FusedGate {
    std::vector<int> qubits;
    std::vector<std::complex<double>> mat;
};

inline std::size_t fused_scatter_index(std::size_t base, std::size_t a,
                                       const std::vector<int>& qubits) {
    std::size_t idx = base;
    for (std::size_t b = 0; b < qubits.size(); ++b) {
        if ((a >> b) & 1u) {
            idx |= std::size_t{1} << qubits[b];
        }
    }
    return idx;
}

inline void apply_fused(SpinState16& s, const FusedGate& g) {
    const std::size_t k = g.qubits.size();
    const std::size_t dk = std::size_t{1} << k;
    std::size_t gmask = 0;
    for (int q : g.qubits) {
        gmask |= std::size_t{1} << q;
    }
    std::vector<std::complex<double>> in(dk), out(dk);
    for (std::size_t base = 0; base < kExactDim; ++base) {
        if ((base & gmask) != 0) {
            continue;  // orbit representatives: all gate bits clear
        }
        for (std::size_t a = 0; a < dk; ++a) {
            in[a] = s.c[fused_scatter_index(base, a, g.qubits)];
        }
        for (std::size_t r = 0; r < dk; ++r) {
            std::complex<double> acc{};
            for (std::size_t c = 0; c < dk; ++c) {
                acc += g.mat[r * dk + c] * in[c];
            }
            out[r] = acc;
        }
        for (std::size_t a = 0; a < dk; ++a) {
            s.c[fused_scatter_index(base, a, g.qubits)] = out[a];
        }
    }
}

// Single-site rotation as a 1-qubit gate (matches exact_site_rotate).
inline FusedGate site_fused_gate(int site, double nx, double ny, double nz,
                                 double angle) {
    const SiteGate u = site_gate_matrix(nx, ny, nz, angle);
    return FusedGate{{site}, {u.a00, u.a01, u.a10, u.a11}};
}

// Bond exp(+i theta sigma.sigma) as a 2-qubit gate (matches exact_bond_gate):
// parallel |00>,|11> -> phase; antiparallel |01>,|10> -> [[diag,off],[off,diag]].
inline FusedGate bond_fused_gate(int si, int sj, double theta) {
    const BondGate b = bond_gate_params(theta);
    const int lo = std::min(si, sj);
    const int hi = std::max(si, sj);
    std::vector<std::complex<double>> m(16, std::complex<double>{0.0, 0.0});
    m[0 * 4 + 0] = b.phase;  // |00>
    m[3 * 4 + 3] = b.phase;  // |11>
    m[1 * 4 + 1] = b.diag;   // antiparallel 2x2
    m[1 * 4 + 2] = b.off;
    m[2 * 4 + 1] = b.off;
    m[2 * 4 + 2] = b.diag;
    return FusedGate{{lo, hi}, std::move(m)};
}

// The exact_step gate list, unfused (site sweeps + palindromic bond sweeps).
inline std::vector<FusedGate> step_gates(double bx, double by, double bz,
                                         double j, double dt) {
    const double bmag = std::sqrt(bx * bx + by * by + bz * bz);
    std::vector<FusedGate> gs;
    int bonds[2 * kExactSites][2];
    const int nb = exact_bonds(bonds);
    const double nx = bmag > 0.0 ? bx / bmag : 0.0;
    const double ny = bmag > 0.0 ? by / bmag : 0.0;
    const double nz = bmag > 0.0 ? bz / bmag : 0.0;
    const double a = bmag * 0.5 * dt;
    if (bmag > 0.0) {
        for (int i = 0; i < kExactSites; ++i) {
            gs.push_back(site_fused_gate(i, nx, ny, nz, a));
        }
    }
    for (int k = 0; k < nb; ++k) {
        gs.push_back(bond_fused_gate(bonds[k][0], bonds[k][1], 0.5 * j * dt));
    }
    for (int k = nb - 1; k >= 0; --k) {
        gs.push_back(bond_fused_gate(bonds[k][0], bonds[k][1], 0.5 * j * dt));
    }
    if (bmag > 0.0) {
        for (int i = 0; i < kExactSites; ++i) {
            gs.push_back(site_fused_gate(i, nx, ny, nz, a));
        }
    }
    return gs;
}

inline void fused_step(SpinState16& s, const std::vector<FusedGate>& gs) {
    for (const FusedGate& g : gs) {
        apply_fused(s, g);
    }
}

}  // namespace ses
