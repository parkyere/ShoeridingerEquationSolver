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

// Sparse matvec out = H psi for H = 1/2 sum_i B.sigma_i - J sum_bonds
// sigma_i.sigma_j: each amplitude gathers 16 single-flip (field) + 24
// double-flip (bond off-diagonal) + diagonal terms. <psi|H psi> == exact_energy.
inline void hamiltonian_apply(SpinState16& out, const SpinState16& in,
                              double bx, double by, double bz, double j) {
    out.c.resize(kExactDim);
    int bonds[2 * kExactSites][2];
    const int nb = exact_bonds(bonds);
    const std::complex<double> I{0.0, 1.0};
    parallel_for(static_cast<int>(kExactDim), [&](int mi) {
        const std::size_t m = static_cast<std::size_t>(mi);
        std::complex<double> acc{0.0, 0.0};
        for (int i = 0; i < kExactSites; ++i) {
            const std::size_t bi = std::size_t{1} << i;
            const bool up = (m & bi) == 0;
            const std::complex<double> flip = in.c[m ^ bi];
            acc += 0.5 * (bx * flip + by * (up ? -I : I) * flip +
                          bz * (up ? 1.0 : -1.0) * in.c[m]);
        }
        for (int k = 0; k < nb; ++k) {
            const std::size_t bi = std::size_t{1} << bonds[k][0];
            const std::size_t bj = std::size_t{1} << bonds[k][1];
            const bool si = (m & bi) != 0;
            const bool sj = (m & bj) != 0;
            acc += -j * ((si == sj ? 1.0 : -1.0) * in.c[m] +
                         (si == sj ? std::complex<double>{0.0, 0.0}
                                   : 2.0 * in.c[(m ^ bi) ^ bj]));
        }
        out.c[m] = acc;
    });
}

// Spectral half-width bound |H| <= 8|B| + 72|J| (field 1/2*16*|B|, bonds
// |sigma.sigma|=3 over 24 bonds), so H/R has spectrum in [-1,1].
inline double hamiltonian_spectral_bound(double bx, double by, double bz,
                                         double j) {
    return 8.0 * std::sqrt(bx * bx + by * by + bz * bz) +
           72.0 * std::abs(j) + 1e-12;
}

// Exact time-independent propagator exp(-i H dt) via Chebyshev expansion:
// exp(-i alpha x) = sum_k (2-d_k0)(-i)^k J_k(alpha) T_k(x), x = H/R, alpha = R dt,
// T_{k+1} = 2 x T_k - T_{k-1}. Spectrally accurate; the canonical propagator for
// fixed H (and a high-accuracy oracle for the Trotter engine).
inline void chebyshev_evolve(SpinState16& psi, double bx, double by,
                             double bz, double j, double dt) {
    const double R = hamiltonian_spectral_bound(bx, by, bz, j);
    const double alpha = R * dt;
    const int K = static_cast<int>(std::ceil(alpha)) + 24;
    auto pow_minus_i = [](int k) -> std::complex<double> {
        switch (k & 3) {
            case 0: return {1.0, 0.0};
            case 1: return {0.0, -1.0};
            case 2: return {-1.0, 0.0};
            default: return {0.0, 1.0};
        }
    };
    auto coeff = [&](int k) {
        return (k == 0 ? 1.0 : 2.0) * pow_minus_i(k) *
               std::cyl_bessel_j(static_cast<double>(k), alpha);
    };
    SpinState16 t0 = psi;  // T_0 = psi
    SpinState16 t1;
    hamiltonian_apply(t1, psi, bx, by, bz, j);  // T_1 = (H/R) psi
    for (auto& z : t1.c) {
        z /= R;
    }
    SpinState16 res;
    res.c.resize(kExactDim);
    const std::complex<double> c0 = coeff(0);
    const std::complex<double> c1 = coeff(1);
    for (std::size_t m = 0; m < kExactDim; ++m) {
        res.c[m] = c0 * t0.c[m] + c1 * t1.c[m];
    }
    SpinState16 tn;
    for (int k = 2; k <= K; ++k) {
        hamiltonian_apply(tn, t1, bx, by, bz, j);  // tn = (H/R) T_{k-1}
        const std::complex<double> ck = coeff(k);
        for (std::size_t m = 0; m < kExactDim; ++m) {
            tn.c[m] = 2.0 * tn.c[m] / R - t0.c[m];  // T_k
            res.c[m] += ck * tn.c[m];
        }
        t0.c.swap(t1.c);
        t1.c.swap(tn.c);
    }
    psi.c.swap(res.c);
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

// Embed a gate matrix on `src` qubits into the `dst` window (src subset of dst,
// both ascending); identity on dst\src. Row-major (1<<|dst|)^2.
inline std::vector<std::complex<double>> expand_matrix(
    const std::vector<std::complex<double>>& mat, const std::vector<int>& src,
    const std::vector<int>& dst) {
    const std::size_t ks = src.size();
    const std::size_t ds = std::size_t{1} << ks;
    const std::size_t dd = std::size_t{1} << dst.size();
    std::vector<int> pos(ks, 0);  // dst-bit position of each src qubit
    for (std::size_t q = 0; q < ks; ++q) {
        for (std::size_t p = 0; p < dst.size(); ++p) {
            if (dst[p] == src[q]) {
                pos[q] = static_cast<int>(p);
                break;
            }
        }
    }
    std::size_t srcmask = 0;
    for (int p : pos) {
        srcmask |= std::size_t{1} << p;
    }
    std::vector<std::complex<double>> e(dd * dd, std::complex<double>{0.0, 0.0});
    for (std::size_t r = 0; r < dd; ++r) {
        for (std::size_t c = 0; c < dd; ++c) {
            if ((r & ~srcmask) != (c & ~srcmask)) {
                continue;  // identity off the src qubits
            }
            std::size_t rs = 0, cs = 0;
            for (std::size_t q = 0; q < ks; ++q) {
                if ((r >> pos[q]) & 1u) rs |= std::size_t{1} << q;
                if ((c >> pos[q]) & 1u) cs |= std::size_t{1} << q;
            }
            e[r * dd + c] = mat[rs * ds + cs];
        }
    }
    return e;
}

inline std::vector<std::complex<double>> mat_mul(
    const std::vector<std::complex<double>>& A,
    const std::vector<std::complex<double>>& B, std::size_t d) {
    std::vector<std::complex<double>> C(d * d, std::complex<double>{0.0, 0.0});
    for (std::size_t i = 0; i < d; ++i) {
        for (std::size_t k = 0; k < d; ++k) {
            const std::complex<double> a = A[i * d + k];
            if (a.real() == 0.0 && a.imag() == 0.0) {
                continue;
            }
            for (std::size_t jc = 0; jc < d; ++jc) {
                C[i * d + jc] += a * B[k * d + jc];
            }
        }
    }
    return C;
}

inline std::vector<int> qubit_union(const std::vector<int>& a,
                                    const std::vector<int>& b) {
    std::vector<int> u = a;
    for (int q : b) {
        if (std::find(u.begin(), u.end(), q) == u.end()) {
            u.push_back(q);
        }
    }
    std::sort(u.begin(), u.end());
    return u;
}

// Greedily merge consecutive gates whose combined support fits kmax qubits into
// one dense fused gate (order preserved: later gate left-multiplies). Amortized
// once per parameter change; the per-frame cost is then #fused passes, not #gates.
inline std::vector<FusedGate> fuse_gates(const std::vector<FusedGate>& gs,
                                         int kmax) {
    std::vector<FusedGate> out;
    FusedGate cur;
    bool have = false;
    for (const FusedGate& g : gs) {
        if (!have) {
            cur = g;
            have = true;
            continue;
        }
        const std::vector<int> w = qubit_union(cur.qubits, g.qubits);
        if (static_cast<int>(w.size()) <= kmax) {
            const std::size_t dd = std::size_t{1} << w.size();
            const std::vector<std::complex<double>> e1 =
                expand_matrix(cur.mat, cur.qubits, w);
            const std::vector<std::complex<double>> e2 =
                expand_matrix(g.mat, g.qubits, w);
            cur.qubits = w;
            cur.mat = mat_mul(e2, e1, dd);  // g applied after cur
        } else {
            out.push_back(cur);
            cur = g;
        }
    }
    if (have) {
        out.push_back(cur);
    }
    return out;
}

// ---- Qubit reordering (Stage 3: the local/global relabel mechanism) ----
// A qubit permutation is a bijection of the 16-bit index. Reordering brings the
// qubits a gate touches into a contiguous low window (coalesced access, no
// fusion-window overflow) -- the single-GPU form of Intel-QS / cuStateVec's
// local/global split. The permutation itself carries no physics; it is undone
// (or tracked) so the evolution is unchanged.

// Lattice transpose (x,y)->(y,x): an involution that turns vertical bonds
// (bit s <-> s+4) into horizontal ones (bit-adjacent) and vice versa.
inline std::vector<int> lattice_transpose_perm() {
    std::vector<int> p(kExactSites);
    for (int y = 0; y < kExactSide; ++y) {
        for (int x = 0; x < kExactSide; ++x) {
            p[static_cast<std::size_t>(y * kExactSide + x)] =
                x * kExactSide + y;
        }
    }
    return p;
}

// Relabel qubits so new qubit b carries old qubit perm[b]: out[j] = in[i] where
// i sets old bit perm[b] iff new index j sets bit b. Applying a gate on new
// qubits (a,b) then undoing the relabel == the gate on old qubits (perm[a],perm[b]).
inline SpinState16 permute_qubits(const SpinState16& s,
                                  const std::vector<int>& perm) {
    SpinState16 o;
    o.c.resize(kExactDim);
    for (std::size_t j = 0; j < kExactDim; ++j) {
        std::size_t i = 0;
        for (int b = 0; b < kExactSites; ++b) {
            if ((j >> b) & 1u) {
                i |= std::size_t{1} << perm[static_cast<std::size_t>(b)];
            }
        }
        o.c[j] = s.c[i];
    }
    return o;
}

}  // namespace ses
