#pragma once

// Orbital-free projection: every amplitude <n|psi> over the tracked manifold
// with no resident 3-D orbital atlas. Central V factorizes |n> = (u_nl/r) Y_lm,
// so ONE grid pass (independent of the state count) deposits
// g_lm[c][j] = sum_cells W_j(r) Y_lm(cell) psi(cell) dV, then each amplitude
// is a 1-D dot sum_j u_nl[j] g_lm[lm][j]. CPU-double oracle for the GPU deposit
// kernel. CONTRACT: W_j(r) must mirror fill_orbital (core/harmonics.hpp) exactly.

#include <core/complex.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/harmonics.hpp>  // real_spherical_harmonic
#include <core/radial.hpp>     // RadialGrid

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ses {

// A tracked state's generative seed: which radial level (index into the
// caller's u tables) and its angular quantum numbers. NO 3-D orbital.
struct ProjectorState {
    int level;
    int l;
    int m;
};

// Flat index of a real harmonic (l, m): l in [0, l_max], m in [-l, l].
inline int lm_index(int l, int m) { return l * l + (l + m); }
inline int lm_count(int l_max) { return (l_max + 1) * (l_max + 1); }

struct RadialAngularProjection {
    std::vector<Complex<double>> amp;               // <n|psi>, unit-normalized
    std::vector<double> norm2;                      // N_n = sum_j u[j]^2 h (=1 for eigen-u)
    std::vector<std::vector<Complex<double>>> g_lm;  // [lm_count][n_radial], built once
    // amp[n] = (sum_j u[j] g_lm[lm(n)][j]) / sqrt(N_n);
    // the RAW amplitude (== the direct grid inner product) = amp[n] * sqrt(N_n).
};

// Static geometry for the GPU deposit: cell -> radial bin depends only on the
// grid, so it is built once. Cells are counting-sorted by primary bin i0 (CSR
// offsets bin_off) so the GPU runs one workgroup per bin -- deterministic
// gather, no atomics. i0 uses the IDENTICAL fp32 arithmetic as the shader so
// both agree on every cell, including ~fp32-eps boundary straddlers.
struct RadialBinIndex {
    std::vector<std::uint32_t> sorted_cell;  // in-sphere cell flat indices, grouped by i0
    std::vector<std::uint32_t> bin_off;      // CSR offsets, length n_radial+1
};

// The fp32-identical bin key for a cell: -1 if r >= rmax (outside the sphere),
// 0 if r < h (origin segment), else i0 = int(r/h - 1). Free function so the
// GLSL kernel and the CPU sort provably share it.
inline int radial_bin_key(const Grid3D& g, const RadialGrid& rgrid, int i, int j,
                          int k) {
    const float hf = static_cast<float>(rgrid.h());
    const float rmaxf = static_cast<float>(rgrid.rmax);
    const float x = static_cast<float>(g.x.xmin) +
                    static_cast<float>(i) * static_cast<float>(g.x.spacing());
    const float y = static_cast<float>(g.y.xmin) +
                    static_cast<float>(j) * static_cast<float>(g.y.spacing());
    const float z = static_cast<float>(g.z.xmin) +
                    static_cast<float>(k) * static_cast<float>(g.z.spacing());
    const float r = std::sqrt(x * x + y * y + z * z);
    if (r >= rmaxf) {
        return -1;
    }
    if (r < hf) {
        return 0;
    }
    int i0 = static_cast<int>(r / hf - 1.0f);
    if (i0 >= rgrid.n) {
        i0 = rgrid.n - 1;  // guard: r < rmax should already keep i0 <= n-1
    }
    return i0;
}

inline RadialBinIndex build_radial_bin_index(const Grid3D& g, const RadialGrid& rgrid) {
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    const int nr = rgrid.n;
    std::vector<std::uint32_t> counts(static_cast<std::size_t>(nr), 0);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int key = radial_bin_key(g, rgrid, i, j, k);
                if (key >= 0) {
                    ++counts[static_cast<std::size_t>(key)];
                }
            }
        }
    }
    RadialBinIndex out;
    out.bin_off.assign(static_cast<std::size_t>(nr + 1), 0);
    for (int b = 0; b < nr; ++b) {
        out.bin_off[static_cast<std::size_t>(b + 1)] =
            out.bin_off[static_cast<std::size_t>(b)] + counts[static_cast<std::size_t>(b)];
    }
    out.sorted_cell.resize(out.bin_off[static_cast<std::size_t>(nr)]);
    std::vector<std::uint32_t> pos(out.bin_off.begin(), out.bin_off.end() - 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int key = radial_bin_key(g, rgrid, i, j, k);
                if (key >= 0) {
                    const std::uint32_t idx =
                        static_cast<std::uint32_t>(i + nx * (j + ny * k));
                    out.sorted_cell[pos[static_cast<std::size_t>(key)]++] = idx;
                }
            }
        }
    }
    return out;
}

// One grid pass over psi -> g_lm; then a 1-D radial dot per state -> amplitudes.
inline RadialAngularProjection project_radial_angular(
    const Field3D& psi, const RadialGrid& rgrid,
    const std::vector<std::vector<double>>& u_by_level,
    const std::vector<ProjectorState>& states, int l_max = 5) {
    const Grid3D& g = psi.grid();
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    const double h = rgrid.h();
    const double rmax = rgrid.rmax;
    const int nr = rgrid.n;
    const double dV = g.cell_volume();
    const int ncomp = lm_count(l_max);

    RadialAngularProjection out;
    out.g_lm.assign(static_cast<std::size_t>(ncomp),
                    std::vector<Complex<double>>(static_cast<std::size_t>(nr),
                                                 Complex<double>{}));

    // Deposit Y_lm(cell) psi(cell) dV into radial bins (weights mirror fill_orbital).
    for (int k = 0; k < nz; ++k) {
        const double z = g.z.coord(k);
        for (int j = 0; j < ny; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < nx; ++i) {
                const double x = g.x.coord(i);
                const double r = std::sqrt(x * x + y * y + z * z);
                if (r >= rmax) {
                    continue;  // outside the radial box -> no deposit (deficit)
                }
                // Bins/weights, exactly the fill_orbital branches: r < h ->
                // bin 0 weight 1/h; else linear tent /r into i0 and i0+1.
                int b0 = 0;
                int b1 = -1;
                double w0 = 0.0;
                double w1 = 0.0;
                if (r < h) {
                    b0 = 0;
                    w0 = 1.0 / h;
                } else {
                    const double t = r / h - 1.0;  // r_i = (i+1) h
                    const int i0 = static_cast<int>(t);
                    const double frac = t - static_cast<double>(i0);
                    b0 = i0;
                    w0 = (1.0 - frac) / r;
                    if (i0 + 1 < nr) {
                        b1 = i0 + 1;
                        w1 = frac / r;  // outermost node = pinned u(rmax)=0: dropped
                    }
                }
                const Complex<double> pdV = psi(i, j, k) * dV;
                for (int l = 0; l <= l_max; ++l) {
                    for (int m = -l; m <= l; ++m) {
                        const double Y = real_spherical_harmonic(l, m, x, y, z);
                        const Complex<double> contrib = pdV * Y;
                        const std::size_t c = static_cast<std::size_t>(lm_index(l, m));
                        out.g_lm[c][static_cast<std::size_t>(b0)] += contrib * w0;
                        if (b1 >= 0) {
                            out.g_lm[c][static_cast<std::size_t>(b1)] += contrib * w1;
                        }
                    }
                }
            }
        }
    }

    // --- per-state amplitudes: raw = sum_j u[j] g_lm[lm][j]; norm = sum u^2 h ---
    out.amp.assign(states.size(), Complex<double>{});
    out.norm2.assign(states.size(), 0.0);
    for (std::size_t s = 0; s < states.size(); ++s) {
        const ProjectorState& st = states[s];
        const std::vector<double>& u =
            u_by_level[static_cast<std::size_t>(st.level)];
        const std::vector<Complex<double>>& gc =
            out.g_lm[static_cast<std::size_t>(lm_index(st.l, st.m))];
        Complex<double> raw{};
        double n2 = 0.0;
        for (int jr = 0; jr < nr; ++jr) {
            raw += u[static_cast<std::size_t>(jr)] * gc[static_cast<std::size_t>(jr)];
            n2 += u[static_cast<std::size_t>(jr)] * u[static_cast<std::size_t>(jr)] * h;
        }
        out.norm2[s] = n2;
        out.amp[s] = (n2 > 0.0) ? raw * (1.0 / std::sqrt(n2)) : Complex<double>{};
    }
    return out;
}

}  // namespace ses
