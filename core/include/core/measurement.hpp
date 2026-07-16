#pragma once

// Soft position measurement (Gaussian POVM): collapse to a Gaussian packet,
// not a delta. Randomness stays OUT of core: callers supply u in [0,1) and
// the samplers invert a discrete CDF in flat-index order (the uniform
// cell-volume factor cancels). POVM consistency: the outcome density for the
// Kraus mask e^{-(r-c)^2/(4 s^2)} is |psi|^2 BLURRED by a Gaussian of std
// sigma_m (E_c = M_c^dag M_c), not raw |psi|^2 -- see sample_povm_index.

#include <core/complex.hpp>
#include <core/field.hpp>
import ses.grid;
import ses.vec;

#include <cmath>
#include <cstddef>
#include <vector>

namespace ses {

// Projective energy measurement over populations P_n = |<phi_n|psi>|^2.
// The tracked manifold is incomplete, so sum(P) <= 1: returns the collapsed
// index n (project psi onto phi_n), or -1 for the 1 - sum(P) deficit
// (continuum / untracked outcome: the caller projects the manifold OUT,
// psi <- (1 - P)|psi>, so bound populations do not survive the verdict).
inline int sample_energy_eigenstate(const std::vector<double>& populations, double u) noexcept {
    double cum = 0.0;
    for (std::size_t n = 0; n < populations.size(); ++n) {
        cum += populations[n];
        if (u < cum) {
            return static_cast<int>(n);
        }
    }
    return -1;  // fell into the 1 - sum(P) deficit: continuum / untracked
}

// First flat index whose cumulative probability exceeds u * total.
inline int sample_collapse_index(const Field3D& psi, double u) noexcept {
    double total = 0.0;
    for (const Complex<double>& z : psi.data()) {
        total += norm_sq(z);
    }
    const double target = u * total;
    double cum = 0.0;
    const std::size_t n = psi.data().size();
    for (std::size_t i = 0; i < n; ++i) {
        cum += norm_sq(psi.data()[i]);
        if (cum > target) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(n - 1);  // u ~ 1 with rounding: last occupied cell
}

// |psi|^2 convolved per axis with a normalized Gaussian of std sigma_m
// (truncated at 4 sigma, periodic wrap -- the grid's FFT topology): the
// Born-rule outcome density of the Gaussian POVM below.
inline std::vector<double> povm_outcome_density(const Field3D& psi,
                                                double sigma_m) {
    const Grid3D& g = psi.grid();
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    std::vector<double> d(psi.data().size());
    for (std::size_t i = 0; i < d.size(); ++i) {
        d[i] = norm_sq(psi.data()[i]);
    }
    std::vector<double> tmp(d.size());
    const Grid1D* axes[3] = {&g.x, &g.y, &g.z};
    const int strides[3] = {1, nx, nx * ny};
    for (int a = 0; a < 3; ++a) {
        const int n = axes[a]->n;
        const double h = axes[a]->spacing();
        const int radius = static_cast<int>(std::ceil(4.0 * sigma_m / h));
        std::vector<double> w(static_cast<std::size_t>(2 * radius + 1));
        double sum = 0.0;
        for (int t = -radius; t <= radius; ++t) {
            const double x = t * h / sigma_m;
            sum += w[static_cast<std::size_t>(t + radius)] = std::exp(-0.5 * x * x);
        }
        for (double& v : w) {
            v /= sum;
        }
        const int stride = strides[a];
        const int lines = nx * ny * nz / n;
#pragma omp parallel for schedule(static)
        for (int line = 0; line < lines; ++line) {
            // Base index of this axis-a line: reinsert the axis dimension
            // into the flattened remaining-dims counter.
            const int base = line % stride + (line / stride) * stride * n;
            for (int p = 0; p < n; ++p) {
                double acc = 0.0;
                for (int t = -radius; t <= radius; ++t) {
                    const int q = (p + t % n + n) % n;
                    acc += w[static_cast<std::size_t>(t + radius)] *
                           d[static_cast<std::size_t>(base + q * stride)];
                }
                tmp[static_cast<std::size_t>(base + p * stride)] = acc;
            }
        }
        d.swap(tmp);
    }
    return d;
}

// L_z bookkeeping for one real-harmonic pair. Convention (harmonics.hpp,
// pinned by tests): Y_{l,+|m|} ~ cos(m phi), Y_{l,-|m|} ~ sin(m phi), so
// |l, +-m> = (|cos> +- i |sin>)/sqrt(2) and the signed-m amplitudes of
// psi = c_cos|cos> + c_sin|sin> are a_+- = (c_cos -+ i c_sin)/sqrt(2).
struct SignedM {
    Complex<double> plus;
    Complex<double> minus;
};
inline SignedM signed_m_amplitudes(Complex<double> c_cos,
                                   Complex<double> c_sin) noexcept {
    const double inv = 1.0 / std::sqrt(2.0);
    const Complex<double> i{0.0, 1.0};
    return SignedM{inv * (c_cos - i * c_sin), inv * (c_cos + i * c_sin)};
}

// Real-pair coefficients of a kept signed-m component a|l, sign*|m|>:
// cos = a/sqrt(2), sin = sign * i a/sqrt(2). Keeping both outcomes and
// summing reconstructs the original pair (projector completeness).
struct RealPair {
    Complex<double> c_cos;
    Complex<double> c_sin;
};
inline RealPair pair_from_signed_m(Complex<double> a, int sign) noexcept {
    const double inv = 1.0 / std::sqrt(2.0);
    const Complex<double> i{0.0, 1.0};
    return RealPair{inv * a, static_cast<double>(sign) * inv * (i * a)};
}

// First flat index whose cumulative POVM outcome probability exceeds
// u * total: detector-consistent sampling (outcomes CAN land on a node of
// raw |psi|^2 that a sigma_m-resolution detector cannot resolve).
inline int sample_povm_index(const Field3D& psi, double sigma_m, double u) {
    const std::vector<double> d = povm_outcome_density(psi, sigma_m);
    double total = 0.0;
    for (double p : d) {
        total += p;
    }
    const double target = u * total;
    double cum = 0.0;
    for (std::size_t i = 0; i < d.size(); ++i) {
        cum += d[i];
        if (cum > target) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(d.size() - 1);  // u ~ 1 with rounding
}

// psi <- psi * exp(-|r - center|^2 / (4 sigma_m^2)), renormalized. Same
// amplitude convention as gaussian_wavepacket, so Gaussian x Gaussian
// posteriors are analytic.
inline void collapse_wavepacket(Field3D& psi, Vec3d center, double sigma_m) noexcept {
    const Grid3D& g = psi.grid();
    const double inv4s2 = 1.0 / (4.0 * sigma_m * sigma_m);
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double dx = g.x.coord(i) - center.x;
                const double dy = g.y.coord(j) - center.y;
                const double dz = g.z.coord(k) - center.z;
                const double mask = std::exp(-(dx * dx + dy * dy + dz * dz) * inv4s2);
                psi(i, j, k) = mask * psi(i, j, k);
            }
        }
    }
    normalize(psi);
}

}  // namespace ses
