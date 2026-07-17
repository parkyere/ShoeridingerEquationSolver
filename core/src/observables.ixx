module;
#include <complex>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <cstdint>
export module ses.observables;
export import ses.spectral;
export import ses.vec;
export import ses.fft;
export import ses.field;


// Expectation values over a Field1D. All observables are scale-invariant
// (they divide by the discrete norm), so they are valid on unnormalized
// fields as well.


export namespace ses {

// Degenerate-input guards. An empty or fully-absorbed field has zero total
// weight, so num/den is 0/0 -> NaN, which would poison the title readout and
// the Larmor path. Return 0 instead. And a near-single-cell state can dip
// <x^2> - <x>^2 slightly negative from FP cancellation -> clamp before sqrt.
// Both are no-ops (bitwise) whenever den > 0 and the variance is positive.
inline double obs_ratio(double num, double den) noexcept {
    return den > 0.0 ? num / den : 0.0;
}
inline double obs_sigma(double second_moment, double mean) noexcept {
    return std::sqrt(std::max(0.0, second_moment - mean * mean));
}

// <x> = sum x_i |psi_i|^2 / sum |psi_i|^2   (grid weight h cancels)
inline double mean_position(const Field1D& f) noexcept {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double w = std::norm(f[i]);
        num += f.grid().coord(i) * w;
        den += w;
    }
    return obs_ratio(num, den);
}

// sigma_x = sqrt(<x^2> - <x>^2)
inline double sigma_x(const Field1D& f) noexcept {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double x = f.grid().coord(i);
        const double w = std::norm(f[i]);
        num += x * x * w;
        den += w;
    }
    return obs_sigma(obs_ratio(num, den), mean_position(f));
}

// <p> = sum k_j |phi_j|^2 / sum |phi_j|^2 with phi = fft(psi)  (weights cancel)
inline double mean_momentum(const Field1D& f) {
    std::vector<std::complex<double>> phi = f.data();
    fft(phi);
    const std::vector<double> k = wavenumbers(f.grid());
    double num = 0.0;
    double den = 0.0;
    for (std::size_t j = 0; j < phi.size(); ++j) {
        const double w = std::norm(phi[j]);
        num += k[j] * w;
        den += w;
    }
    return obs_ratio(num, den);
}

// <H> = <T> + <V>: kinetic average in k-space (T = k^2/2), potential average
// in real space. Both averages are scale-invariant, so the Parseval factor
// between the two representations cancels within each term.
inline double mean_energy(const Field1D& f, const std::vector<double>& potential) {
    double num_v = 0.0;
    double den_x = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double w = std::norm(f[i]);
        num_v += potential[static_cast<std::size_t>(i)] * w;
        den_x += w;
    }

    std::vector<std::complex<double>> phi = f.data();
    fft(phi);
    const std::vector<double> k = wavenumbers(f.grid());
    double num_t = 0.0;
    double den_k = 0.0;
    for (std::size_t j = 0; j < phi.size(); ++j) {
        const double w = std::norm(phi[j]);
        num_t += 0.5 * k[j] * k[j] * w;
        den_k += w;
    }

    return obs_ratio(num_v, den_x) + obs_ratio(num_t, den_k);
}

// Absolute probability content of the half-open interval [a, b):
//     P = sum_{a <= x_i < b} |psi_i|^2 h
// Deliberately NOT scale-invariant (unlike the observables above): the
// tunneling readout T = P(right of barrier) is measured against the initial
// unit norm, so flux removed by the absorbing mask must reduce -- never
// inflate -- the report. Whole-box total equals norm_sq.
inline double probability_in_range(const Field1D& f, double a, double b) noexcept {
    double acc = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        const double x = f.grid().coord(i);
        if (x >= a && x < b) {
            acc += std::norm(f[i]);
        }
    }
    return acc * f.grid().spacing();
}

// ---- 3D observables (per-axis, scale-invariant) ----

inline Vec3d mean_position(const Field3D& f) noexcept {
    const Grid3D& g = f.grid();
    Vec3d num{};
    double den = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(f(i, j, k));
                num.x += g.x.coord(i) * w;
                num.y += g.y.coord(j) * w;
                num.z += g.z.coord(k) * w;
                den += w;
            }
        }
    }
    return Vec3d{obs_ratio(num.x, den), obs_ratio(num.y, den),
                 obs_ratio(num.z, den)};
}

inline Vec3d sigma_position(const Field3D& f) noexcept {
    const Grid3D& g = f.grid();
    Vec3d num{};
    double den = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(f(i, j, k));
                const double x = g.x.coord(i);
                const double y = g.y.coord(j);
                const double z = g.z.coord(k);
                num.x += x * x * w;
                num.y += y * y * w;
                num.z += z * z * w;
                den += w;
            }
        }
    }
    const Vec3d m = mean_position(f);
    return Vec3d{obs_sigma(obs_ratio(num.x, den), m.x),
                 obs_sigma(obs_ratio(num.y, den), m.y),
                 obs_sigma(obs_ratio(num.z, den), m.z)};
}

inline Vec3d mean_momentum(const Field3D& f) {
    Field3D phi = f;
    fft(phi);
    const Grid3D& g = f.grid();
    const std::vector<double> kx = wavenumbers(g.x);
    const std::vector<double> ky = wavenumbers(g.y);
    const std::vector<double> kz = wavenumbers(g.z);
    Vec3d num{};
    double den = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(phi(i, j, k));
                num.x += kx[static_cast<std::size_t>(i)] * w;
                num.y += ky[static_cast<std::size_t>(j)] * w;
                num.z += kz[static_cast<std::size_t>(k)] * w;
                den += w;
            }
        }
    }
    return Vec3d{obs_ratio(num.x, den), obs_ratio(num.y, den),
                 obs_ratio(num.z, den)};
}

inline double mean_energy(const Field3D& f, const std::vector<double>& potential) {
    double num_v = 0.0;
    double den_x = 0.0;
    for (std::size_t i = 0; i < f.data().size(); ++i) {
        const double w = std::norm(f.data()[i]);
        num_v += potential[i] * w;
        den_x += w;
    }

    Field3D phi = f;
    fft(phi);
    const Grid3D& g = f.grid();
    const std::vector<double> kx = wavenumbers(g.x);
    const std::vector<double> ky = wavenumbers(g.y);
    const std::vector<double> kz = wavenumbers(g.z);
    double num_t = 0.0;
    double den_k = 0.0;
    for (int k = 0; k < g.z.n; ++k) {
        for (int j = 0; j < g.y.n; ++j) {
            for (int i = 0; i < g.x.n; ++i) {
                const double w = std::norm(phi(i, j, k));
                const double kxx = kx[static_cast<std::size_t>(i)];
                const double kyy = ky[static_cast<std::size_t>(j)];
                const double kzz = kz[static_cast<std::size_t>(k)];
                num_t += 0.5 * (kxx * kxx + kyy * kyy + kzz * kzz) * w;
                den_k += w;
            }
        }
    }

    return obs_ratio(num_v, den_x) + obs_ratio(num_t, den_k);
}

}  // namespace ses
