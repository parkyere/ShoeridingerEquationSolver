#pragma once

// Complex scalar field over a Grid1D -- the container for the wavefunction psi.
//
// The discrete norm includes the grid weight:
//     ||psi||^2 = sum_i |psi_i|^2 * h
// so a continuum-normalized function sampled on the grid keeps unit norm.

#include <complex>
import ses.grid;

#include <cmath>
#include <vector>

namespace ses {

class Field1D {
public:
    explicit Field1D(Grid1D grid)
        : grid_(grid), data_(static_cast<std::size_t>(grid.n)) {}

    int size() const noexcept { return grid_.n; }
    const Grid1D& grid() const noexcept { return grid_; }

    std::complex<double>& operator[](int i) noexcept { return data_[static_cast<std::size_t>(i)]; }
    const std::complex<double>& operator[](int i) const noexcept { return data_[static_cast<std::size_t>(i)]; }

    // Raw storage access for in-place spectral transforms (fft/ifft).
    std::vector<std::complex<double>>& data() noexcept { return data_; }
    const std::vector<std::complex<double>>& data() const noexcept { return data_; }

private:
    Grid1D grid_;
    std::vector<std::complex<double>> data_;
};

// ||psi||^2 = sum_i |psi_i|^2 * h
inline double norm_sq(const Field1D& f) noexcept {
    double acc = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        acc += std::norm(f[i]);
    }
    return acc * f.grid().spacing();
}

// Rescale so that ||psi||^2 == 1.
inline void normalize(Field1D& f) noexcept {
    const double inv = 1.0 / std::sqrt(norm_sq(f));
    for (int i = 0; i < f.size(); ++i) {
        f[i] = inv * f[i];
    }
}

// Complex scalar field over a Grid3D (x-fastest flat storage).
class Field3D {
public:
    explicit Field3D(Grid3D grid)
        : grid_(grid), data_(static_cast<std::size_t>(grid.size())) {}

    int size() const noexcept { return grid_.size(); }
    const Grid3D& grid() const noexcept { return grid_; }

    std::complex<double>& operator()(int i, int j, int k) noexcept {
        return data_[static_cast<std::size_t>(grid_.flat(i, j, k))];
    }
    const std::complex<double>& operator()(int i, int j, int k) const noexcept {
        return data_[static_cast<std::size_t>(grid_.flat(i, j, k))];
    }

    std::vector<std::complex<double>>& data() noexcept { return data_; }
    const std::vector<std::complex<double>>& data() const noexcept { return data_; }

private:
    Grid3D grid_;
    std::vector<std::complex<double>> data_;
};

// ||psi||^2 = sum_ijk |psi_ijk|^2 * hx hy hz
inline double norm_sq(const Field3D& f) noexcept {
    double acc = 0.0;
    for (const std::complex<double>& z : f.data()) {
        acc += std::norm(z);
    }
    return acc * f.grid().cell_volume();
}

inline void normalize(Field3D& f) noexcept {
    const double n2 = norm_sq(f);
    if (n2 <= 0.0) {
        return;  // zero field (e.g. a deflated-away seed): 1/0 -> Inf -> NaN
    }
    const double inv = 1.0 / std::sqrt(n2);
    for (std::complex<double>& z : f.data()) {
        z = inv * z;
    }
}

// Discrete inner product <a|b> = sum conj(a_i) b_i * dV. The building block
// of state projections (deflation) and populations |<phi|psi>|^2.
inline std::complex<double> inner_product(const Field3D& a, const Field3D& b) noexcept {
    double re = 0.0;
    double im = 0.0;
    for (std::size_t i = 0; i < a.data().size(); ++i) {
        const std::complex<double> t = std::conj(a.data()[i]) * b.data()[i];
        re += t.real();
        im += t.imag();
    }
    const double dv = a.grid().cell_volume();
    return std::complex<double>{re * dv, im * dv};
}

// |psi|^2 per grid point -- the probability-density scalar field handed to
// visualization (marching cubes, volume rendering). No volume weight: this
// is a density, not an integral.
inline std::vector<double> probability_density(const Field3D& f) {
    std::vector<double> rho(f.data().size());
    for (std::size_t i = 0; i < rho.size(); ++i) {
        rho[i] = std::norm(f.data()[i]);
    }
    return rho;
}

}  // namespace ses
