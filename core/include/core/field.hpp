#pragma once

// Complex scalar field over a Grid1D -- the container for the wavefunction psi.
//
// The discrete norm includes the grid weight:
//     ||psi||^2 = sum_i |psi_i|^2 * h
// so a continuum-normalized function sampled on the grid keeps unit norm.

#include <core/complex.hpp>
#include <core/grid.hpp>

#include <cmath>
#include <vector>

namespace ses {

class Field1D {
public:
    explicit Field1D(Grid1D grid)
        : grid_(grid), data_(static_cast<std::size_t>(grid.n)) {}

    int size() const { return grid_.n; }
    const Grid1D& grid() const { return grid_; }

    Complex<double>& operator[](int i) { return data_[static_cast<std::size_t>(i)]; }
    const Complex<double>& operator[](int i) const { return data_[static_cast<std::size_t>(i)]; }

    // Raw storage access for in-place spectral transforms (fft/ifft).
    std::vector<Complex<double>>& data() { return data_; }
    const std::vector<Complex<double>>& data() const { return data_; }

private:
    Grid1D grid_;
    std::vector<Complex<double>> data_;
};

// ||psi||^2 = sum_i |psi_i|^2 * h
inline double norm_sq(const Field1D& f) {
    double acc = 0.0;
    for (int i = 0; i < f.size(); ++i) {
        acc += norm_sq(f[i]);
    }
    return acc * f.grid().spacing();
}

// Rescale so that ||psi||^2 == 1.
inline void normalize(Field1D& f) {
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

    int size() const { return grid_.size(); }
    const Grid3D& grid() const { return grid_; }

    Complex<double>& operator()(int i, int j, int k) {
        return data_[static_cast<std::size_t>(grid_.flat(i, j, k))];
    }
    const Complex<double>& operator()(int i, int j, int k) const {
        return data_[static_cast<std::size_t>(grid_.flat(i, j, k))];
    }

    std::vector<Complex<double>>& data() { return data_; }
    const std::vector<Complex<double>>& data() const { return data_; }

private:
    Grid3D grid_;
    std::vector<Complex<double>> data_;
};

// ||psi||^2 = sum_ijk |psi_ijk|^2 * hx hy hz
inline double norm_sq(const Field3D& f) {
    double acc = 0.0;
    for (const Complex<double>& z : f.data()) {
        acc += norm_sq(z);
    }
    return acc * f.grid().cell_volume();
}

inline void normalize(Field3D& f) {
    const double inv = 1.0 / std::sqrt(norm_sq(f));
    for (Complex<double>& z : f.data()) {
        z = inv * z;
    }
}

}  // namespace ses
