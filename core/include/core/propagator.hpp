#pragma once

// Split-operator (Fourier) propagator for the 1D TDSE in atomic units:
//     i d(psi)/dt = H psi,   H = -1/2 d^2/dx^2 + V(x)
// One step of exp(-i H dt) via Strang splitting:
//     psi <- e^{-i V dt/2} . IFFT . e^{-i k^2 dt/2} . FFT . e^{-i V dt/2} psi
// Unitary by construction (all factors are pure phases), O(dt^2) splitting
// error for V != 0, and time-EXACT for the free particle.
//
// Phase factors are precomputed once per (grid, potential, dt).

#include <core/complex.hpp>
#include <core/fft.hpp>
#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/spectral.hpp>

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ses {

class SplitOperator1D {
public:
    SplitOperator1D(const Grid1D& g, const std::vector<double>& potential, double dt)
        : dt_(dt) {
        assert(static_cast<int>(potential.size()) == g.n);
        const std::size_t n = potential.size();

        half_v_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double th = -0.5 * potential[i] * dt;
            half_v_[i] = Complex<double>{std::cos(th), std::sin(th)};
        }

        const std::vector<double> k = wavenumbers(g);
        kinetic_.resize(n);
        for (std::size_t j = 0; j < n; ++j) {
            const double th = -0.5 * k[j] * k[j] * dt;
            kinetic_[j] = Complex<double>{std::cos(th), std::sin(th)};
        }
    }

    double dt() const { return dt_; }

    void step(Field1D& psi, int nsteps = 1) const {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            apply_phase(half_v_, psi.data());
            fft(psi.data());
            apply_phase(kinetic_, psi.data());
            ifft(psi.data());
            apply_phase(half_v_, psi.data());
        }
    }

private:
    static void apply_phase(const std::vector<Complex<double>>& phase,
                            std::vector<Complex<double>>& a) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = a[i] * phase[i];
        }
    }

    double dt_;
    std::vector<Complex<double>> half_v_;   // e^{-i V dt/2} per grid point
    std::vector<Complex<double>> kinetic_;  // e^{-i k^2 dt/2} per FFT bin
};

// 3D split-operator: identical structure, kinetic phase over the 3D k-grid.
class SplitOperator3D {
public:
    SplitOperator3D(const Grid3D& g, const std::vector<double>& potential, double dt)
        : dt_(dt) {
        assert(static_cast<int>(potential.size()) == g.size());
        const std::size_t n = potential.size();

        half_v_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double th = -0.5 * potential[i] * dt;
            half_v_[i] = Complex<double>{std::cos(th), std::sin(th)};
        }

        const std::vector<double> kx = wavenumbers(g.x);
        const std::vector<double> ky = wavenumbers(g.y);
        const std::vector<double> kz = wavenumbers(g.z);
        kinetic_.resize(n);
        for (int k = 0; k < g.z.n; ++k) {
            for (int j = 0; j < g.y.n; ++j) {
                for (int i = 0; i < g.x.n; ++i) {
                    const double kxx = kx[static_cast<std::size_t>(i)];
                    const double kyy = ky[static_cast<std::size_t>(j)];
                    const double kzz = kz[static_cast<std::size_t>(k)];
                    const double th = -0.5 * (kxx * kxx + kyy * kyy + kzz * kzz) * dt;
                    kinetic_[static_cast<std::size_t>(g.flat(i, j, k))] =
                        Complex<double>{std::cos(th), std::sin(th)};
                }
            }
        }
    }

    double dt() const { return dt_; }

    // Read access to the phase tables so the GPU engine consumes the TESTED
    // tables instead of re-deriving them.
    const std::vector<Complex<double>>& half_potential_phase() const { return half_v_; }
    const std::vector<Complex<double>>& kinetic_phase() const { return kinetic_; }

    void step(Field3D& psi, int nsteps = 1) const {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            apply_phase(half_v_, psi.data());
            fft(psi);
            apply_phase(kinetic_, psi.data());
            ifft(psi);
            apply_phase(half_v_, psi.data());
        }
    }

private:
    // Elementwise (disjoint) multiply: threaded result is bitwise identical.
    static void apply_phase(const std::vector<Complex<double>>& phase,
                            std::vector<Complex<double>>& a) {
        const std::int64_t n = static_cast<std::int64_t>(a.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t i = 0; i < n; ++i) {
            a[static_cast<std::size_t>(i)] =
                a[static_cast<std::size_t>(i)] * phase[static_cast<std::size_t>(i)];
        }
    }

    double dt_;
    std::vector<Complex<double>> half_v_;
    std::vector<Complex<double>> kinetic_;
};

}  // namespace ses
