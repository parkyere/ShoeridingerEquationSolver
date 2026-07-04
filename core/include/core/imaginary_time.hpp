#pragma once

// Imaginary-time relaxation to the ground state.
//
// t -> -i tau turns e^{-i H dt} into e^{-H dtau}: every eigencomponent decays
// as e^{-E_n tau}, so with per-step renormalization the state converges to
// the ground state. Same Strang splitting as the real-time propagator, but
// the factors are REAL decay weights e^{-V dtau/2} and e^{-k^2 dtau/2}
// (the flow is not unitary -- renormalization is mandatory, not cosmetic).

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

class ImaginaryTimePropagator1D {
public:
    ImaginaryTimePropagator1D(const Grid1D& g, const std::vector<double>& potential,
                              double dtau) {
        assert(static_cast<int>(potential.size()) == g.n);
        const std::size_t n = potential.size();

        half_v_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            half_v_[i] = std::exp(-0.5 * potential[i] * dtau);
        }

        const std::vector<double> k = wavenumbers(g);
        kinetic_.resize(n);
        for (std::size_t j = 0; j < n; ++j) {
            kinetic_[j] = std::exp(-0.5 * k[j] * k[j] * dtau);
        }
    }

    void relax(Field1D& psi, int nsteps) const {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            apply_weight(half_v_, psi.data());
            fft(psi.data());
            apply_weight(kinetic_, psi.data());
            ifft(psi.data());
            apply_weight(half_v_, psi.data());
            normalize(psi);
        }
    }

private:
    static void apply_weight(const std::vector<double>& weight,
                             std::vector<Complex<double>>& a) {
        for (std::size_t i = 0; i < a.size(); ++i) {
            a[i] = weight[i] * a[i];
        }
    }

    std::vector<double> half_v_;   // e^{-V dtau/2} per grid point
    std::vector<double> kinetic_;  // e^{-k^2 dtau/2} per FFT bin
};

// 3D imaginary-time relaxation: identical structure over the 3D k-grid.
class ImaginaryTimePropagator3D {
public:
    ImaginaryTimePropagator3D(const Grid3D& g, const std::vector<double>& potential,
                              double dtau) {
        assert(static_cast<int>(potential.size()) == g.size());
        const std::size_t n = potential.size();

        half_v_.resize(n);
        for (std::size_t i = 0; i < n; ++i) {
            half_v_[i] = std::exp(-0.5 * potential[i] * dtau);
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
                    kinetic_[static_cast<std::size_t>(g.flat(i, j, k))] =
                        std::exp(-0.5 * (kxx * kxx + kyy * kyy + kzz * kzz) * dtau);
                }
            }
        }
    }

    // Read access to the weight tables so the GPU relax path consumes the
    // TESTED tables instead of re-deriving them (docs/GPU_PLAN.md G7).
    const std::vector<double>& half_potential_weight() const { return half_v_; }
    const std::vector<double>& kinetic_weight() const { return kinetic_; }

    void relax(Field3D& psi, int nsteps) const {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            apply_weight(half_v_, psi.data());
            fft(psi);
            apply_weight(kinetic_, psi.data());
            ifft(psi);
            apply_weight(half_v_, psi.data());
            normalize(psi);
        }
    }

    // Relax within the orthogonal complement of the given lower eigenstates:
    // each step projects them out (Gram-Schmidt deflation), so the flow
    // converges to the NEXT excited state instead of falling back down.
    void relax_deflated(Field3D& psi, const std::vector<const Field3D*>& lower,
                        int nsteps) const {
        assert(psi.data().size() == half_v_.size());
        for (int s = 0; s < nsteps; ++s) {
            apply_weight(half_v_, psi.data());
            fft(psi);
            apply_weight(kinetic_, psi.data());
            ifft(psi);
            apply_weight(half_v_, psi.data());
            for (const Field3D* phi : lower) {
                const Complex<double> c = inner_product(*phi, psi);
                std::vector<Complex<double>>& p = psi.data();
                const std::vector<Complex<double>>& q = phi->data();
                for (std::size_t i = 0; i < p.size(); ++i) {
                    p[i] = p[i] - c * q[i];
                }
            }
            normalize(psi);
        }
    }

private:
    // Elementwise (disjoint) scale: threaded result is bitwise identical.
    static void apply_weight(const std::vector<double>& weight,
                             std::vector<Complex<double>>& a) {
        const std::int64_t n = static_cast<std::int64_t>(a.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (std::int64_t i = 0; i < n; ++i) {
            a[static_cast<std::size_t>(i)] =
                weight[static_cast<std::size_t>(i)] * a[static_cast<std::size_t>(i)];
        }
    }

    std::vector<double> half_v_;
    std::vector<double> kinetic_;
};

}  // namespace ses
