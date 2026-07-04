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

}  // namespace ses
