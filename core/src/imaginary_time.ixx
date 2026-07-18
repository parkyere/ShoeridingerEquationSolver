module;
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>
export module ses.imaginary_time;
export import ses.grid;
export import ses.spectral;
export import ses.fft;
export import ses.field;
import ses.parallel;


// Imaginary-time relaxation to the ground state: e^{-H dtau} decays each
// eigencomponent as e^{-E_n tau}. Same Strang splitting as the real-time
// propagator but with REAL decay weights e^{-V dtau/2}, e^{-k^2 dtau/2};
// the flow is not unitary -- per-step renormalization is mandatory.


export namespace ses {

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
                             std::vector<std::complex<double>>& a) noexcept {
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
    // TESTED tables instead of re-deriving them.
    const std::vector<double>& half_potential_weight() const noexcept { return half_v_; }
    const std::vector<double>& kinetic_weight() const noexcept { return kinetic_; }

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

    // Gram-Schmidt deflation of `lower` each step -> converges to the next excited state.
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
                const std::complex<double> c = inner_product(*phi, psi);
                std::vector<std::complex<double>>& p = psi.data();
                const std::vector<std::complex<double>>& q = phi->data();
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
                             std::vector<std::complex<double>>& a) noexcept {
        parallel_for(static_cast<int>(a.size()), [&](int i) {
            a[static_cast<std::size_t>(i)] =
                weight[static_cast<std::size_t>(i)] * a[static_cast<std::size_t>(i)];
        });
    }

    std::vector<double> half_v_;
    std::vector<double> kinetic_;
};

}  // namespace ses
